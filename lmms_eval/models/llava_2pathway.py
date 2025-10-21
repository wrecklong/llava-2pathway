import os
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from typing import List, Optional, Union, Tuple
import torch
from tqdm import tqdm
from decord import VideoReader, cpu
import numpy as np
import math
from datetime import timedelta
from transformers import AutoConfig
from huggingface_hub import snapshot_download
import requests

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria
from lmms_eval.models.model_utils.load_video import read_video_pyav2

import subprocess
from loguru import logger as eval_logger

import json
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import concurrent.futures
from PIL import Image
from PIL import ImageDraw, ImageFont
import gc

@register_model("llava_2pathway")
class Llava2pathwayVideoMLLM(lmms):
    def __init__(
        self,
        pretrained: str = "/workspace/group_share/adc-perception-xplanner/songsy/llava-qwen-2pathway/baseline_siglip/finetune/",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_flash_attn=True,
        device_map="auto",
        conv_template="qwen_1_5",
        use_cache=True,
        truncate_context=False,
        num_frames: int = 32,
        fps: int = 2,
        add_ts = False,
        video_decode_backend="pyav",
        **kwargs,
    ) -> None:
        super().__init__()

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
            # self._device = accelerator.device
            # self.device_map = accelerator.device
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        self.pretrained = pretrained
        self.model_path = pretrained
        self.model_name = get_model_name_from_path(pretrained)
        self.num_frames = num_frames
        self.video_decode_backend = video_decode_backend
        
        accelerator.wait_for_everyone()
        self._tokenizer, self._model, self.image_processor, self._max_length = load_pretrained_model(
            self.model_path,
            None,
            self.model_name,
            device_map=self.device_map,
            use_flash_attn=use_flash_attn,
        )

        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        self.fps = fps
        self.add_ts = add_ts
        
                
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def add_timestamp_to_frame(self, frame, start_sec, end_sec, font_size=40):
        # 计算时间戳区域高度（基于原图像高度）
        timestamp_height = int(frame.height * 0.04)  # 原图像高度的10%
        
        # 创建新的画布（高度增加时间戳区域）
        new_height = frame.height + timestamp_height
        new_frame = Image.new('RGB', (frame.width, new_height), color=(255, 255, 255))
        
        # 将原图像粘贴到新画布的下半部分
        new_frame.paste(frame, (0, timestamp_height))
        
        # 在新画布的上半部分添加时间戳
        draw = ImageDraw.Draw(new_frame)
        font_size = int(frame.height * 0.04)
        #font = ImageFont.load_default(font_size)
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        
        text = f"{self.sec2hms(start_sec)}-{self.sec2hms(end_sec)}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # 时间戳居中显示在上方区域
        x = (frame.width - text_w) // 2
        y = (timestamp_height - text_h - 3) // 2
        
        # 修复矩形绘制：使用 RGB 颜色，不透明
        draw.rectangle([x-15, y-6, x+text_w+15, y+text_h+6], fill=(50, 50, 50))  # 移除了透明度
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        return new_frame
    
    def sec2hms(self,seconds):
        seconds = int(round(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def load_video(self, video_path, add_ts=False):
        max_frames_num = self.num_frames
        vr = VideoReader(video_path, num_threads=4)
        total_frame_num = len(vr)
        fps = round(vr.get_avg_fps())
        frame_idx = [i for i in range(0, len(vr), fps)]
        
        uniform_sampled_frames = np.linspace(0, len(vr) - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()        
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        
        total_time = total_frame_num / fps
        time_interval = round(total_time/len(frame_idx),3)
        start_sec, end_sec = 0,0
        frames_with_ts = []
        if add_ts:
            for i, frame in enumerate(spare_frames):
                start_sec = i * time_interval
                end_sec = start_sec + time_interval
                frame_with_ts = self.add_timestamp_to_frame(frame, start_sec, end_sec)
                frames_with_ts.append(frame_with_ts)
            spare_frames = frames_with_ts
        video_info_string = f"Time: {round(total_time, 2)}s; Time interval between frame {round(total_time/len(frame_idx), 3)}s; video tokens:"
    
        return spare_frames, video_info_string

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_response(self, input_ids, video, attention_masks, pad_token_ids, gen_kwargs):
        do_sample = getattr(gen_kwargs, "do_sample", False)
        temperature = getattr(gen_kwargs, "temperature", 0.0)
        max_new_tokens = getattr(gen_kwargs, "max_new_tokens", 256)
        num_beams = getattr(gen_kwargs, "num_beams", 1)
        top_p = getattr(gen_kwargs, "top_p", 1.0)
        
        with torch.inference_mode():
            # 避免计算图及中间缓存
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_masks,
                pad_token_id=pad_token_ids,
                images=video,
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                top_p=top_p,
                use_cache=self.use_cache,
            )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # 及时释放 GPU 张量
        del output_ids
        return outputs

    def generate_until(self, requests):
        res = []
        total_requests = len(requests)


        pbar = tqdm(total=total_requests, disable=(self.rank != 0), desc="Model Responding")

        def process_request(contexts, gen_kwargs, doc_to_visual, doc_id, task, split):
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            videos = []
            
            video_info_string = None 
            for visual in visuals:
                if isinstance(visual, Image.Image):
                    with torch.inference_mode():
                        image_tensor = self.image_processor.preprocess(visual, return_tensors="pt")["pixel_values"].half().to(self.device, non_blocking=True)
                        videos.append(image_tensor)
                else:
                    try:
                        if self.video_decode_backend == "decord":
                            video, video_info_string = self.load_video(visual, self.add_ts)
                        elif self.video_decode_backend == "pyav":
                            video, video_info_string = read_video_pyav2(visual, num_frm=self.num_frames, target_fps=self.fps, add_ts=self.add_ts)
                        with torch.inference_mode():
                            video_tensor = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().to(self.device, non_blocking=True)
                            videos.append(video_tensor)
                        del video
                    except Exception as e:
                        eval_logger.error(f"Error {e} in reading video file {visual}")
                        outputs = ""
                        return outputs
                        
            if len(videos) > self.num_frames:
                uniform_sampled_frames = np.linspace(0, len(videos) - 1, self.num_frames, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
                videos = [videos[idx_] for idx_ in frame_idx]
                videos = [torch.cat(videos, dim=0)]
            
            qs = contexts
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            
            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device, non_blocking=True)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device, non_blocking=True)
            
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.generate_response, input_ids, videos, attention_masks, pad_token_ids, gen_kwargs)
                    outputs = future.result(timeout=300)
            except concurrent.futures.TimeoutError:
                eval_logger.error("Generation process timed out")
                outputs = ""
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                outputs = ""

            # 清理本轮请求中分配的显存/内存
            try:
                del input_ids
                del attention_masks
                del input_ids_list
                for v in videos:
                    del v
                del videos
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return outputs
                
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:        
            outputs = process_request(contexts, gen_kwargs, doc_to_visual, doc_id, task, split)
            pbar.update(1)
            res.append(outputs)
            
            if "perceptiontest" in task.lower() and self.num_frames > 64:
                with open('perception_test_temp_cache.jsonl', "a") as file:
                    json_line = json.dumps({doc_id: outputs})  # Convert the dictionary to a JSON string
                    file.write(json_line + "\n")  
            
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        return super().loglikelihood(requests)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size
    
    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids