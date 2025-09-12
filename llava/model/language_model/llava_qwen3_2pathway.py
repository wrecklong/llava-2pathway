#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import OrderedDict

import transformers
# from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config, Qwen3Model, Qwen3ForCausalLM
from transformers.cache_utils import Cache, DynamicCache, StaticCache

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.model.language_model.hybrid_decoder_layer_qwen3 import Qwen3DecoderLayer, Qwen3HybridDecoderLayer, DynamicRouter, Predictor
from llava.model.language_model.merge import bipartite_soft_matching
import deepspeed

class LlavaQwen2pathwayConfig(Qwen3Config):
    model_type = "llava_qwen_2pathway"


class LlavaQwen2pathwayModel(LlavaMetaModel, Qwen3Model):
    config_class = LlavaQwen2pathwayConfig

    def __init__(self, config: Qwen3Config):
        super(LlavaQwen2pathwayModel, self).__init__(config)
        
        # initialize the cross-attention layers
        self.slow_branch_is_initialized = False
        
        if hasattr(config, "cross_attn_implementation"):
            self.initialize_slow_branch_modules(config)
    
    def initialize_slow_branch_modules(self, args):
        if self.slow_branch_is_initialized:
            return 
        # number of decoder layers
        num_layers = len(self.layers) 
        
        cross_attn_gating_type = args.cross_attn_gating_type
        cross_attn_implementation = args.cross_attn_implementation
        cross_attn_max_layer_depth = getattr(args, "cross_attn_max_layer_depth", num_layers)
        cross_attn_min_layer_depth = getattr(args, "cross_attn_min_layer_depth", 0)
        if cross_attn_max_layer_depth is None:
            cross_attn_max_layer_depth = num_layers
        if cross_attn_min_layer_depth is None:
            cross_attn_min_layer_depth = 0
        
        self.config.cross_attn_implementation = cross_attn_implementation
        self.config.cross_attn_gating_type = cross_attn_gating_type
        self.config.cross_attn_max_layer_depth = cross_attn_max_layer_depth
        self.config.cross_attn_min_layer_depth = cross_attn_min_layer_depth
        
        # substitute the original decoder layer with hybrid layer
        for idx in range(len(self.layers)):
            original_decoder_layer = self.layers[idx]
            if isinstance(original_decoder_layer, Qwen3HybridDecoderLayer):
                continue
            hybrid_decoder_layer = Qwen3HybridDecoderLayer(self.config, layer_idx=idx, is_hyper_enabled=True, cross_attn_gating_type=cross_attn_gating_type, cross_attn_implementation=cross_attn_implementation)
            #self._manual_copy_parameters(original_decoder_layer, hybrid_decoder_layer)
            _, unexpected_keys = hybrid_decoder_layer.load_state_dict(original_decoder_layer.state_dict(), strict=False) # cause problem when using deepspeed zero3
            self.layers[idx] = hybrid_decoder_layer
        
        # fast token config
        # self.config.fast_token_spatial_stride = args.fast_token_spatial_stride
        # self.config.fast_token_temporal_stride = args.fast_token_temporal_stride
        # self.config.fast_token_temporal_sampling_stride = args.fast_token_temporal_sampling_stride
        
        self.slow_branch_is_initialized = True

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        activation_mask=None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                # logger.warning_once(
                #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                # )
                use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                # logger.warning_once(
                #     "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                #     "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                #     "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                # )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)


        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
       
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            layer_mask = activation_mask[:, layer_idx]
            if self.gradient_checkpointing and self.training:
                #if not isinstance(decoder_layer, Qwen3HybridDecoderLayer):
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    layer_mask
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    activation_mask=layer_mask
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class LlavaQwen2pathwayForCausalLM(Qwen3ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwen2pathwayConfig

    def __init__(self, config):
        Qwen3ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen_2pathway"
        config.rope_scaling = None
        
        self.model = LlavaQwen2pathwayModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_hidden_layers = config.num_hidden_layers
        
        self.is_router_initialized = False

        if hasattr(config, 'routing_ratio'):
            self.token_router = Predictor(self.model.config.hidden_size, config.routing_ratio)
            self.cross_dynamic_router = DynamicRouter(self.model.config.hidden_size, self.model.config.num_hidden_layers, config.cross_attn_experts)
            self.is_router_initialized = True
        # else:
        #     self.initialize_router(config)

        # if hasattr(config, "routing_ratio"):
        #     self.initialize_router(config)
        self.post_init()

    def get_model(self):
        return self.model
    
    def initialize_router(self, args):
        if self.is_router_initialized:
            return
        else:
            self.token_router = Predictor(self.model.config.hidden_size, args.routing_ratio)
            self.cross_dynamic_router = DynamicRouter(self.model.config.hidden_size, self.model.config.num_hidden_layers, args.cross_attn_experts)
            
            self.model.config.cross_attn_experts = args.cross_attn_experts
            self.model.config.routing_ratio = args.routing_ratio

            self.is_router_initialized = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Qwen3HybridDecoderLayer):
            module.gradient_checkpointing = value
    
    def ratio_loss(self, masks, ratio):
        pred_loss = 0.0
        if isinstance(masks, torch.Tensor):
            masks = [masks[i,:] for i in range(masks.shape[0])]
        for i, mask in enumerate(masks):
            pos_ratio = mask.to(torch.float32).mean()
            pred_loss = pred_loss + ((pos_ratio - ratio) ** 2).mean()
        
        return pred_loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        cache_position=None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # import pdb;pdb.set_trace()
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            activation_mask=self.activation_mask,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
            cross_attn_ratio_loss = self.ratio_loss(self.activation_mask, float(self.model.config.cross_attn_experts/self.num_hidden_layers))
            token_routing_ratio_loss = self.ratio_loss(self.text_related_features_masks, self.model.config.routing_ratio)
        
            loss = loss + cross_attn_ratio_loss + token_routing_ratio_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


    def select_features_and_merge_features(self, image_features, input_ids):
        if isinstance(image_features, torch.Tensor):
            image_features = [image_features[i:i+1] for i in range(image_features.size(0))]

        text_related_features = []
        less_related_features = []
        text_related_features_masks = []

        for batch_idx, (cur_input_ids, cur_image_features) in enumerate(zip(input_ids, image_features)):
            t, _ ,_ = cur_image_features.shape
            cur_input_ids_without_image = cur_input_ids[cur_input_ids != IMAGE_TOKEN_INDEX]
            cur_input_text_embeds = self.get_model().embed_tokens(cur_input_ids_without_image)
            text_related_features_mask = self.token_router(cur_image_features, cur_input_text_embeds)

            cur_text_related_features = cur_image_features.flatten(0,1)[text_related_features_mask]
            cur_less_related_features = cur_image_features.flatten(0,1)[~text_related_features_mask]
            n, _ = cur_less_related_features.shape 
            if t > 1:
                merge, _ = bipartite_soft_matching(cur_less_related_features.unsqueeze(0), n//2)
                merged_cur_less_related_feature = merge(cur_less_related_features.unsqueeze(0))
                less_related_features.append(merged_cur_less_related_feature.squeeze(0))
            else:
                less_related_features.append(cur_less_related_features)
    
            text_related_features_masks.append(cur_less_related_features)
            text_related_features.append(cur_text_related_features)

        return text_related_features, less_related_features, text_related_features_masks

    def get_decoder(self):
        return self.model        

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            # clear the visual tokens if current one is a pure text sample
            if images is None and input_ids.shape[1] > 1:
                for layer in self.get_decoder().layers:
                        if hasattr(layer, "clear_vis_x"):
                            layer.clear_vis_x()
            
            token_types = torch.ones_like(input_ids, dtype=input_ids.dtype, device=input_ids.device)
            for layer in self.get_decoder().layers:
                if hasattr(layer, "condition_vis_x"):
                    layer.media_locations = token_types
            
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # handle image input
        images = [image if len(image.shape) == 4 else image.unsqueeze(0) for image in images] # list [ [T, C, H, W], ]
        feature_split_size = [x.shape[0] for x in images]
        all_features, feature_split_size = self.encode_images(torch.cat(images, dim=0), feature_split_size)

        raw_image_features = torch.split(all_features, feature_split_size, dim=0)
        image_features = []

        for sample_feat in raw_image_features:   # initial spatial pooling for all video tokens
            if sample_feat.shape[0] > 1 and self.config.mm_video_pooling_stride > 1:
                b, n, c = sample_feat.shape
                h = w = int(n**0.5)                    
                sample_feat = nn.functional.avg_pool2d(sample_feat.reshape(b, h, w, c).permute(0, 3, 1, 2),
                                                        kernel_size=self.config.mm_video_pooling_stride, 
                                                        stride=self.config.mm_video_pooling_stride).flatten(2,3).transpose(1,2)
                                                    
            image_features.append(sample_feat.contiguous())
        del raw_image_features, all_features

        # # TODO: image start / end is not implemented here to support pretraining.
        # if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
        #     raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # select more text related features & less text lerated features 
    
        text_related_features, less_related_features, text_related_features_masks = self.select_features_and_merge_features(image_features, input_ids)
        self.text_related_features_masks = text_related_features_masks

        ## set cross-attention states
        if isinstance(less_related_features, (list, tuple)):
            padded_tensors = torch.nn.utils.rnn.pad_sequence(less_related_features, batch_first=True)
            cross_attn_mask = torch.ones(padded_tensors.shape[:-1], dtype=torch.bool, device=padded_tensors.device)
            for i, tensor in enumerate(less_related_features):
                cross_attn_mask[i, len(tensor):] = False  # Mark padded elements as False  
            less_related_features = padded_tensors
        else:
            cross_attn_mask = torch.ones(less_related_features.shape[:-1], dtype=torch.bool, device=less_related_features.device)
        
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError


        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        new_token_types = [] 
        # NOTE: Min: we need to record the type of tokens so that we can split the tokens in the hybrid decoder layer
        # Token type 1: user's input and system tokens, 2: response text tokens, 3: visual tokens, 4: invalid tokens (padding)
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_text_related_features = text_related_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_text_related_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                
                cur_token_type = torch.full((cur_input_ids.shape[0],), 2, dtype=cur_input_ids[-1].dtype, device=cur_input_ids[-1].device) 
                cur_token_type[labels[batch_idx] == IGNORE_INDEX] = 1 # token with ignore tokens are considered as user input 
                new_token_types.append(cur_token_type)
                
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            cur_token_type_noim = []
            
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
                
                cur_token = torch.full((cur_labels_noim[-1].shape[0],), 2, dtype=cur_input_ids_noim[-1].dtype, device=cur_input_ids_noim[-1].device) 
                cur_token[cur_labels[image_token_indices[i]+1:image_token_indices[i+1]] == IGNORE_INDEX] = 1 # ingored tokens are considered as user input 
                cur_token_type_noim.append(cur_token) 
                
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_token_type = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_token_type.append(cur_token_type_noim[i])
                
                if i < num_images:
                    cur_text_related_features = text_related_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_text_related_features)
                    cur_new_labels.append(torch.full((cur_text_related_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_token_type.append(torch.full((cur_text_related_features.shape[0],), 3, device=cur_labels.device, dtype=cur_labels.dtype)) # insert image token type

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_token_type = torch.cat(cur_new_token_type) ##

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_token_types.append(cur_new_token_type) ##
            
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_token_types = [x[:tokenizer_model_max_length] for x in new_token_types]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        new_token_types_padded = torch.full((batch_size, max_len), 4, dtype=new_labels[0].dtype, device=new_labels[0].device) ## 4 is invalid token type (padding)
        
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    new_token_types_padded[i, -cur_len:] = new_token_types[i] ## 
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    new_token_types_padded[i, :cur_len] = new_token_types[i]

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        # token type
        token_types = new_token_types_padded
        # send token type to cross-attn layers
        
        activation_mask = self.cross_dynamic_router(less_related_features)
        self.activation_mask = activation_mask
        
        if _input_ids is not None and _input_ids.shape[-1] == 1:
            pass
        else:
            if less_related_features is not None:
                for layer in self.get_decoder().layers:
                    if hasattr(layer, "condition_vis_x"):
                        layer.condition_vis_x(less_related_features, 
                                              cross_attn_mask,
                                              token_type=token_types)
            else:
                for layer in self.get_decoder().layers:
                    if hasattr(layer, "clear_vis_x"):
                        layer.clear_vis_x() 
        
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

AutoConfig.register("llava_qwen_2pathway", LlavaQwen2pathwayConfig)
AutoModelForCausalLM.register(LlavaQwen2pathwayConfig, LlavaQwen2pathwayForCausalLM)