#!/bin/bash
NAME=$1
ANNOTATION_PATH="/workspace/p-songsy@xiaopeng.com/Slow-Fast-Video-Multimodal-LLM/scripts/exp_finetune.yaml"
IMAGE_DIR="/workspace/group_share/adc-perception-xplanner/songsy/datasets/"
VIDEO_DIR="/workspace/group_share/adc-perception-xplanner/songsy/datasets/LLaVA-Video-178K"

python -m torch.distributed.run \
    --nproc_per_node 1 --nnodes 1 --node_rank 0 \
    --master_addr "localhost" --master_port 25031 \
    train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /workspace/group_share/adc-perception-xplanner/songsy/pretrained_models/models--shi-labs--slowfast-video-mllm-qwen2-7b-convnext-576-frame64-s1t4/snapshots/eb494661d4986acd1ea085ac9ae148619fd53b07\
    --version qwen_1_5 \
    --data_path ${ANNOTATION_PATH} \
    --image_folder ${IMAGE_DIR} \
    --video_folder ${VIDEO_DIR}  \
    --vision_tower "convnext-576" \
    --mm_projector_type mlp2x_gelu \
    --video_frames 64 \
    --min_fast_frames 16 \
    --tile_image_input False \
    --fps 4 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --cross_attn_every_n_layers 8 \
    --cross_attn_head_num 28 \
    --cross_attn_kv_head_num 14 \
    --cross_attn_implementation text-only-vanilla-flashattn \
    --cross_attn_gating_type whole-dynamic-tanh-warmup \
    --initialize_cross_attn_kv_from_lm True \
    --mm_video_pooling_stride 2 \
    --fast_token_spatial_stride 1 \
    --fast_token_temporal_stride 4 \
    --fast_token_temporal_sampling_stride 1 \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /workspace/p-songsy@xiaopeng.com/Slow-Fast-Video-Multimodal-LLM/output_dir/$NAME  \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 12 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name ${NAME}  
