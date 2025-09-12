#!/bin/bash
NAME=$1
ANNOTATION_PATH="/workspace/p-songsy@xiaopeng.com/Slow-Fast-Video-Multimodal-LLM/scripts/exp_pretrain.yaml"
IMAGE_DIR="/workspace/group_share/adc-perception-xplanner/songsy/datasets/LLaVA-Pretrain"
VIDEO_DIR="/workspace/group_share/adc-perception-xplanner/songsy/datasets/LLaVA-Video-178K"

pip install transformers==4.51.0

python -m torch.distributed.run \
    --nproc_per_node 1 --nnodes 1 --node_rank 0 \
    --master_addr "localhost" --master_port 29500 \
    train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path  /workspace/group_share/adc-perception-xplanner/songsy/pretrained_models/Qwen3-4B-Instruct-2507\
    --version qwen_1_5 \
    --data_path ${ANNOTATION_PATH} \
    --image_folder ${IMAGE_DIR} \
    --video_folder ${VIDEO_DIR}  \
    --vision_tower "/workspace/group_share/adc-perception-xplanner/songsy/pretrained_models/google/siglip-so400m-patch14-384" \
    --mm_projector_type mlp2x_gelu \
    --fps 4 \
    --video_frames 64 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --tile_image_input False \
    --cross_attn_implementation text-only-vanilla-flashattn \
    --cross_attn_gating_type whole-dynamic-tanh-warmup \
    --initialize_cross_attn_kv_from_lm True \
    --mm_video_pooling_stride 3 \
    --bf16 True \
    --output_dir /workspace/p-songsy@xiaopeng.com/Slow-Fast-Video-Multimodal-LLM/output_dir/$NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --cross_attention_layer_lr 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 5000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard\
    --run_name ${NAME} \
    --routing_ratio 0.25 \
    --cross_attn_experts 4
    
    # --min_fast_frames 16\
    # --cross_attn_every_n_layers 8 \
    # --fast_token_spatial_stride 1 \
    # --fast_token_temporal_stride 4 \ 
    # --fast_token_temporal_sampling_stride 1 
