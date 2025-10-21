#!/bin/bash
NAME=$1
ANNOTATION_PATH="/workspace/p-songsy@xiaopeng.com/llava-2pathway/scripts/exp_finetune.yaml"
IMAGE_DIR="/workspace/group_share/adc-perception-xplanner/songsy/datasets/"
VIDEO_DIR="/workspace/group_share/adc-perception-xplanner/songsy/datasets/LLaVA-Video-178K"

# MASTER_ADDR="localhost"
# MASTER_PORT=29500
# WORLD_SIZE=1
# NODE_RANK=0

pip install transformers==4.51.0
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000000
export TORCH_NCCL_TRACE_DUMP_DIR=/workspace/p-songsy@xiaopeng.com/Slow-Fast-Video-Multimodal-LLM/nccl_log


torchrun --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes=${WORLD_SIZE} \
    --nproc_per_node=16 --node_rank=$NODE_RANK  train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /workspace/group_share/adc-perception-xplanner/songsy/llava-qwen-2pathway/$NAME/pretrain/\
    --version qwen_1_5 \
    --data_path ${ANNOTATION_PATH} \
    --image_folder ${IMAGE_DIR} \
    --video_folder ${VIDEO_DIR}  \
    --vision_tower "convnext-576" \
    --mm_projector_type mlp2x_gelu \
    --video_frames 64 \
    --tile_image_input False \
    --fps 4 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --cross_attn_implementation text-only-vanilla-flashattn \
    --cross_attn_gating_type whole-dynamic-tanh-warmup \
    --mm_video_pooling_stride 2 \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /workspace/group_share/adc-perception-xplanner/songsy/llava-qwen-2pathway/$NAME/finetune  \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 10 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name ${NAME}  \
    --routing_ratio 0.25 \
    --cross_attn_experts 4