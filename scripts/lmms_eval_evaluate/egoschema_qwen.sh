MODEL_PATH=$1
TEST_FRAMES=$2

#export HF_HOME=$(realpath ~/.cache/huggingface)
export HF_HOME=/workspace/group_share/adc-perception-xplanner/songsy/evaluate_datasets

python -m accelerate.commands.launch \
        --num_processes=8 \
        lmms_eval_evaluate.py \
        --model slowfast_videomllm \
        --model_args pretrained=${MODEL_PATH},video_decode_backend=pyav,conv_template=qwen_1_5,num_frames=${TEST_FRAMES},device_map='auto' \
        --tasks egoschema \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix egoschema_slowfastvideomllm_ \
        --output_path ./logs/  