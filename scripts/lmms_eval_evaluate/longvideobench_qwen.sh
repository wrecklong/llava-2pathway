MODEL_PATH=$1
TEST_FRAMES=$2

#export HF_HOME=$(realpath ~/.cache/huggingface)
export HF_HOME=/workspace/group_share/adc-perception-xplanner/songsy/evaluate_datasets

pip install loguru
pip install tenacity
pip install hf_transfer
pip install --upgrade transformers
pip install pathlib
pip install pywsd
pip install pycocoevalcap
# pip install sqlitedict
# pip install evaluate
# pip install sacrebleu

python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger_eng')"
python -m accelerate.commands.launch \
        --num_processes=16 \
        lmms_eval_evaluate.py \
        --model slowfast_videomllm \
        --model_args pretrained=${MODEL_PATH},video_decode_backend=pyav,conv_template=qwen_1_5,num_frames=${TEST_FRAMES},device_map='' \
        --tasks longvideobench_val_v \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix longvideobench_slowfastvideomllm_ \
        --output_path /workspace/p-songsy@xiaopeng.com/Slow-Fast-Video-Multimodal-LLM/logs/  

