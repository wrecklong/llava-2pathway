MODEL_PATH=$1
TEST_FRAMES=$2

#export HF_HOME=$(realpath ~/.cache/huggingface)
# export HF_ENDPOINT=https://aifasthub.com/
export HF_ENDPOINT=https://hf_mirror.com/
export HF_HOME=/workspace/group_share/adc-perception-xplanner/songsy/evaluate_datasets
export HF_DATASETS_OFFLINE=1

pip install loguru
pip install tenacity
pip install hf_transfer
pip install transformers==4.51.0
pip install pathlib
pip install pywsd
pip install pycocoevalcap

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger_eng')"
python -m accelerate.commands.launch \
        --num_processes=16 \
        lmms_eval_evaluate.py \
        --model llava_2pathway \
        --model_args pretrained=${MODEL_PATH},video_decode_backend=pyav,conv_template=qwen_1_5,num_frames=${TEST_FRAMES},device_map='' \
        --tasks tempcompass \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix tempcompass_llava_2pathway \
        --output_path /workspace/p-songsy@xiaopeng.com/llava-2pathway/logs/ 


#需要openai api