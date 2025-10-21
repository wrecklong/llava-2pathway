MODEL_PATH=$1
TEST_FRAMES=$2
NAME=$3

#export HF_HOME=$(realpath ~/.cache/huggingface)
#export HF_ENDPOINT=https://aifasthub.com/
export HF_ENDPOINT=https://hf_mirror.com/
export HF_HOME=/workspace/group_share/adc-perception-xplanner/songsy/evaluate_datasets
export HF_DATASETS_OFFLINE=1  # 首次使用请保持联网下载，离线请先准备本地缓存

pip install loguru
pip install tenacity
pip install hf_transfer
pip install transformers==4.51.0
pip install pathlib
pip install pywsd
pip install pycocoevalcap
# pip install sqlitedict
# pip install evaluate
# pip install sacrebleu

cp -r nltk_data  /root
export NLTK_DATA='/workspace/p-songsy@xiaopeng.com/llava-2pathway/nltk_data'
export NLTK_DOWNLOAD='false'

python -m accelerate.commands.launch \
        --num_processes=16 \
        lmms_eval_evaluate.py \
        --model llava_2pathway \
        --model_args pretrained=${MODEL_PATH},video_decode_backend=pyav,conv_template=qwen_1_5,num_frames=${TEST_FRAMES},add_ts=True,device_map='' \
        --tasks egoschema \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix egoschema_llava_2pathway \
        --output_path /workspace/p-songsy@xiaopeng.com/llava-2pathway-logs/$NAME   

#ok 本地加载 修改task.py 和对应task的yaml文件