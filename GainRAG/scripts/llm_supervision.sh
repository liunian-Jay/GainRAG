# source ~/.bashrc
. $HOME/anaconda3/etc/profile.d/conda.sh
# 设置运行环境
conda activate DeepSpeed

cd ../gainRAG
python -m llm_supervision.construct_hf \
    --data_path  TODOpath/data.jsonl \
    --output_path  TODOpath/data_train.json \
    --task HotpotQA \
    --alpha 0.5
