. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate DeepSpeed

cd ../gainRAG
python -m selector_engine.selector_gainRag \
    --model_name_or_path "path/GainRAG/model_outputs/2025-1-19" \
    --data_path "path/GainRAG/data/eval_data/HotpotQA.jsonl" \
    --output_path "path/GainRAG/data/test.json" \
    --K_docs 1