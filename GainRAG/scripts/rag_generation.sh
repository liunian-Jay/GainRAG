. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate DeepSpeed

cd ../gainRAG
python -m rag_workflow.rag_generation \
    --data_path " path/GainRAG/data/selection_data/HotpotQA.jsonl" \
    --task "HotpotQA" \
    --lm_type "Llama-3-8B-Instruct" \
    --K_docs 1