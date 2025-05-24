. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate DeepSpeed

cd ../gainRAG/
python -m rag_workflow.run_baselines \
    --data_path " path/GainRAG/data/eval_data/TriviaQA.jsonl" \
    --task "SQuAD" \
    --lm_type "Llama-3-8B-Instruct" \
    --K_docs 1