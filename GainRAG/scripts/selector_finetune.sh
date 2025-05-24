. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate DeepSpeed

export all_proxy=''

cd ../gainRAG
torchrun --nproc_per_node 1 \
	-m selector_finetune \
	--model_name_or_path  path/bge-rerank-base \
    --train_data path/data.jsonl \
	--deepspeed path/deepspeed/ds_stage0.json \
	--output_dir path/model_outputs/ \
	--overwrite_output_dir \
    --train_group_size 16 \
	--knowledge_distillation True \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --learning_rate 6e-5 \
    --fp16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --save_steps 1000