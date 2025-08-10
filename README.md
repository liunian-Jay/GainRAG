<div align="center">

# GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis 

<p align="center">

  <a href="https://arxiv.org/abs/2505.18710">
    <img src="https://img.shields.io/badge/arXiv-2505.18710-b31b1b.svg" alt="arXiv">
  </a>
</p>
</div>


<div align="center">
<img src="images/framework.png" alt="framework" width="800">

**GainRAG Framework**
</div>

## 🛠 Installation


<details>
<summary>
The main dependencies are torch 2.5.1, vllm 0.7.3, FlagEmbedding 1.3.3, DeepSpeed, trl, peft, faiss/faiss-gpu.
</summary>

```bash
conda create -n GainRAG python=3.9.18
conda activate GainRAG
pip install -r requirements.txt
```
</details>


## 💡 Preparation
***Download Corpus & Index***

<details>
<summary>
Retrieval is performed on the set of Wikipeda passages used in DPR. Download passages:
</summary>

```bash
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
```
</details>

<details>
<summary>
Download passage embeddings pre-computed with Contriever or Contriever-msmarco:
</summary>
    
```bash
wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever/wikipedia_embeddings.tar
wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar
```
</details>

<details>
<summary>
Retrieve top-k passages:
</summary>
  
```bash
cd ./gainRAG/retrieval_engine
python retrieval.py # Remember to configure your parameters
```
</details>


## 🎯 Train Selector
<details>
<summary>
Gain Signal Synthesis:
</summary>
  
```bash
cd ./gainRAG
python -m llm_supervision.construct_hf \
    --data_path  TODOpath/data.jsonl \
    --output_path  TODOpath/data_train.json \
    --task HotpotQA \
    --alpha 0.5
```
</details>

<details>
<summary>
Data format conversion:
</summary>
  
```bash
cd ./data
python data2selector.py # Remember to configure your parameters
```
</details>

<details>
<summary>
Selector Training:
</summary>

The initial weight of the model is bge-rerank-base，
```bash
cd ./gainRAG
torchrun --nproc_per_node 1 \
	-m selector_finetune \
	--model_name_or_path  path/bge-rerank-base \
    --train_data TODOpath/data.jsonl \
	--deepspeed TODOpath/deepspeed/ds_stage0.json \
	--output_dir TODOpath/model_outputs/\
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
```
</details>

## 📈 Run Evaluation
<details>
<summary>
0. Download Evaluation Data:
</summary>
  
[HotpotQA](https://hotpotqa.github.io/), [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop), [WebQuestions](https://nlp.stanford.edu/software/sempre/), [NaturalQA](https://ai.google.com/research/NaturalQuestions), [TriviaQA](http://nlp.cs.washington.edu/triviaqa/), [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
</details>


<details>
<summary>
1. Retrieve top-k passages:
</summary>
  
```bash
cd ./gainRAG/retrieval_engine
python retrieval.py # Remember to configure your parameters
```
</details>

<details>
<summary>
2. Select top-1 passages:
</summary>
  
```bash
cd ./gainRAG
python -m selector_engine.selector_gainRag \
    --model_name_or_path "model_path/" \
    --data_path "path/GainRAG/data/eval_data/HotpotQA.jsonl" \
    --output_path "path/GainRAG/data/test.json" \
    --K_docs 1
```
</details>

<details>
<summary>
3. Run generation & evaluation:
</summary>
  
```bash
cd ./gainRAG
python -m rag_workflow.rag_generation \
    --data_path "selector_output_path" \
    --task "HotpotQA" \
    --lm_type "Llama-3-8B-Instruct" \
    --K_docs 1
```
</details>


## Citation
```bibtex
@article{jiang2025gainrag,
  title={GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis},
  author={Jiang, Yi and Zhao, Sendong and Li, Jianbo and Wang, Haochun and Qin, Bing},
  journal={arXiv preprint arXiv:2505.18710},
  year={2025}
}
```

Thanks for your interest in our work!




