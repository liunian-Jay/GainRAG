from tqdm import tqdm
from llm_inference.build_prompts import llm_prompts
from llm_inference.inference_hf import load_model, llm_response
from llm_inference.inference_vllm import load_model_vllm, llm_response_vllm
from evaluation.evaluation import f1_max_over_ground_truths
from evaluation.evaluation import em_max_over_ground_truths
from evaluation.evaluation import accuracy
from rag_workflow.prompts import get_input



def main(data_path, task, lm_type, K = 1, batch_size = 256):
    retrieved_inputs = get_input(data_path, retrieval = True, k = K)
    standard_inputs = get_input(data_path, retrieval = False)
    total_items = len(retrieved_inputs)
    # model, tokenizer = load_model(lm_type)
    model, tokenizer = load_model_vllm(lm_type)

    scores = {'without_retrieval': 0, 'with_retrieval': 0}
    f1_scores = {'without_retrieval': 0, 'with_retrieval': 0}

    for start in tqdm(range(0, len(retrieved_inputs), batch_size)):
        end = min(start+batch_size, len(retrieved_inputs))
        retrieved_items = retrieved_inputs[start:end]
        standard_items = standard_inputs[start:end]

        retrieved_prompts = [item['prompt'] for item in retrieved_items]
        standard_prompts = [item['prompt'] for item in standard_items]
        # TODO
        standard_prompts = []

        # retrieved_tokens = llm_prompts(lm_type, retrieved_prompts, tokenizer).to('cuda')
        # standard_tokens = llm_prompts(lm_type, standard_prompts, tokenizer).to('cuda')
        # retrieved_response = llm_response(retrieved_tokens, model, tokenizer)
        # standard_response = llm_response(standard_tokens, model, tokenizer)

        retrieved_tokens = llm_prompts(lm_type, retrieved_prompts, tokenizer, tokenize=False)
        standard_tokens = llm_prompts(lm_type, standard_prompts, tokenizer, tokenize=False)
        retrieved_response = llm_response_vllm(model, retrieved_tokens)
        standard_response = llm_response_vllm(model, standard_tokens)

        for i, (retrieved_item, standard_item) in enumerate(zip(retrieved_items, standard_items)):
            print(f"Response {start+i} without retrieval: {standard_response[i]}")
            print(f"Response {start+i} with retrieval: {retrieved_response[i]}")
            print(f"The gold answer to {start+i}: {standard_item['answers']}")
            
            if task == 'ARC_Challenge':
                scores['without_retrieval'] += accuracy(standard_response[i], standard_item['answers'])
                scores['with_retrieval'] += accuracy(retrieved_response[i], retrieved_item['answers'])
            else:
                scores['without_retrieval'] += em_max_over_ground_truths(standard_response[i], standard_item['answers'], regex=True)
                scores['with_retrieval'] += em_max_over_ground_truths(retrieved_response[i], retrieved_item['answers'], regex=True)
                f1_scores['without_retrieval'] += f1_max_over_ground_truths(standard_response[i], standard_item['answers'])
                f1_scores['with_retrieval'] += f1_max_over_ground_truths(retrieved_response[i], retrieved_item['answers'])
            
            print(f"{start+i}: W/O Retrieval - EM: {scores['without_retrieval']}, F1: {f1_scores['without_retrieval']}")
            print(f"{start+i}: With Retrieval - EM: {scores['with_retrieval']}, F1: {f1_scores['with_retrieval']}")


    avg_em_o = scores['without_retrieval'] / total_items
    avg_em = scores['with_retrieval'] / total_items
    avg_f1_o = f1_scores['without_retrieval'] / total_items
    avg_f1 = f1_scores['with_retrieval'] / total_items
    print("-"*100, f'\nFinal Results:\nW/O Retrieval EM: {avg_em_o*100}\nWith Retrieval EM: {avg_em*100}\n', "-"*100)
    print(f'W/O Retrieval F1: {avg_f1_o*100}\nWith Retrieval F1: {avg_f1*100}')
    
        

import torch
import random
import numpy as np
seed = 2024
torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for retrieval configuration.")
    parser.add_argument("--batch_size", type=int, default=128, help="Path to the model.")
    parser.add_argument("--K_docs", type=int, default=1, help="Number of documents to retrieve.")
    parser.add_argument("--task", type=str, default='HotpotQA', help="Task prompt template to use.")
    parser.add_argument("--lm_type", type=str, default='Llama-3-8B-Instruct', help="LLM to use.")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the input file.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to the output file.")
    args = parser.parse_args()
 
    main(args.data_path, args.task, lm_type=args.lm_type, K=args.K_docs, batch_size=args.batch_size)
 