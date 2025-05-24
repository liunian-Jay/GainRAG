import os
import json
import torch
from tqdm import tqdm

from evaluation.evaluation import em_max_over_ground_truths, f1_max_over_ground_truths
from llm_inference.build_prompts import llm_prompts
from llm_inference.inference_hf import load_model
from llm_inference.inference_vllm import load_model_vllm, llm_response_vllm
from rag_workflow.prompts import get_input_with_R_Kth, INSTRUCTION_PROMPT, TASK_INST
from .decoding import ContrastiveTool

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def generate_passages(lm_type, prompts_data, task):
    model, tokenizer = load_model_vllm(lm_type)
    instruction = 'Please provide background for the question below in 100 words. Do not respond with anything other than background. If you do not know or are unsure, please generate "N/A" directly. \n\n'
    prompts = [f'{instruction} Question: {item["question"]}' for item in prompts_data]
    tokens = llm_prompts('Llama-3-8B-Instruct', prompts, tokenizer, tokenize=False)
    responses = llm_response_vllm(model, tokens)
    for passage, item in zip(responses, prompts_data):
        g_passage = {'title':'','text': passage}
        evidence = 'Passage #1  title:' + g_passage['title'] + '\n' + 'Passage #1  text:' + g_passage['text'] + '\n\n'

        passage_item = dict()
        passage_item['title'] = g_passage['title']
        passage_item['text'] = g_passage['text']
        passage_item['prompt_retrieved'] = INSTRUCTION_PROMPT['Instruction_With_Retrieval'].format_map({'passage':evidence,'instruction':TASK_INST[task], 'input': item['question']})
        item['passages'].append(passage_item)
    
    for i in tqdm(range(0, len(prompts_data))):
        retrieved_prompts = [prompt_item['prompt_retrieved'] for prompt_item in prompts_data[i]['passages']]
        retrieved_tokens = llm_prompts(lm_type, retrieved_prompts, tokenizer, tokenize=False)
        retrieved_responses = llm_response_vllm(model, retrieved_tokens)

        for prompt_item, retrieved_response in zip(prompts_data[i]['passages'], retrieved_responses):
            with_retrieval_em = em_max_over_ground_truths(retrieved_response, prompts_data[i]['answers'], regex=True)
            with_retrieval_f1 = f1_max_over_ground_truths(retrieved_response, prompts_data[i]['answers'])
            prompt_item['EM_with_retrieval'] = with_retrieval_em
            prompt_item['F1_with_retrieval'] = with_retrieval_f1
            # print(f"{i+1} : EM:{with_retrieval_em} F1:{with_retrieval_f1}")
    
    del model
    del tokenizer
    


def signal_construction(data_path, output_file, lm_type, task, alpha=0.5):
    contrastive_tool = ContrastiveTool()
    prompts_data = get_input_with_R_Kth(data_path, k = 20, task = task)
    generate_passages(lm_type, prompts_data, task)
    print('total_items: ', len(prompts_data))
    torch.cuda.empty_cache()

    model, tokenizer = load_model(lm_type)

    for i in tqdm(range(0, len(prompts_data))):
        retrieved_prompts = list()
        standard_prompt = prompts_data[i]['prompt_standard']
        for prompt_item in prompts_data[i]['passages']:
            retrieved_prompt = prompt_item['prompt_retrieved']
            retrieved_prompts.append(retrieved_prompt)

        standard_query = llm_prompts(lm_type, standard_prompt, tokenizer, tokenize=False)[0]
        retrieved_queries = llm_prompts(lm_type, retrieved_prompts, tokenizer, tokenize=False)
        
        gold_answers = list(set(prompts_data[i]['answers'])) 
        min_PPL_CD_list = [float('inf')]*len(prompts_data[i]['passages'])
        for gold_answer in gold_answers:
            PPL_CD_list = contrastive_tool.contrastive_PPL_multi_pro(model, tokenizer, standard_query, retrieved_queries, gold_answer, alpha=alpha)
            min_PPL_CD_list = [min(a, b) for a, b in zip(min_PPL_CD_list, PPL_CD_list)]
        
        print(i,':', min_PPL_CD_list)
        for prompt_item, min_PPL_CD in zip(prompts_data[i]['passages'], min_PPL_CD_list):
            prompt_item['PPL_CD'] = min_PPL_CD

    with open(output_file, "w") as f:
        json.dump(prompts_data, f, indent=4)
    # return prompts_data
    

# def passage_evaluation(prompts_data, lm_type, output_file):
#     model, tokenizer = load_model_vllm(lm_type)

#     for i in tqdm(range(0, len(prompts_data))):
#         retrieved_prompts = [prompt_item['prompt_retrieved'] for prompt_item in prompts_data[i]['passages']]
#         retrieved_tokens = llm_prompts(lm_type, retrieved_prompts, tokenizer, tokenize=False)
#         retrieved_responses = llm_response_vllm(model, retrieved_tokens)

#         for prompt_item, retrieved_response in zip(prompts_data[i]['passages'], retrieved_responses):
#             with_retrieval_em = em_max_over_ground_truths(retrieved_response, prompts_data[i]['answers'], regex=True)
#             with_retrieval_f1 = f1_max_over_ground_truths(retrieved_response, prompts_data[i]['answers'])
#             prompt_item['EM_with_retrieval'] = with_retrieval_em
#             prompt_item['F1_with_retrieval'] = with_retrieval_f1
#             print(f"{i+1} : EM:{with_retrieval_em} F1:{with_retrieval_f1}")
    
#     with open(output_file, "w") as f:
#         json.dump(prompts_data, f, indent=4)


from transformers import set_seed
set_seed(2024)
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm_type", type=str, default='Llama-3-8B-Instruct', help="LLM to use.")
    parser.add_argument("--task", type=str, default='HotpotQA', help="Task prompt template to use.")
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha of contrastive decoding.")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the input file.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to the output file.")
    args = parser.parse_args()
    
    prompts_data = signal_construction(args.data_path, args.output_path, args.lm_type, args.task, args.alpha)
    # passage_evaluation(prompts_data, args.lm_type, args.output_path)

