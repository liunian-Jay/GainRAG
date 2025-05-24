
import json
from tqdm import tqdm

from llm_inference.build_prompts import llm_prompts
from llm_inference.inference_vllm import load_model_vllm, llm_response_vllm
from evaluation.evaluation import f1_max_over_ground_truths
from evaluation.evaluation import em_max_over_ground_truths
from rag_workflow.prompts import INSTRUCTION_PROMPT, TASK_INST, load_data

def eval_generate(input_file, output_file, task='HotpotQA', lm_type='Llama-3-8B-Instruct', batch_size=128):
    data = load_data(input_file)
    model, tokenizer = load_model_vllm(lm_type)

    def build_prompt(question, passage, task='HotpotQA'):
        evidence = f'Passage #1 title:{passage["title"]}\nPassage #1 text:{passage["text"]}\n\n'
        prompt = INSTRUCTION_PROMPT['Instruction_With_Retrieval'].format_map({'passage':evidence,'instruction':TASK_INST[task], 'input': question})
        return prompt
    
    retrieved_inputs = list()
    for item in data:
        question = item['question']
        answers = item['answers']
        passages = item['passages']

        sorted_passages = sorted(passages, key=lambda x: x["PPL_CD"])
        silver_passage = sorted_passages[0]
        prompt = build_prompt(question,silver_passage,task=task)
        retrieved_inputs.append({
            'prompt': prompt,
            'answers': answers
        })
    print('total items:', len(retrieved_inputs))

    em_scores = 0
    f1_scores = 0
    indices = list()
    for start in tqdm(range(0, len(retrieved_inputs), batch_size)):
        end = min(start+batch_size, len(retrieved_inputs))
        retrieved_items = retrieved_inputs[start:end]
        retrieved_prompts = [item['prompt'] for item in retrieved_items]
        retrieved_tokens = llm_prompts(lm_type, retrieved_prompts, tokenizer, tokenize=False)
        retrieved_response = llm_response_vllm(model, retrieved_tokens)

        for i, retrieved_item in enumerate(retrieved_items):
            em_score = em_max_over_ground_truths(retrieved_response[i], retrieved_item['answers'], regex=True)
            f1_score = f1_max_over_ground_truths(retrieved_response[i], retrieved_item['answers'])
            if em_score == 1:
                indices.append(start+i)
            em_scores += em_score
            f1_scores += f1_score
            print(f"{start+i}: With Retrieval - EM: {em_scores}, F1: {f1_scores}")

    filter_data = [data[i] for i in indices]
    with open(output_file, "w") as file:
        json.dump(filter_data, file)
    print('total items:', len(filter_data))


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for retrieval configuration.")
    parser.add_argument("--batch_size", type=int, default=128, help="Path to the model.")
    parser.add_argument("--task", type=str, default='HotpotQA', help="Task prompt template to use.")
    parser.add_argument("--lm_type", type=str, default='Llama-3-8B-Instruct', help="LLM to use.")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the input file.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to the output file.")
    args = parser.parse_args()
 
    eval_generate(args.data_path, args.output_path, args.task, lm_type=args.lm_type, batch_size=args.batch_size)