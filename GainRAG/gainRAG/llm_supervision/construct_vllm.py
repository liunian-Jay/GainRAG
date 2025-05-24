import json
from tqdm import tqdm
import math
import numpy as np
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from llm_inference.build_prompts import llm_prompts
from llm_inference.inference_vllm import load_model_vllm
from rag_workflow.prompts import load_data



def tokens_tokenize(tokens, answers, tokenizer):
    prompts_ids = [tokenizer(prompt, add_special_tokens=False)['input_ids'] for prompt in tokens]
    answers_ids = [tokenizer(prompt, add_special_tokens=False)['input_ids'] for prompt in answers]
    inputs_ids = [TokensPrompt(prompt_token_ids=x+y) for x,y in zip(prompts_ids,answers_ids)]
    return inputs_ids, answers_ids

def vllm_answer_ppl(llm, inputs_ids, answers_ids):
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens = 1, logprobs=1, prompt_logprobs=0)
    outputs = llm.generate(inputs_ids, sampling_params)
    res = list()
    for output, answer_ids in zip(outputs, answers_ids):
        prompt_logprobs = output.prompt_logprobs[-len(answer_ids):]
        logprobs = list()
        for pl, answer_id in zip(prompt_logprobs, answer_ids):
            logprob = pl[answer_id].logprob
            logprobs.append(logprob)
        avg_logprobs = -sum(logprobs)/len(logprobs)
        ppl = math.exp(avg_logprobs)
        res.append(ppl)
    return res





def passage_ppl(data_path, output_path, lm_type='Llama-3-8B-Instruct', batch_size = 512):
    
    prompts_data = load_data(data_path)
    total_items = len(prompts_data)
    print('total_items: ', total_items)
    model, tokenizer = load_model_vllm(lm_type)


    ####################################################
    retrieved_prompts = list()
    gold_answers_list = list() 
    gold_answers_nums = list()
    for i in tqdm(range(0, len(prompts_data))):
        gold_answer = list(set(prompts_data[i]['answers'])) #去重
        gold_answers_nums.append(len(gold_answer))
        for prompt_item in prompts_data[i]['passages']: 
            retrieved_prompt = prompt_item['prompt_retrieved']
            retrieved_prompts += [retrieved_prompt] * len(gold_answer)
            gold_answers_list += gold_answer

    print('X'*100, '\n', len(retrieved_prompts),len(gold_answers_list))
    assert len(retrieved_prompts) == len(gold_answers_list)
    ####################################################

    # retrieved_queries = llm_prompts(lm_type, retrieved_prompts, tokenizer, tokenize=False)
    # retrieved_inputs_ids, retrieved_answers_ids = tokens_tokenize(retrieved_queries, gold_answers_list, tokenizer)
    # retrieved_ppls = vllm_answer_ppl(model, retrieved_inputs_ids, retrieved_answers_ids)

    retrieved_ppls = []
    for batch_start in tqdm(range(0, len(retrieved_prompts), batch_size)):
        batch_retrieved_prompts = retrieved_prompts[batch_start:batch_start + batch_size]
        batch_retrieved_queries = llm_prompts(lm_type, batch_retrieved_prompts, tokenizer, tokenize=False)
        batch_gold_answers = gold_answers_list[batch_start:batch_start + batch_size]
        batch_retrieved_inputs_ids, batch_retrieved_answers_ids = tokens_tokenize(batch_retrieved_queries, batch_gold_answers, tokenizer)
        batch_ppls = vllm_answer_ppl(model, batch_retrieved_inputs_ids, batch_retrieved_answers_ids)
        retrieved_ppls.extend(batch_ppls)

    start_idx = 0
    for i in range(0, len(prompts_data)):
        for prompt_item in prompts_data[i]['passages']: 
            prompt_item['retrieved_ppl'] = min(retrieved_ppls[start_idx:start_idx + gold_answers_nums[i]])
            start_idx += gold_answers_nums[i]
    
    with open(output_path, "w") as f:
        json.dump(prompts_data, f, indent=4)


from transformers import set_seed
set_seed(2024)

if __name__ == '__main__':
    data_path = ' TODOpath/GainRAG/data/signal_data/merged_filter_data.json'
    output_path = ' TODOpath/GainRAG/data/signal_data/merged_filter_data_ppl.json'
    passage_ppl(data_path, output_path, lm_type='Llama-3-8B-Instruct', batch_size = 4096)