import os
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer




def load_model_vllm(lm_type):
    if lm_type == "Llama-2-7b-chat-hf":
        model_name_or_path = os.path.abspath("TODOpath/Llama-2-7b-chat-hf")
    elif lm_type == "Llama-2-13b-chat-hf":
        model_name_or_path = os.path.abspath("TODOpath/Llama-2-13b-chat-hf")
    elif lm_type == "Llama-3-8B-Instruct":
        model_name_or_path = os.path.abspath("TODOpath/Meta-Llama-3-8B-Instruct")
    elif lm_type == "Llama-3-70B-Instruct":
        model_name_or_path = os.path.abspath("TODOpath/Meta-Llama-3-70B-Instruct")
    elif lm_type == "Mistral-7B-Instruct-v0.2":
        model_name_or_path = os.path.abspath("TODOpath/Mistral-7B-Instruct-v0.2")
    elif lm_type == "Mistral-7B-Instruct-v0.3":
        model_name_or_path = os.path.abspath("TODOpath/Mistral-7B-Instruct-v0.3") 

    elif lm_type == "Llama-3.1-8B-Instruct":
        model_name_or_path = os.path.abspath("TODOpath/Meta-Llama-3.1-8B-Instruct")
    elif lm_type == "Llama-3.1-70B-Instruct":
        model_name_or_path = os.path.abspath("TODOpath/Meta-Llama-3.1-70B-Instruct")   
    else:
        raise ValueError
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # llm = LLM(model=model_name_or_path)
    llm = LLM(model=model_name_or_path, tensor_parallel_size=torch.cuda.device_count(), max_model_len = 2048, gpu_memory_utilization=0.85, max_logprobs=1)
    # llm = LLM(model=model_name_or_path, tensor_parallel_size=torch.cuda.device_count(), max_model_len = 2048, gpu_memory_utilization=0.5, max_logprobs=len(tokenizer))
    return llm,tokenizer


def llm_response_vllm(llm, prompts, max_tokens=2048):
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens = max_tokens, logprobs=1)
    outputs = llm.generate(prompts, sampling_params)

    res = []
    for output in outputs:
        generated_text = output.outputs[0].text
        res.append(generated_text)
    return res


