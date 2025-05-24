"""
    Adapt to the special input format of each model
"""


def llama2_prompt(query, tokenizer):
    def build_prompt(input):
        messages = [
            {"role": "system", "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being factually coherent. If you don't know the answer to a question, please don't share false information."}, 
            {"role": "user", "content": input}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    prompts = [build_prompt(inp) for inp in query] if type(query) == list else [build_prompt(query)] 
    return prompts  


def llama3_prompt(query, tokenizer):
    def build_prompt(input):
        messages = [
            {"role": "system", "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being factually coherent. If you don't know the answer to a question, please don't share false information."}, 
            {"role": "user", "content": input}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt
    
    prompts = [build_prompt(inp) for inp in query] if type(query) == list else [build_prompt(query)]
    return prompts    
    

def mistral_prompt(query, tokenizer):
    def build_prompt(input):
        messages = [{"role": "user", "content": input}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt
    
    prompts = [build_prompt(inp) for inp in query] if type(query) == list else [build_prompt(query)]  
    return prompts 

def gpt2_prompt(query, tokenizer=None):
    def build_prompt(input_text):
        return input_text
    prompts = [build_prompt(inp) for inp in query] if type(query) == list else [build_prompt(query)]  
    return prompts 


def llm_prompts(lm_type, query, tokenizer, tokenize=True):
    if lm_type == "Llama-2-7b-chat-hf":
        prompts = llama2_prompt(query, tokenizer)
    elif lm_type == "Llama-2-13b-chat-hf":
        prompts = llama2_prompt(query, tokenizer)
    elif lm_type == "Llama-3-8B-Instruct":
        prompts = llama3_prompt(query, tokenizer)
    elif lm_type == "Llama-3-70B-Instruct":
        prompts = llama3_prompt(query, tokenizer)
    elif lm_type == "Mistral-7B-Instruct-v0.2":
        prompts = mistral_prompt(query, tokenizer)
    elif lm_type == "Mistral-7B-Instruct-v0.3":
        prompts = mistral_prompt(query, tokenizer)

    elif lm_type == "Llama-3.1-8B-Instruct":
        prompts = llama3_prompt(query, tokenizer)
    elif lm_type == "Llama-3.1-70B-Instruct":
        prompts = llama3_prompt(query, tokenizer)  

    elif lm_type == "gpt2":
        prompts = gpt2_prompt(query, tokenizer) 
    else:
        raise ValueError
    
    if tokenize and tokenizer:
        tokenizer.pad_token = tokenizer.eos_token
        model_inputs = tokenizer(prompts,  return_tensors="pt", add_special_tokens=False, padding = True, padding_side="left")
        return model_inputs
    else:
        return prompts
