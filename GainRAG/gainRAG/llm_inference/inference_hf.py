import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(lm_type):
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

    elif lm_type == "gpt2":
        model_name_or_path = os.path.abspath("TODOpath/gpt2")   
    else:
        raise ValueError
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def llm_response(inputs, model, tokenizer):
    """ Call generate to generate a reply """
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            return_dict_in_generate=True,
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,

            eos_token_id = tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    outputs = tokenizer.batch_decode(generate_ids.sequences[:,inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    for i, ans in enumerate(outputs):
        if ans.strip() == "":
            outputs[i] = 'N/A'
    return outputs


#############################################################################
###################### Custom step-by-step generation #######################
#############################################################################

def get_topk_tokens(model, inputs, num_branches=5):
    """ Get the initial k tokens, their probability values, indexes, and probability distributions """
    # Generate logits for the next token
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
    # Use softmax to convert logits into probabilities
    probabilities = torch.softmax(next_token_logits, dim = -1)
    # Get top-k tokens and their probabilities
    topk_values, topk_indices = torch.topk(probabilities, num_branches)
    return topk_values, topk_indices, probabilities


def generate_response(model, tokenizer, inputs, max_length=500):
    """ K decoding, then greedily generating the complete reply """
    response = []
    for _ in range(max_length):
        # Get the initial k tokens
        _, topk_indices, _ = get_topk_tokens(model,  inputs,  num_branches=1)
        # Append the token with the highest probability to the response
        response.append(topk_indices[:, 0].unsqueeze(-1))
        # Stop at the end of the sequence marker
        if topk_indices[:, 0] == tokenizer.eos_token_id:
            break
        # Add token to the input of the next iteration
        inputs['input_ids'] = torch.cat([inputs['input_ids'], topk_indices[:, 0].unsqueeze(-1)], dim=1)
    
    response = torch.cat(response, dim=1)
    output = tokenizer.batch_decode(response)#[0]
    return output



