import torch
import torch.nn.functional as F
from typing import Union, List, Optional
import copy
import math
import numpy as np

class ContrastiveTool:
    def __init__(self):
        pass

    def _top_p_sampling(self, 
                        logits: torch.Tensor, 
                        top_p: float = 0.9, 
                        filter_value: float = -float("Inf"), 
                        min_tokens_to_keep: int = 1
                        ) -> torch.Tensor :

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep - 1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        return logits


    def _top_k_sampling(self, 
                        logits: torch.Tensor, 
                        top_k: int = 20, 
                        filter_value: float = -float("Inf"), 
                        min_tokens_to_keep: int = 1
                        ) -> torch.Tensor :

        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None] # * logit 값이 Top-k의 토큰 중 가장 작은 값보다 작은 토큰의 인덱스 반환 
        logits[indices_to_remove] = filter_value
        return logits


    def predict_next_token(self, 
                           logits: torch.Tensor, 
                           decoding_strategy: str, 
                           top_p: float, 
                           top_k: int, 
                           use_repetition_penalty: bool, 
                           repetition_penalty_value: float, 
                           generated_tokens: List[set] = None
                           ) -> torch.Tensor :

        # * Repetitin Penalty 참고 코드 : https://huggingface.co/transformers/v2.11.0/_modules/transformers/modeling_utils.html#PreTrainedModel.enforce_repetition_penalty_
        if use_repetition_penalty:
            assert repetition_penalty_value >= 1.0, "Repetition penalty must be >= 1."
            mask = torch.zeros_like(logits)
            for i, token_set in enumerate(generated_tokens):
                mask[i, list(token_set)] = 1.0
            penalty = torch.where(mask == 1.0, repetition_penalty_value, 1.0) # generated_tokens에 있는 토큰들은 penalty를 repetition_penalty_value로, 없는 토큰들은 1.0(현상 유지)으로 설정
            logits *= torch.where(logits < 0, penalty, 1.0/penalty) # if logit is smaller than 0, multiply with penalty, else divide by penalty
        
        if decoding_strategy == 'top_p':
            assert top_p is not None, "top_p must be provided for top_p sampling"
            logits = self._top_p_sampling(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze()

        elif decoding_strategy == 'top_k':
            assert top_k is not None, "top_k must be provided for top_k sampling"
            logits = self._top_k_sampling(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze()

        elif decoding_strategy == 'greedy':
            next_token = torch.argmax(logits, dim=-1)

        return next_token


    def contrastive_generate(self, 
                model,
                tokenizer,
                tokenized_inputs, 
                tokenized_inputs_with_contexts,

                alpha: float = 0.5,
                max_new_length: int = 256,
                decoding_strategy: str = 'top_p',
                top_p_value: float = 0.9,
                top_k_value: int = 20,
                use_repetition_penalty: bool = False, 
                repetition_penalty_value: float = 1.0,
                ) -> List[List[int]]:
        
        # Tokenize 'input_texts' and create attention masks
        input_ids = copy.deepcopy(tokenized_inputs['input_ids'])
        attention_mask = copy.deepcopy(tokenized_inputs['attention_mask'])

        # Tokenize 'contexts' after concatenating with 'input_ids' if 'contexts' is not None
        if tokenized_inputs_with_contexts:
            input_ids_with_contexts = copy.deepcopy(tokenized_inputs_with_contexts['input_ids'])
            attention_mask_with_contexts = copy.deepcopy(tokenized_inputs_with_contexts['attention_mask'])

        # Initialize variables for generation loop
        cur_len = 0
        batch_size = len(input_ids)
        unfinished_sents = input_ids_with_contexts.new(batch_size).fill_(1)
        sent_lengths = input_ids_with_contexts.new(batch_size).fill_(max_new_length)

        generated_tokens = [[] for _ in range(batch_size)] # e.g., [[4132, 102, 29402], [2378, 7893, 23001]]

        # Generate tokens
        with torch.no_grad():
            for cur_len in range(1, max_new_length + 1):
                
                outputs = model(input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :] # (batch_size, vocab_size)

                # * Context-aware Decoding
                if tokenized_inputs_with_contexts:
                    outputs_with_contexts = model(input_ids_with_contexts, attention_mask=attention_mask_with_contexts)
                    next_token_logits_with_contexts = outputs_with_contexts.logits[:, -1, :]
                    next_token_logits = (1 + alpha) * next_token_logits_with_contexts - alpha * next_token_logits

                # Predict next token according to decoding strategy
                next_token = self.predict_next_token(logits=next_token_logits, 
                                                    decoding_strategy=decoding_strategy, 
                                                    top_p=top_p_value, 
                                                    top_k=top_k_value, 
                                                    use_repetition_penalty=use_repetition_penalty, 
                                                    repetition_penalty_value=repetition_penalty_value, 
                                                    generated_tokens=[set(tokens) for tokens in generated_tokens])

                # Handle EOS token and padding
                if tokenizer.eos_token_id is not None:
                    tokens_to_add = next_token * unfinished_sents + (tokenizer.pad_token_id) * (1 - unfinished_sents)
                else:
                    tokens_to_add = next_token

                # Update input_ids and attention masks for the next forward pass
                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, unfinished_sents.unsqueeze(-1)], dim=-1)

                input_ids_with_contexts = torch.cat([input_ids_with_contexts, tokens_to_add.unsqueeze(-1)], dim=-1)
                attention_mask_with_contexts = torch.cat([attention_mask_with_contexts, unfinished_sents.unsqueeze(-1)], dim=-1)

                # Update generated tokens and check for completion
                for i, token in enumerate(tokens_to_add.tolist()):
                    if unfinished_sents[i] == 1:
                        generated_tokens[i].append(token)

                # Check for sentences that are finished
                if tokenizer.eos_token_id is not None:
                    eos_in_sents = tokens_to_add == tokenizer.eos_token_id
                    is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                    sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                    unfinished_sents.mul_((~eos_in_sents).long())

                # Break if all sentences are finished : stop when there is a EOS token in each sentence, or if we exceed the maximul length
                if unfinished_sents.max() == 0:
                    break

        # Return the generated tokens
        return generated_tokens
    

    def contrastive_PPL(self, 
                model,
                tokenizer, 
                query_prompt: str,                  # 问题提示（不带上下文）
                query_prompt_with_contexts: str,    # 带上下文的提示
                gold_answer: str,                   # 正确答案
                alpha: float = 0.5                  # contrastive的权重系数
                ) -> int:
        
        # Tokenize 'input_texts' and create attention masks
        inputs = tokenizer(query_prompt,  return_tensors="pt", add_special_tokens=False).to('cuda')
        inputs_with_contexts = tokenizer(query_prompt_with_contexts,  return_tensors="pt", add_special_tokens=False).to('cuda')
        answer_ids = tokenizer(gold_answer,  return_tensors="pt", add_special_tokens=False)["input_ids"].to('cuda')

        # print(inputs['input_ids'].size())
        # print(answer_ids.size())

        answer_log_loss = 0
        # Generate tokens
        with torch.no_grad():
            for i in range(answer_ids.size(1)): # 遍历答案的每个token
                answer_id = answer_ids[:, i]    # 获取当前答案的 token id（shape 为 [batch_size]）

                # Update input_ids and attention masks for the next forward pass
                inputs['input_ids'] = torch.cat([inputs['input_ids'], answer_id.unsqueeze(-1)], dim=-1)
                inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones_like(answer_id).unsqueeze(-1)], dim=-1)
                inputs_with_contexts['input_ids'] = torch.cat([inputs_with_contexts['input_ids'], answer_id.unsqueeze(-1)], dim=-1)
                inputs_with_contexts['attention_mask'] = torch.cat([inputs_with_contexts['attention_mask'], torch.ones_like(answer_id).unsqueeze(-1)], dim=-1)
                
                # contrastive 
                outputs = model(**inputs)
                current_token_logits = outputs.logits[:, -2, :]     # 形状 [batch_size, vocab_size]

                outputs_with_contexts = model(**inputs_with_contexts)
                current_token_logits_with_contexts = outputs_with_contexts.logits[:, -2, :]

                # 使用对比解码公式计算新 logit
                # current_token_logits = (1 + alpha) * current_token_logits_with_contexts - alpha * current_token_logits
                current_token_logits =  current_token_logits_with_contexts - current_token_logits
            
                # Predict the answer token logits
                log_probs = current_token_logits.log_softmax(dim=-1)# 计算 log softmax，形状 [batch_size, vocab_size]
                logprob = log_probs[0, answer_id.item()].item()     # 形状 [batch_size, vocab_size]
                answer_log_loss -= logprob

        # 平均 log-loss
        average_log_loss = answer_log_loss / answer_ids.size(1)
        # 转换为困惑度
        ppl = math.exp(average_log_loss)
        return ppl
    

    def get_gold_answer_PPL(self, model, tokenizer, query_prompt, gold_answer):
        """ 获取gold Answer的PPL"""
        # 拼接输入
        all_prompt = query_prompt + gold_answer
        all_input_ids = tokenizer(all_prompt,  return_tensors="pt", add_special_tokens=False)["input_ids"].to('cuda')
        answer_input_ids = tokenizer(gold_answer,  return_tensors="pt", add_special_tokens=False)["input_ids"].to('cuda')

        # 获取模型的输出
        with torch.no_grad():
            outputs = model(all_input_ids, labels=all_input_ids)
            log_probs = outputs.logits.log_softmax(dim=-1)      # 计算 log 概率

        prompt_length = len(tokenizer(query_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0])
        log_probs_answer = log_probs[:, prompt_length-1:-1]     # Answer 范围

        # 计算 log-loss
        answer_log_loss = 0
        for i in range(answer_input_ids.size(1)):
            token_id = answer_input_ids[0, i]
            logprob = log_probs_answer[0, i, token_id].item()
            answer_log_loss -= logprob

        # 平均 log-loss
        average_log_loss = answer_log_loss / answer_input_ids.size(1)
        # 转换为困惑度
        ppl = math.exp(average_log_loss)
        return ppl
    


    def contrastive_PPL_multi(self, 
                model,
                tokenizer, 
                query_prompt: str,                  # 问题提示（不带上下文）
                query_prompt_w_contexts: List[str], # 带上下文的提示
                gold_answer: str,                   # 正确答案
                alpha: float = 0.5                  # contrastive的权重系数
                ) -> List[int]:
        # Tokenize 'input_texts' and create attention masks
        inputs = tokenizer(query_prompt,  return_tensors="pt", add_special_tokens=False).to('cuda')
        answer_ids = tokenizer(gold_answer,  return_tensors="pt", add_special_tokens=False)["input_ids"].to('cuda')

        inputs_w_contexts_list = []
        for x in query_prompt_w_contexts:
            inputs_w_contexts_list.append(tokenizer(x,  return_tensors="pt", add_special_tokens=False).to('cuda'))

        answer_log_loss = [0]*len(query_prompt_w_contexts)
        
        # Generate tokens
        with torch.no_grad():
            for i in range(answer_ids.size(1)): # 遍历答案的每个token
                answer_id = answer_ids[:, i]    # 获取当前答案的 token id（shape 为 [batch_size]）

                # Update input_ids and attention masks for the next forward pass
                inputs['input_ids'] = torch.cat([inputs['input_ids'], answer_id.unsqueeze(-1)], dim=-1)
                inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones_like(answer_id).unsqueeze(-1)], dim=-1)
                outputs = model(**inputs)
                token_logits_origin = outputs.logits[:, -2, :]     # 形状 [batch_size, vocab_size]

                token_logits_w_contexts_list = list()
                for inputs_w_contexts in inputs_w_contexts_list:
                    inputs_w_contexts['input_ids'] = torch.cat([inputs_w_contexts['input_ids'], answer_id.unsqueeze(-1)], dim=-1)
                    inputs_w_contexts['attention_mask'] = torch.cat([inputs_w_contexts['attention_mask'], torch.ones_like(answer_id).unsqueeze(-1)], dim=-1)
                    outputs_w_contexts = model(**inputs_w_contexts)
                    token_logits_w_contexts = outputs_w_contexts.logits[:, -2, :]
                    token_logits_w_contexts_list.append(token_logits_w_contexts)
                
                for i in range(len(token_logits_w_contexts_list)):
                    token_logits_w_contexts = token_logits_w_contexts_list[i]

                    # Contrastive Decoding
                    token_logits = (1 + alpha) * token_logits_w_contexts - alpha * token_logits_origin
                    # token_logits =  token_logits_w_contexts - token_logits_origin

                    # Predict the answer token logits
                    log_probs = token_logits.log_softmax(dim=-1)        # 计算 log softmax，形状 [batch_size, vocab_size]
                    logprob = log_probs[0, answer_id.item()].item()     # 形状 [batch_size, vocab_size]
                    answer_log_loss[i] -= logprob

        # Compute PPL
        average_log_loss = np.array(answer_log_loss) / answer_ids.size(1)
        ppl_list = np.exp(average_log_loss).tolist()
        return ppl_list
    
    def contrastive_PPL_multi_pro(self, 
                model,
                tokenizer, 
                query_prompt: str,                  # 问题提示（不带上下文）
                query_prompt_w_contexts: List[str], # 带上下文的提示
                gold_answer: str,                   # 正确答案
                alpha: float = 0.5                  # contrastive的权重系数
                ) -> List[int]:
        
        # Tokenize 'input_texts' and create attention masks
        inputs = tokenizer(query_prompt,  return_tensors="pt", add_special_tokens=False).to('cuda')
        answer_ids = tokenizer(gold_answer,  return_tensors="pt", add_special_tokens=False)["input_ids"].to('cuda')

        inputs_w_contexts_list = []
        for x in query_prompt_w_contexts:
            inputs_w_contexts_list.append(tokenizer(x,  return_tensors="pt", add_special_tokens=False).to('cuda'))

        answer_log_loss = [0]*len(query_prompt_w_contexts)
        
        # Generate tokens
        with torch.no_grad():
            inputs['input_ids'] = torch.cat([inputs['input_ids'], answer_ids], dim=-1)
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones_like(answer_ids)], dim=-1)
            outputs = model(**inputs)
            answer_logits_origin = outputs.logits[:, -(1+answer_ids.size(1)):-1, :]     # 形状 [batch_size=1, answer_ids.size(1), vocab_size]

            answer_logits_w_contexts_list = list()
            for inputs_w_contexts in inputs_w_contexts_list:
                inputs_w_contexts['input_ids'] = torch.cat([inputs_w_contexts['input_ids'], answer_ids], dim=-1)
                inputs_w_contexts['attention_mask'] = torch.cat([inputs_w_contexts['attention_mask'], torch.ones_like(answer_ids)], dim=-1)
                outputs_w_contexts = model(**inputs_w_contexts)
                answer_logits_w_contexts = outputs_w_contexts.logits[:, -(1+answer_ids.size(1)):-1, :]
                answer_logits_w_contexts_list.append(answer_logits_w_contexts)
            
            for i in range(len(answer_logits_w_contexts_list)):
                answer_logits_w_contexts = answer_logits_w_contexts_list[i]
                # Contrastive Decoding
                answer_logits = (1 + alpha) * answer_logits_w_contexts - alpha * answer_logits_origin

                # Predict the answer token logits
                log_probs = answer_logits.log_softmax(dim=-1)[0]    # 计算 log softmax，形状 [answer_ids.size(1), vocab_size]
                for j in range(answer_ids.size(1)):
                    answer_id = answer_ids[0, j]
                    logprob = log_probs[j, answer_id.item()].item() # 数值      
                    answer_log_loss[i] -= logprob

        # Compute PPL
        average_log_loss = np.array(answer_log_loss) / answer_ids.size(1)
        ppl_list = np.exp(average_log_loss).tolist()
        return ppl_list