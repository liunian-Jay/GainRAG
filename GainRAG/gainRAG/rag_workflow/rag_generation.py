from tqdm import tqdm

from llm_inference.build_prompts import llm_prompts
from llm_inference.inference_vllm import load_model_vllm, llm_response_vllm
from evaluation.evaluation import f1_max_over_ground_truths
from evaluation.evaluation import em_max_over_ground_truths

from rag_workflow.prompts import load_data, build_input_with_retrieval
from rag_workflow.prompts import preprocess_ARC_Challenge, preprocess_PubHealth


def eval_generate(data, task='HotpotQA', K=1, lm_type = 'Llama-3-8B-Instruct', batch_size = 256):
    if task == 'ARC_Challenge':
        data = preprocess_ARC_Challenge(data)
    if task == 'PubHealth':
        data = preprocess_PubHealth(data)
    
    retrieved_inputs = build_input_with_retrieval(data, K, task)
    total_items = len(retrieved_inputs)
    model, tokenizer = load_model_vllm(lm_type)
    scores = {'with_retrieval': 0}
    f1_scores = {'with_retrieval': 0}

    count = 0
    for start in tqdm(range(0, len(retrieved_inputs), batch_size)):
        end = min(start+batch_size, len(retrieved_inputs))
        retrieved_items = retrieved_inputs[start:end]
        retrieved_prompts = [item['prompt'] for item in retrieved_items]
        retrieved_tokens = llm_prompts(lm_type, retrieved_prompts, tokenizer, tokenize=False)
        retrieved_response = llm_response_vllm(model, retrieved_tokens)

        for i, retrieved_item in enumerate(retrieved_items):
            em_score = em_max_over_ground_truths(retrieved_response[i], retrieved_item['answers'], regex=True)
            f1_score = f1_max_over_ground_truths(retrieved_response[i], retrieved_item['answers'])
            scores['with_retrieval'] += em_score
            f1_scores['with_retrieval'] += f1_score
            print(f"{start+i}: With Retrieval - EM: {scores['with_retrieval']}, F1: {f1_scores['with_retrieval']}")


    avg_em = scores['with_retrieval'] / total_items
    avg_f1 = f1_scores['with_retrieval'] / total_items
    print(f'Final Results: \nWith Retrieval EM: {avg_em*100} F1: {avg_f1*100}\n')


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for retrieval configuration.")
    parser.add_argument("--batch_size", type=int, default=256, help="Path to the model.")
    parser.add_argument("--K_docs", type=int, default=1, help="Number of documents to retrieve.")
    parser.add_argument("--task", type=str, default='HotpotQA', help="Task prompt template to use.")
    parser.add_argument("--lm_type", type=str, default='Llama-3-8B-Instruct', help="LLM to use.")
    parser.add_argument("--data_path", type=str, default=None, help="Path or dir to the input data.")
    args = parser.parse_args()

    data = load_data(args.data_path)
    for i in [1,2,3,4,5]:
        hit = sum(any(passage['hasanswer'] for passage in item['ctxs'][:i]) for item in data)
        print(f'Recall@{i}:', hit/len(data))
    eval_generate(data, args.task, K=args.K_docs, lm_type=args.lm_type, batch_size=args.batch_size)
