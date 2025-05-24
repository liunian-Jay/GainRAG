import os
import json
import logging
import regex
import unicodedata
import collections
import numpy as np
from tqdm import tqdm
from typing import List
from functools import partial
from multiprocessing import Pool as ProcessPool

##########################################################
######################## check ###########################
##########################################################
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
QAMatchStats = collections.namedtuple('QAMatchStats', ['top_k_hits', 'questions_doc_hits'])

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

def _normalize(text):
    return unicodedata.normalize('NFD', text)

def check_answer(example, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []

    for i, doc in enumerate(ctxs):
        text = doc['text']

        if text is None:  # cannot find the document for some reason
            logger.warning("no doc in db")
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits

def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False

def calculate_matches(data: List, workers_num: int):
    """
    Evaluates answers presence in the set of documents. This function is supposed to be used with a large collection of
    documents and results. It internally forks multiple sub-processes for evaluation and then merges results
    :param all_docs: dictionary of the entire documents database. doc_id -> (doc_text, title)
    :param answers: list of answers's list. One list per question
    :param closest_docs: document ids of the top results along with their scores
    :param workers_num: amount of parallel threads to process data
    :param match_type: type of answer matching. Refer to has_answer code for available options
    :return: matching information tuple.
    top_k_hits - a list where the index is the amount of top documents retrieved and the value is the total amount of
    valid matches across an entire dataset.
    questions_doc_hits - more detailed info with answer matches for every question and every retrieved document
    """

    logger.info('Matching answers in top docs...')

    tokenizer = SimpleTokenizer()
    get_score_partial = partial(check_answer, tokenizer=tokenizer)

    processes = ProcessPool(processes=workers_num)
    scores = processes.map(get_score_partial, data)

    logger.info('Per question validation results len=%d', len(scores))

    n_docs = len(data[0]['ctxs'])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    return QAMatchStats(top_k_hits, scores)

def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    print("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits]
    message = ""
    for k in [1, 5, 10, 20, 25, 50,100]:
        if k <= len(top_k_hits):
            message += f"R@{k}: {top_k_hits[k-1]} "
    print(message)
    return match_stats.questions_doc_hits

def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]

def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data

##########################################################
##################### LLM InfoInside #####################
##########################################################
from llm_inference.build_prompts import llm_prompts
from llm_inference.inference_vllm import load_model_vllm, llm_response_vllm

def generate_passages(model, tokenizer, data):
    # instruction = 'Please provide context for this question in 100 words or less. Do not respond with anything other than context. If you do not know or are unsure, please generate "N/A" directly. \n\n'
    instruction = 'Please provide background for the question below in 100 words. Do not respond with anything other than background. If you do not know or are unsure, please generate "N/A" directly. \n\n'
    prompts = [f'{instruction} Question: {item["question"]}' for item in data]
    tokens = llm_prompts('Llama-3-8B-Instruct', prompts, tokenizer, tokenize=False)
    responses = llm_response_vllm(model, tokens)
    # print(responses)
    for passage,item in zip(responses,data):
        item['ctxs'].append({'title':'','text':passage})

##########################################################
###################### selection #########################
##########################################################
from FlagEmbedding.inference.reranker.encoder_only.base import BaseReranker

def get_top_k_passages(query, passages, model, k=5):
    pairs = [(query,passage) for passage in passages]
    scores = model.compute_score(pairs)
    # Get top-k passages
    scores_np = np.array(scores)
    top_k_indices = np.argsort(scores_np)[-k:][::-1]  # Get the indices of the k largest elements
    top_k_scores = scores_np[top_k_indices]
    print(top_k_indices)
    return top_k_indices, top_k_scores

def select_data(data, K, model_name_or_path, output_path):
    model = BaseReranker(model_name_or_path, use_fp16=True, devices=['cuda:0'])
    
    use_internal = 0
    for item in tqdm(data):
        query = item['question']
        passages = [i['title']+'\n'+i['text'] for i in item['ctxs']]
        top_k_indices, top_k_scores = get_top_k_passages(query, passages, model, k=K)
        use_internal += 100 in top_k_indices[:1]
        item['ctxs'] = [item['ctxs'][i] for i in top_k_indices]
    print('x'*100,'\n', use_internal, '\n','x'*100)

    hasanswer = validate(data, workers_num=8)
    add_hasanswer(data, hasanswer)
    for i in [1,3,5]:
        hit = sum(any(passage['hasanswer'] for passage in item['ctxs'][:i]) for item in data)
        print(f'Rerank Recall@{i}:', hit/len(data))

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print('Save OK!')


##########################################################
########################## main ##########################
##########################################################
def main(args):
    # Loda data
    data = load_data(args.data_path)

    # Initial search 100
    for item in data:
        item['ctxs'] = item['ctxs'][:100]
    for i in [1,3,5]:
        hit = sum(any(passage['hasanswer'] for passage in item['ctxs'][:i]) for item in data)
        print(f'Origin Recall@{i}:', hit/len(data))

    # Generate internal knowledge
    model, tokenizer = load_model_vllm('Llama-3-8B-Instruct')
    generate_passages(model, tokenizer, data)
    # Select One
    select_data(data, K=args.K_docs, model_name_or_path=args.model_name_or_path, output_path=args.output_path)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for retrieval configuration.")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Path to the model.")
    parser.add_argument("--data_path", type=str, default=None, help="Path or dir to the input data.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to the output file.")
    parser.add_argument("--K_docs", type=int, default=1, help="Number of documents to retrieve.")

    # Parsing arguments
    args = parser.parse_args()
    main(args)