import os
import json
import torch
import logging
import pickle
import time
import glob
import numpy as np
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


import regex
import unicodedata
import collections
from functools import partial
from multiprocessing import Pool as ProcessPool
from typing import List


"""
Evaluation code from DPR: https://github.com/facebookresearch/DPR
"""

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



#########################################################
#########################################################
#########################################################
def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        print(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    print("Data indexing completed.")

def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids

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

def add_passages(data, passages, top_passages_and_scores):
    # add passages to original data
    merged_data = []
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        d["ctxs"] = [
            {
                "id": results_and_scores[0][c],
                "title": docs[c]["title"],
                "text": docs[c]["text"],
                "score": scores[c],
            }
            for c in range(ctxs_num)
        ]

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


#########################################################
####################### main ############################
#########################################################

import os
import sys
import time
import glob
from pathlib import Path
import jsonlines

from index import Indexer, load_passages
from modeling import Contriever

def embed_queries(queries, model, tokenizer,  batch_size=64):
    model.eval()
    embeddings = list()
    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            batch_texts = queries[i:i + batch_size]
            encoded_batch = tokenizer.batch_encode_plus(
                batch_texts,
                return_tensors="pt",
                max_length=512,
                padding=True,
                truncation=True,
            )
            encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
            output = model(**encoded_batch)
            embeddings.append(output)
        embeddings = torch.cat(embeddings, dim=0)
    print(f"Questions embeddings shape: {embeddings.size()}")
    return embeddings.cpu().numpy()
   
def main(args):

    print(f"Loading model from: {args.model_name_or_path}")
    model = Contriever.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model.eval()
    model = model.cuda()
   
    index = Indexer(args.projection_size, args.n_subquantizers, args.n_bits, args.use_gpu)
    input_paths = glob.glob(args.passages_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")
    
    if args.save_or_load_index and os.path.exists(index_path):
        index.deserialize_from(embeddings_dir)
    else:
        # index all passages
        print(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, args.indexing_batch_size)
        print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        if args.save_or_load_index:
            index.serialize(embeddings_dir)

    # load passages
    passages = load_passages(args.passages)
    passage_id_map = {x["id"]: x for x in passages}

    data_paths = glob.glob(args.data_path_or_dir)
    for path in data_paths:
        data = load_data(path)
        output_path = os.path.join(args.output_dir, os.path.basename(path))
        queries = [ex["question"] for ex in data]
        questions_embedding = embed_queries(queries, model, tokenizer)
        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
        add_passages(data, passage_id_map, top_ids_and_scores)
        hasanswer = validate(data, args.validation_workers)
        add_hasanswer(data, hasanswer)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with jsonlines.open(output_path, mode='w') as writer:
            for item in data:
                writer.write(item)
        print(f"Saved results to {output_path}")


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for retrieval configuration.")
    
    parser.add_argument("--projection_size", type=int, default=768, help="Projection size of the embeddings.")
    parser.add_argument("--n_subquantizers", type=int, default=0, help="Number of subquantizers.")
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits for quantization.")
    parser.add_argument("--indexing_batch_size", type=int, default=1000000, help="Batch size for indexing.")
    parser.add_argument("--validation_workers", type=int, default=8, help="Number of workers for validation.")
    parser.add_argument("--save_or_load_index",  type=lambda x: x.lower() == 'true', default=True, help="Save or load index (default: True).")
    parser.add_argument("--passages_embeddings", type=str, default=None, help="Path to passages embeddings.")
    parser.add_argument("--passages", type=str, default=None, help="Path to passages file.")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Path to the model.")
    parser.add_argument("--data_path_or_dir", type=str, default=None, help="Path or dir to the input data.")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to the output directory.")
    parser.add_argument("--n_docs", type=int, default=100, help="Number of documents to retrieve.")
    parser.add_argument("--use_gpu", type=lambda x: x.lower() == 'true', default=True, help="Enable GPU for computation (default: True).")

    # Parsing arguments
    args = parser.parse_args()
    main(args)
