import csv
import os
import json
import logging
import pickle
import torch
from typing import List, Tuple
logger = logging.getLogger(__name__)


from normalize import normalize

# Used for passage retrieval
def load_passages(path):
    if not os.path.exists(path):
        logger.info(f"{path} does not exist")
        return
    logger.info(f"Loading passages from: {path}")
    passages = []
    with open(path) as fin:
        if path.endswith(".jsonl"):
            for k, line in enumerate(fin):
                ex = json.loads(line)
                passages.append(ex)
        else:
            reader = csv.reader(fin, delimiter="\t")
            for k, row in enumerate(reader):
                if not row[0] == "id":
                    ex = {"id": row[0], "title": row[2], "text": row[1]}
                    passages.append(ex)
    return passages

def embed_passages(
    passages: List[dict],
    model,
    tokenizer,
    per_gpu_batch_size: int = 32,
    passage_maxlength: int = 512,
    no_title: bool = False,
    lowercase: bool = False,
    normalize_text = True,
) -> Tuple[List[str], torch.Tensor]:
    """Processes passages and generates embeddings in batches.
    Returns:
        Tuple[List[str], torch.Tensor]: List of passage IDs and their embeddings.
    """

    def prepare_batch(batch):
        """Prepare batch texts and IDs."""
        batch_texts, batch_ids = [], []
        for passage in batch:
            text = passage["text"]
            if not no_title and "title" in passage:
                text = f"{passage['title']} {text}"
            if lowercase:
                text = text.lower()
            if normalize_text:
                text = normalize(text)
            batch_texts.append(text)
            batch_ids.append(passage["id"])
        return batch_ids, batch_texts

    def encode_batch(batch_texts):
        """Encode a batch of texts and store their embeddings."""
        encoded_batch = tokenizer.batch_encode_plus(
            batch_texts,
            return_tensors="pt",
            max_length=passage_maxlength,
            padding=True,
            truncation=True,
        )
        encoded_batch = {key: value.cuda() for key, value in encoded_batch.items()}
        embeddings = model(**encoded_batch).cpu()
        return embeddings

    all_ids, all_embeddings = [], []
    with torch.no_grad():
        # Split passages into batches
        for start_idx in range(0, len(passages), per_gpu_batch_size):
            batch = passages[start_idx:start_idx + per_gpu_batch_size]
            batch_ids, batch_texts = prepare_batch(batch)

            embeddings = encode_batch(batch_texts)
            all_ids.extend(batch_ids)
            all_embeddings.append(embeddings)

            if len(all_ids) % 100000 == 0:
                print(f"Encoded passages: {len(all_ids)}")

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    return all_ids, all_embeddings

def save_embedings(allids, allembeddings, output_dir, prefix = 'passages'):
    save_file = os.path.join(output_dir, prefix)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving {len(allids)} passage embeddings to {save_file}.")
    with open(save_file, mode="wb") as f:
        pickle.dump((allids, allembeddings), f)
    print(f"Total passages processed {len(allids)}. Written to {save_file}.")


def main():
    output_dir = ''
    passages_path = ''

    model = ''
    tokenizer = ''

    passages = load_passages(passages_path)
    allids, allembeddings = embed_passages(
        passages, model, tokenizer,
        per_gpu_batch_size = 128,
        passage_maxlength=512,
        no_title=True,
        lowercase=True,
        normalize_text=True)
    save_embedings(allids, allembeddings, output_dir)
