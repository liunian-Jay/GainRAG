import json
import math

input_file = '/path/filter_data_ppl.json'
output_file = '/path/train_selector_ppl_log.jsonl'

with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

new_data = []
for item in data:
    question = item['question']
    answers = item['answers']
    passages = item['passages']
    sorted_passages = sorted(passages, key=lambda x: x["retrieved_ppl"])
    texts = [passage['title']+'\n'+passage['text'] for passage in sorted_passages]
    scores= [-math.log(passage['retrieved_ppl']+1) for passage in sorted_passages]
    new_item = dict()
    new_item['query'] = question
    new_item['pos'] = texts[0:1]
    new_item['neg'] = texts[1:]
    new_item['pos_scores'] = scores[0:1]
    new_item['neg_scores'] = scores[1:]
    # new_item['prompt'] = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    new_item['prompt'] = "Given a query A and a passage B, determine whether the passage directly or indirectly contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    new_data.append(new_item)
    

with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(json.dumps(record, ensure_ascii=False) + '\n' for record in new_data)






