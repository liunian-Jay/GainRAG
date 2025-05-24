import json
import os


directory = '/path/data'
filenames = ['xxx']
output_file = '/path/data/merged_data.json'


merged_data = list()
for filename in filenames:
    filename += '.json'
    filepath = os.path.join(directory, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    new_data = list()
    for item in data:
        passages = item['passages']
        sorted_passages = sorted(passages, key=lambda x: x["PPL_CD"])
        silver_passage = sorted_passages[0]
        if silver_passage['EM_with_retrieval']:
            new_data.append(item)
    print(filename, ': ',len(new_data))
    merged_data.extend(new_data)
print('data items:', len(merged_data))

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)
