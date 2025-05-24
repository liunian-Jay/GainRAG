import json

#################################################
################### Load data ###################
#################################################
def read_jsonl(file_path):
    """Read a JSONL file and return a list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def write_jsonl(data, output_path):
    """Write a list of dictionaries to a JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(json.dumps(record, ensure_ascii=False) + '\n' for record in data)

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


#################################################
#################### prompts ####################
#################################################
INSTRUCTION_PROMPT = {
    "Instruction_Without_Retrieval": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n"
    ),
    "Instruction_With_Retrieval": (
        "{passage}\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n"
    ),
}

TASK_INST = {
    "NaturalQA": "Answer the question below concisely in a few words.",
    "TriviaQA": "Answer the question below concisely in a few words.",
    "WebQuestions": "Answer the question below concisely in a few words.",
    "2WikiMultiHopQA":"Answer the question below concisely in a few words.",
    "SQuAD": "Answer the question below concisely in a few words.",
    "HotpotQA": "Answer the question below concisely in a few words.",
}


def preprocess_ARC_Challenge(data):
    choice_dict = {'A':'A','B':'B','C':'C','D':'D','E':'E','1':'A','2':'B','3':'C','4':'D','5':'E'}
    new_data = []
    for item in data:
        choice_text = ''
        choices = item["choices"]
        for i in range(len(choices["label"])):
            answer_key = choices["label"][i]
            answer_key = choice_dict[answer_key]
            text = choices["text"][i]
            choice_text += "\n{0}: {1}".format(answer_key, text)

        item["question"] = item["question"] + choice_text
        item["answers"] = [item["answerKey"]]
        new_data.append(item)
    return new_data

def preprocess_PubHealth(data):
    new_data = []
    for item in data:
        answer = 'true' if item['label']=='SUPPORTS' else 'false'
        item["question"] = item["claim"]
        item["answers"] = [answer]
        new_data.append(item)
    return new_data


###############################################
################# base build ##################
###############################################
def build_input_without_retrieval(data, task='ARC_Challenge'):
    for item in data:
        item['prompt'] = INSTRUCTION_PROMPT['Instruction_Without_Retrieval'].format_map({'instruction':TASK_INST[task], 'input': item['question']})
    return data

def build_input_with_retrieval(data, k=5, task='ARC_Challenge'):
    import random
    for item in data:
        passages = [passage for passage in item['ctxs'][:k]] 
        evidences = [f'Passage #{i+1}{passage["title"]}\nPassage #{i+1}{passage["text"]}'  for i, passage in enumerate(passages)]
        evidence = '\n\n'.join(evidences)
        item['prompt'] = item['prompt'] = INSTRUCTION_PROMPT['Instruction_With_Retrieval'].format_map({'passage':evidence,'instruction':TASK_INST[task], 'input': item['question']})
    return data

################# get input ###################
def get_input(data_path, retrieval = False, k = 5, task = None):
    data = read_jsonl(data_path)

    task =  data_path.split('/')[-1].split('.')[0] if  not task else task
    if task == 'ARC_Challenge':
        data = preprocess_ARC_Challenge(data)
    if task == 'PubHealth':
        data = preprocess_PubHealth(data)

    return build_input_with_retrieval(data, k, task) if retrieval else build_input_without_retrieval(data, task)

###############################################
################## K-th build  ################
###############################################
def build_input_with_retrieval_K_th(data, k_indices=[0,1,2,3,4,5], task='ARC_Challenge'):
    new_data = list()
    for item in data:
        new_item = dict()
        new_item['question'] = item['question']
        new_item['answers'] = item['answers']
        new_item['prompt_standard'] = INSTRUCTION_PROMPT['Instruction_Without_Retrieval'].format_map({'instruction':TASK_INST[task], 'input': item['question']})

        new_item['passages'] = list()
        for i in k_indices:
            passage = item['ctxs'][i]
            evidence = 'Passage #1  title:' + passage['title'] + '\n' + 'Passage #1  text:' + passage['text'] + '\n\n'
            
            passage_item = dict()
            passage_item['title'] = passage['title']
            passage_item['text'] = passage['text']
            passage_item['has_answer'] = passage['hasanswer']
            passage_item['retrieval_score'] = passage['score']
            passage_item['prompt_retrieved'] = INSTRUCTION_PROMPT['Instruction_With_Retrieval'].format_map({'passage':evidence,'instruction':TASK_INST[task], 'input': item['question']})
            new_item['passages'].append(passage_item)
            
        new_data.append(new_item)
    return new_data

################# get input K-th ###################
def get_input_with_R_Kth(data_path, k = 5, task = 'HotpotQA'):
    data = load_data(data_path)

    if task == 'ARC_Challenge':
        data = preprocess_ARC_Challenge(data)
    if task == 'PubHealth':
        data = preprocess_PubHealth(data)  

    return build_input_with_retrieval_K_th(data, list(range(k)), task)


