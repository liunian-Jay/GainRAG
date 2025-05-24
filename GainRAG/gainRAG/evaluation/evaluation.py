import re
import string
import numpy as np
from collections import Counter

def get_evaluation(task, res_answer, gold_answers, regex = True):
  """ TODO: Different evaluations need to be adapted """
  if task == 'ARC_Challenge':
      score = accuracy(res_answer, gold_answers)
      return score, 0 
  else:
      score = em_max_over_ground_truths(res_answer, gold_answers, regex = regex)
      f1_score = f1_max_over_ground_truths(res_answer, gold_answers)
      return score, f1_score
  

#################################################
########        READER EVALUATION        ########
#################################################
def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
  """ EM: ground_truth is a str """
  return normalize_answer(prediction) == normalize_answer(ground_truth)

def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            normalize_answer(pattern),
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(normalize_answer(text)) is not None
    
def f1_score(prediction, ground_truth):
  """ F1: ground_truth is a str """
  prediction_tokens = normalize_answer(prediction).split()
  ground_truth_tokens = normalize_answer(ground_truth).split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
      return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1



def em_max_over_ground_truths(prediction, ground_truths, regex=True):
  """ EM: ground_truth is a list """
  return max([regex_match(prediction, gt) if regex else exact_match_score(prediction, gt) for gt in ground_truths])

def f1_max_over_ground_truths(prediction, ground_truths):
  """ F1: ground_truth is a list """
  return max([f1_score(prediction, gt) for gt in ground_truths])

def accuracy(prediction, ground_truths):
  """ ACC  """
  if type(prediction) is str and (prediction[0] == "#" or prediction[0] == ":") and len(prediction)>=1:
    prediction = prediction[1:]
  if len(prediction)>1:
    prediction = prediction.split(':')[0]
  if len(prediction)>1:
    prediction = prediction.split('.')[0]
  
  # print(prediction)

  for gt in ground_truths:
    if gt == prediction:
        return 1
  return 0

def compute_str_em(prediction, item):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """
    def exact_presence(short_answers, context):
      """Verify if any of the answers is present in the given context.
      Args:
          short_answers: list of short answers to look for in the context
          context: a paragraph to search for short answers
      Returns:
          true if any of the short answers is present in the context
      """

      n_short_answers = [normalize_answer(sa) for sa in short_answers]
      n_context = normalize_answer(context)

      for ans in n_short_answers:
          if ans in n_context:
              return True

      return False

    if 'qa_pairs' not in item or item['qa_pairs'] is None:
        return 0, 0

    loc_acc = []
    for qa_pair in item['qa_pairs']:
        loc_acc.append(exact_presence(qa_pair['short_answers'], prediction))
    acc = np.mean(loc_acc)
    hit = int(np.mean(loc_acc) == 1) 

    return acc, hit
