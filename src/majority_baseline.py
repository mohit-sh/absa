import json
from collections import Counter
from sklearn.metrics import accuracy_score

REST_TRAIN_FILE= "../data/se14-task4/processed/rest/train.json"
LAPTOP_TRAIN_FILE = "../data/se14-task4/processed/laptop/train.json"

REST_TEST_FILE = "../data/se14-task4/processed/rest/test.json"
LAPTOP_TEST_FILE = "../data/se14-task4/processed/laptop/test.json"

rest_train_counter = Counter([v["polarity"] for k, v in json.load(open(REST_TRAIN_FILE)).items()])
laptop_train_counter = Counter([v["polarity"] for k, v in json.load(open(LAPTOP_TRAIN_FILE)).items()])

rest_majority_class = max(rest_train_counter.items(), key= lambda d: d[-1])[0]
laptop_majority_class = max(rest_train_counter.items(), key= lambda d: d[-1])[0]

POLARITY_MAP = {"positive" : 2, "neutral" : 1, "negative" : 0}

# Compute majority voting accuracy for restaurant dataset

GT_LABELS = [POLARITY_MAP[v["polarity"]] for k, v in json.load(open(REST_TEST_FILE)).items()]
PRED_LABELS = [POLARITY_MAP[rest_majority_class] for _ in range(len(GT_LABELS))]

acc = accuracy_score(GT_LABELS, PRED_LABELS) 
print(f"REST::\tACC:\t{acc:.4f}")
# Compute majority voting accuracy for laptop dataset
GT_LABELS = [POLARITY_MAP[v["polarity"]] for k, v in json.load(open(LAPTOP_TEST_FILE)).items()]
PRED_LABELS = [POLARITY_MAP[laptop_majority_class] for _ in range(len(GT_LABELS))]

acc = accuracy_score(GT_LABELS, PRED_LABELS) 
print(f"LAPT::\tACC:\t{acc:.4f}")
