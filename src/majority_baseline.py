import json
import logging
from enum import Enum
import numpy as np
import operator

from collections import Counter
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Sentiment(Enum):
    NEGATIVE = 0
    NEUTRAL = 1
    POSITIVE = 2

class Split(Enum):
    TRAIN = "train"
    TEST = "test"
    DEV = "dev"

class Dataset(Enum):
    REST = "Restaurant"
    LAPTOP = "Laptop"

POLARITY_MAP = {"positive" : Sentiment.POSITIVE, "neutral" : Sentiment.NEUTRAL, "negative" : Sentiment.NEGATIVE}
REVERSE_POLARITY_MAP = {v: k for k, v in POLARITY_MAP.items()}


class DataStatistics:

    def __init__(self, path):
        self.data = json.load(open(path))

        label_counter = Counter([v["polarity"] for k, v in self.data.items()])

        self.sentiment_counter = {k: label_counter[v] for k,v in REVERSE_POLARITY_MAP.items()}

        self.pos, self.neu, self.neg = self.sentiment_counter[Sentiment.POSITIVE], self.sentiment_counter[Sentiment.NEUTRAL], self.sentiment_counter[Sentiment.NEGATIVE]

        self.total = self.pos + self.neu + self.neg

        self.majority_class = max(self.sentiment_counter.items(), key=operator.itemgetter(1))[0]
    
    def count(self, sentiment=None):
        
        return self.sentiment_counter[sentiment] if sentiment else self.total

def log_label_distribution(data_statistics, dataset, split):
    total_labels = data_statistics.count()
    count_string = "\t".join([f"{REVERSE_POLARITY_MAP[k].upper()}:\t{data_statistics.count(k)}\t/\t{total_labels}\t=\t{data_statistics.count(k)*100/total_labels:.2f}" for k in Sentiment])
    log_string = f"[{dataset.value.upper()}]\t[{split.value.upper()}]" + "\t" + count_string
    
    print(log_string)
    logger.info(log_string)

REST_TRAIN_FILE= "../data/se14-task4/processed/rest/train.json"
REST_DEV_FILE= "../data/se14-task4/processed/rest/dev.json"
REST_TEST_FILE = "../data/se14-task4/processed/rest/test.json"

LAPTOP_TRAIN_FILE = "../data/se14-task4/processed/laptop/train.json"
LAPTOP_DEV_FILE = "../data/se14-task4/processed/laptop/dev.json"
LAPTOP_TEST_FILE = "../data/se14-task4/processed/laptop/test.json"

rest_train_statistics = DataStatistics(REST_TRAIN_FILE)
rest_dev_statistics = DataStatistics(REST_DEV_FILE)
rest_test_statistics = DataStatistics(REST_TEST_FILE)

laptop_train_statistics = DataStatistics(LAPTOP_TRAIN_FILE)
laptop_dev_statistics = DataStatistics(LAPTOP_DEV_FILE)
laptop_test_statistics = DataStatistics(LAPTOP_TEST_FILE)

log_label_distribution(rest_train_statistics, Dataset.REST, Split.TRAIN)
log_label_distribution(rest_dev_statistics, Dataset.REST, Split.DEV)
log_label_distribution(rest_test_statistics, Dataset.REST, Split.TEST)

log_label_distribution(laptop_train_statistics, Dataset.LAPTOP, Split.TRAIN)
log_label_distribution(laptop_dev_statistics, Dataset.LAPTOP, Split.DEV)
log_label_distribution(laptop_test_statistics, Dataset.LAPTOP, Split.TEST)

#rest_train_counter = Counter([v["polarity"] for k, v in json.load(open(REST_TRAIN_FILE)).items()])
#laptop_train_counter = Counter([v["polarity"] for k, v in json.load(open(LAPTOP_TRAIN_FILE)).items()])
#
#rest_majority_class = max(rest_train_counter.items(), key= lambda d: d[-1])[0]
#laptop_majority_class = max(laptop_train_counter.items(), key= lambda d: d[-1])[0]
#
#
## Compute majority voting accuracy for restaurant dataset
#
#GT_LABELS = [POLARITY_MAP[v["polarity"]] for k, v in json.load(open(REST_TEST_FILE)).items()]
#MAJORITY_PRED_LABELS = [POLARITY_MAP[rest_majority_class]] * len(GT_LABELS)
#RANDOM_PRED_LABELS = np.random.choice(list(POLARITY_MAP.values()), size=len(GT_LABELS))
#
#majority_acc = accuracy_score(GT_LABELS, MAJORITY_PRED_LABELS) 
#random_acc = accuracy_score(GT_LABELS, RANDOM_PRED_LABELS)
#
#rest_label_count = Counter(GT_LABELS) 
#rest_neu, rest_pos, rest_neg = rest_label_count[1],  rest_label_count[2], rest_label_count[0]
#rest_total = rest_neu + rest_pos + rest_neg
#print(f"REST::\tPOSITIVE:{rest_pos}/{rest_total}={rest_pos/rest_total}\tNEGATIVE:{rest_neg}/{rest_total}={rest_neg/rest_total}\tNEUTRAL:{rest_neu}/{rest_total}={rest_neg/rest_total}")
#print(f"REST::\tMAJORITY ACC:\t{majority_acc:.4f}")
#print(f"REST::\tRANDOM:\t{random_acc:.4f}")
#
## Compute majority voting accuracy for laptop dataset
#GT_LABELS = [POLARITY_MAP[v["polarity"]] for k, v in json.load(open(LAPTOP_TEST_FILE)).items()]
#MAJORITY_PRED_LABELS = [POLARITY_MAP[laptop_majority_class] for _ in range(len(GT_LABELS))]
#
#acc = accuracy_score(GT_LABELS, MAJORITY_PRED_LABELS) 
#print(f"LAPT::\tACC:\t{acc:.4f}")
