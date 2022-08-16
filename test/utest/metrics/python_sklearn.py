from __future__ import print_function
import numpy as np
from sklearn.metrics import roc_auc_score, ndcg_score
import argparse
import sys

parser = argparse.ArgumentParser(description="Compute AUC or NDCG with sklearn")
parser.add_argument("count", help="Number of elements", type=int)
parser.add_argument("fp_bytes", help="Bytes per float in the file", type=int)
parser.add_argument("input_file", help="Input filename")
parser.add_argument("num_classes", help="Number of classes", type=int)
parser.add_argument("metric", help="Metric to compute", type=str)
args = parser.parse_args()

dtype = [
    ("labels", np.float32, (args.count)),
    ("scores", np.float32 if args.fp_bytes == 4 else np.float16, (args.count)),
]
data = np.fromfile(args.input_file, dtype=dtype)

if args.metric == "AUC":
    if args.num_classes == 1:
        print(
            roc_auc_score(
                data["labels"].flatten().astype(int), data["scores"].flatten().astype(float)
            )
        )
    else:
        print(
            roc_auc_score(
                data["labels"].flatten().astype(int).reshape((-1, args.num_classes)),
                data["scores"].flatten().astype(float).reshape((-1, args.num_classes)),
            )
        )
elif args.metric == "NDCG":
    labels = data["labels"].flatten().reshape(1, -1).astype(float)
    scores = data["scores"].flatten().reshape(1, -1).astype(float)
    print(ndcg_score(labels, scores))
else:
    print("Invalid metric")
