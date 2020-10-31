from __future__ import print_function
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse
import sys

parser = argparse.ArgumentParser(description='Compute AUC with sklearn')
parser.add_argument('count',      help='Number of elements', type=int)
parser.add_argument('fp_bytes',   help='Bytes per float in the file', type=int)
parser.add_argument('input_file', help='Input filename')
args = parser.parse_args()

dtype = [('labels', np.float32, (args.count)), ('scores', np.float32 if args.fp_bytes == 4 else np.float16, (args.count))]
data = np.fromfile(args.input_file, dtype=dtype)

# print(data['labels'].shape, file=sys.stderr)
# print(data['labels'].flatten().tolist(), file=sys.stderr)
# print(data['scores'].flatten().tolist(), file=sys.stderr)

print( roc_auc_score(data['labels'].flatten().astype(int), data['scores'].flatten().astype(float)) )
