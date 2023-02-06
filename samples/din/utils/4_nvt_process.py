import os
import argparse
import time
import numpy as np
import shutil

import nvtabular as nvt
from nvtabular.ops import FillMissing

MAX_SIZE = 10

CAT_COLUMNS = (
    ["UID"]
    + ["MID" + str(x) for x in range(MAX_SIZE + 1)]
    + ["CID" + str(x) for x in range(MAX_SIZE + 1)]
)
LABEL = ["LABEL"]
COLUMNS = CAT_COLUMNS + ["LABEL"]


# Processing using NVT
def process(args):
    train_path = os.path.abspath("../din_data/train")
    test_path = os.path.abspath("../din_data/valid")

    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    os.mkdir(train_path)
    os.mkdir(test_path)

    # Path to save temp parquet
    train_temp = "../din_data/train_temp.parquet"
    valid_temp = "../din_data/test_temp.parquet"

    # Path to save final parquet
    train_output = train_path
    valid_output = test_path

    runtime = time.time()

    ##Real works here
    features = LABEL + CAT_COLUMNS >> FillMissing()

    workflow = nvt.Workflow(features)

    train_ds_iterator = nvt.Dataset(train_temp, engine="parquet")
    valid_ds_iterator = nvt.Dataset(valid_temp, engine="parquet")

    ##Shuffle
    shuffle = None
    if args.shuffle == "PER_WORKER":
        shuffle = nvt.io.Shuffle.PER_WORKER
    elif args.shuffle == "PER_PARTITION":
        shuffle = nvt.io.Shuffle.PER_PARTITION

    dict_dtypes = {}
    for col in CAT_COLUMNS:
        dict_dtypes[col] = np.int64
    for col in LABEL:
        dict_dtypes[col] = np.float32

    workflow.fit(train_ds_iterator)

    workflow.transform(train_ds_iterator).to_parquet(
        output_path=train_output,
        dtypes=dict_dtypes,
        cats=CAT_COLUMNS,
        labels=LABEL,
        shuffle=shuffle,
        out_files_per_proc=1,
        num_threads=0,
    )

    workflow.transform(valid_ds_iterator).to_parquet(
        output_path=valid_output,
        dtypes=dict_dtypes,
        cats=CAT_COLUMNS,
        labels=LABEL,
        shuffle=shuffle,
        out_files_per_proc=1,
        num_threads=0,
    )
    print("Time:", time.time() - runtime)


def parse_args():
    parser = argparse.ArgumentParser(description=("Multi-GPU Preprocessing"))
    parser.add_argument(
        "-s",
        "--shuffle",
        choices=["PER_WORKER", "PER_PARTITION", "NONE"],
        default="PER_PARTITION",
        help="Shuffle algorithm to use when writing output data to disk (Default PER_PARTITION)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    process(args)
