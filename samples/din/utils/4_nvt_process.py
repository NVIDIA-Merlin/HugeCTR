import os
import sys
import argparse
import glob
import time
import numpy as np
import pandas as pd
import concurrent.futures as cf
from concurrent.futures import as_completed
import shutil

import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask.utils import parse_bytes
from dask.delayed import delayed

import cudf
import rmm
import nvtabular as nvt
from nvtabular.io import Shuffle
from nvtabular.utils import device_mem_size
from nvtabular.ops import Categorify, Rename, Operator, get_embedding_sizes
from nvtabular.column_group import ColumnGroup

MAX_SIZE = 10

CAT_COLUMNS = (
    ["UID"]
    + ["MID" + str(x) for x in range(MAX_SIZE + 1)]
    + ["CID" + str(x) for x in range(MAX_SIZE + 1)]
)
LABEL = ["LABEL"]
COLUMNS = CAT_COLUMNS + ["LABEL"]

# Initialize RMM pool on ALL workers
def setup_rmm_pool(client, pool_size):
    client.run(rmm.reinitialize, pool_allocator=True, initial_pool_size=pool_size)
    return None


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

    # Deploy a Single-Machine Multi-GPU Cluster
    device_size = device_mem_size(kind="total")
    cluster = None
    if args.protocol == "ucx":
        UCX_TLS = os.environ.get("UCX_TLS", "tcp,cuda_copy,cuda_ipc,sockcm")
        os.environ["UCX_TLS"] = UCX_TLS
        cluster = LocalCUDACluster(
            protocol=args.protocol,
            CUDA_VISIBLE_DEVICES=args.devices,
            n_workers=len(args.devices.split(",")),
            enable_nvlink=True,
            device_memory_limit=int(device_size * args.device_limit_frac),
            dashboard_address=":" + args.dashboard_port,
        )
    else:
        cluster = LocalCUDACluster(
            protocol=args.protocol,
            n_workers=len(args.devices.split(",")),
            CUDA_VISIBLE_DEVICES=args.devices,
            device_memory_limit=int(device_size * args.device_limit_frac),
            dashboard_address=":" + args.dashboard_port,
        )

    # Create the distributed client
    client = Client(cluster)
    if args.device_pool_frac > 0.01:
        setup_rmm_pool(client, int(args.device_pool_frac * device_size))

    runtime = time.time()

    ##Real works here
    features = LABEL + ColumnGroup(CAT_COLUMNS)

    workflow = nvt.Workflow(features, client=client)

    train_ds_iterator = nvt.Dataset(
        train_temp, engine="parquet", part_size=int(args.part_mem_frac * device_size)
    )
    valid_ds_iterator = nvt.Dataset(
        valid_temp, engine="parquet", part_size=int(args.part_mem_frac * device_size)
    )

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
        out_files_per_proc=args.out_files_per_proc,
        num_threads=args.num_io_threads,
    )

    workflow.transform(valid_ds_iterator).to_parquet(
        output_path=valid_output,
        dtypes=dict_dtypes,
        cats=CAT_COLUMNS,
        labels=LABEL,
        shuffle=shuffle,
        out_files_per_proc=args.out_files_per_proc,
        num_threads=args.num_io_threads,
    )

    client.close()

    print("Time:", time.time() - runtime)


def parse_args():
    parser = argparse.ArgumentParser(description=("Multi-GPU Preprocessing"))

    #
    # System Options
    #

    parser.add_argument(
        "-d",
        "--devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        type=str,
        help='Comma-separated list of visible devices (e.g. "0,1,2,3"). ',
    )
    parser.add_argument(
        "-p",
        "--protocol",
        choices=["tcp", "ucx"],
        default="tcp",
        type=str,
        help="Communication protocol to use (Default 'tcp')",
    )
    parser.add_argument(
        "--device_limit_frac",
        default=0.5,
        type=float,
        help="Worker device-memory limit as a fraction of GPU capacity (Default 0.8). ",
    )
    parser.add_argument(
        "--device_pool_frac",
        default=0.5,
        type=float,
        help="RMM pool size for each worker  as a fraction of GPU capacity (Default 0.5). "
        "The RMM pool frac is the same for all GPUs, make sure each one has enough memory size",
    )
    parser.add_argument(
        "--num_io_threads",
        default=0,
        type=int,
        help="Number of threads to use when writing output data (Default 0). "
        "If 0 is specified, multi-threading will not be used for IO.",
    )

    parser.add_argument(
        "--part_mem_frac",
        default=0.125,
        type=float,
        help="Maximum size desired for dataset partitions as a fraction "
        "of GPU capacity (Default 0.125)",
    )
    parser.add_argument(
        "--out_files_per_proc",
        default=1,
        type=int,
        help="Number of output files to write on each worker (Default 1)",
    )

    #
    # Preprocessing Options
    #

    parser.add_argument(
        "-f",
        "--freq_limit",
        default=0,
        type=int,
        help="Frequency limit for categorical encoding (Default 0)",
    )
    parser.add_argument(
        "-s",
        "--shuffle",
        choices=["PER_WORKER", "PER_PARTITION", "NONE"],
        default="PER_PARTITION",
        help="Shuffle algorithm to use when writing output data to disk (Default PER_PARTITION)",
    )

    #
    # Diagnostics Options
    #

    parser.add_argument(
        "--dashboard_port",
        default="8787",
        type=str,
        help="Specify the desired port of Dask's diagnostics-dashboard (Default `8787`). "
        "The dashboard will be hosted at http://<IP>:<PORT>/status",
    )

    args = parser.parse_args()
    args.n_workers = len(args.devices.split(","))
    return args


if __name__ == "__main__":

    args = parse_args()

    process(args)
