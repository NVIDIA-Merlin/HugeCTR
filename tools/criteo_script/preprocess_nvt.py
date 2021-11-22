import os
import sys
import argparse
import glob
import time
from cudf.io.parquet import ParquetWriter
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
from nvtabular.ops import Categorify, Clip, FillMissing, HashBucket, LambdaOp, Normalize, Rename, Operator, get_embedding_sizes
#%load_ext memory_profiler

import logging
logging.basicConfig(format='%(asctime)s %(message)s')
logging.root.setLevel(logging.NOTSET)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# define dataset schema
CATEGORICAL_COLUMNS=["C" + str(x) for x in range(1, 27)]
CONTINUOUS_COLUMNS=["I" + str(x) for x in range(1, 14)]
LABEL_COLUMNS = ['label']
COLUMNS =  LABEL_COLUMNS + CONTINUOUS_COLUMNS +  CATEGORICAL_COLUMNS
#/samples/criteo mode doesn't have dense features
criteo_COLUMN=LABEL_COLUMNS +  CATEGORICAL_COLUMNS
#For new feature cross columns
CROSS_COLUMNS = []


NUM_INTEGER_COLUMNS = 13
NUM_CATEGORICAL_COLUMNS = 26
NUM_TOTAL_COLUMNS = 1 + NUM_INTEGER_COLUMNS + NUM_CATEGORICAL_COLUMNS


# Initialize RMM pool on ALL workers
def setup_rmm_pool(client, pool_size):
    client.run(rmm.reinitialize, pool_allocator=True, initial_pool_size=pool_size)
    return None

#compute the partition size with GB
def bytesto(bytes, to, bsize=1024):
    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    r = float(bytes)
    return bytes / (bsize ** a[to])

class FeatureCross(Operator):
    def __init__(self, dependency):
        self.dependency = dependency

    def transform(self, columns, gdf):
        new_df = type(gdf)()
        for col in columns.names:
            new_df[col] = gdf[col] + gdf[self.dependency]
        return new_df

    def dependencies(self):
        return [self.dependency]

#process the data with NVTabular
def process_NVT(args):

    if args.feature_cross_list:
        feature_pairs = [pair.split("_") for pair in args.feature_cross_list.split(",")]
        for pair in feature_pairs:
            CROSS_COLUMNS.append(pair[0]+'_'+pair[1])


    logging.info('NVTabular processing')
    train_input = os.path.join(args.data_path, "train/train.txt")
    val_input = os.path.join(args.data_path, "val/test.txt")
    PREPROCESS_DIR_temp_train = os.path.join(args.out_path, 'train/temp-parquet-after-conversion')
    PREPROCESS_DIR_temp_val = os.path.join(args.out_path, 'val/temp-parquet-after-conversion')
    PREPROCESS_DIR_temp = [PREPROCESS_DIR_temp_train, PREPROCESS_DIR_temp_val]
    train_output = os.path.join(args.out_path, "train")
    val_output = os.path.join(args.out_path, "val")

    # Make sure we have a clean parquet space for cudf conversion
    for one_path in PREPROCESS_DIR_temp:
        if os.path.exists(one_path):
            shutil.rmtree(one_path)
        os.mkdir(one_path)


    ## Get Dask Client

    # Deploy a Single-Machine Multi-GPU Cluster
    device_size = device_mem_size(kind="total")
    cluster = None
    if args.protocol == "ucx":
        UCX_TLS = os.environ.get("UCX_TLS", "tcp,cuda_copy,cuda_ipc,sockcm")
        os.environ["UCX_TLS"] = UCX_TLS
        cluster = LocalCUDACluster(
            protocol = args.protocol,
            CUDA_VISIBLE_DEVICES = args.devices,
            n_workers = len(args.devices.split(",")),
            enable_nvlink=True,
            device_memory_limit = int(device_size * args.device_limit_frac),
            dashboard_address=":" + args.dashboard_port
        )
    else:
        cluster = LocalCUDACluster(
            protocol = args.protocol,
            n_workers = len(args.devices.split(",")),
            CUDA_VISIBLE_DEVICES = args.devices,
            device_memory_limit = int(device_size * args.device_limit_frac),
            dashboard_address=":" + args.dashboard_port
        )



    # Create the distributed client
    client = Client(cluster)
    if args.device_pool_frac > 0.01:
        setup_rmm_pool(client, int(args.device_pool_frac*device_size))


    #calculate the total processing time
    runtime = time.time()

    #test dataset without the label feature
    if args.dataset_type == 'test':
        global LABEL_COLUMNS
        LABEL_COLUMNS = []

    ##-----------------------------------##
    # Dask rapids converts txt to parquet
    # Dask cudf dataframe = ddf

    ## train/valid txt to parquet
    train_valid_paths = [(train_input,PREPROCESS_DIR_temp_train),(val_input,PREPROCESS_DIR_temp_val)]

    for input, temp_output in train_valid_paths:

        ddf = dask_cudf.read_csv(input,sep='\t',names=LABEL_COLUMNS + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS)

        ## Convert label col to FP32
        if args.parquet_format and args.dataset_type == 'train':
            ddf["label"] = ddf['label'].astype('float32')

        # Save it as parquet format for better memory usage
        ddf.to_parquet(temp_output,header=True)
        ##-----------------------------------##

    COLUMNS =  LABEL_COLUMNS + CONTINUOUS_COLUMNS + CROSS_COLUMNS + CATEGORICAL_COLUMNS
    train_paths = glob.glob(os.path.join(PREPROCESS_DIR_temp_train, "*.parquet"))
    valid_paths = glob.glob(os.path.join(PREPROCESS_DIR_temp_val, "*.parquet"))

    categorify_op = Categorify(freq_threshold=args.freq_limit)
    cat_features = CATEGORICAL_COLUMNS >> categorify_op
    cont_features = CONTINUOUS_COLUMNS >> FillMissing() >> Clip(min_value=0) >> Normalize()
    cross_cat_op = Categorify(freq_threshold=args.freq_limit)

    features = LABEL_COLUMNS
    
    if args.criteo_mode == 0:
        features += cont_features
        if args.feature_cross_list:
            feature_pairs = [pair.split("_") for pair in args.feature_cross_list.split(",")]
            for pair in feature_pairs:
                col0 = pair[0]
                col1 = pair[1]
                features += col0 >> FeatureCross(col1)  >> Rename(postfix="_"+col1) >> cross_cat_op
            
    features += cat_features

    workflow = nvt.Workflow(features, client=client)

    logging.info("Preprocessing")

    output_format = 'hugectr'
    if args.parquet_format:
        output_format = 'parquet'

    # just for /samples/criteo model
    train_ds_iterator = nvt.Dataset(train_paths, engine='parquet', part_size=int(args.part_mem_frac * device_size))
    valid_ds_iterator = nvt.Dataset(valid_paths, engine='parquet', part_size=int(args.part_mem_frac * device_size))

    shuffle = None
    if args.shuffle == "PER_WORKER":
        shuffle = nvt.io.Shuffle.PER_WORKER
    elif args.shuffle == "PER_PARTITION":
        shuffle = nvt.io.Shuffle.PER_PARTITION

    logging.info('Train Datasets Preprocessing.....')

    dict_dtypes = {}
    for col in CATEGORICAL_COLUMNS:
        dict_dtypes[col] = np.int64
    if not args.criteo_mode:
        for col in CONTINUOUS_COLUMNS:
            dict_dtypes[col] = np.float32
    for col in CROSS_COLUMNS:
        dict_dtypes[col] = np.int64
    for col in LABEL_COLUMNS:
        dict_dtypes[col] = np.float32
    
    conts = CONTINUOUS_COLUMNS if not args.criteo_mode else []
    
    workflow.fit(train_ds_iterator)
    
    if output_format == 'hugectr':
        workflow.transform(train_ds_iterator).to_hugectr(
                cats=CATEGORICAL_COLUMNS + CROSS_COLUMNS,
                conts=conts,
                labels=LABEL_COLUMNS,
                output_path=train_output,
                shuffle=shuffle,
                out_files_per_proc=args.out_files_per_proc,
                num_threads=args.num_io_threads)
    else:
        workflow.transform(train_ds_iterator).to_parquet(
                output_path=train_output,
                dtypes=dict_dtypes,
                cats=CATEGORICAL_COLUMNS + CROSS_COLUMNS,
                conts=conts,
                labels=LABEL_COLUMNS,
                shuffle=shuffle,
                out_files_per_proc=args.out_files_per_proc,
                num_threads=args.num_io_threads)
        
        
        
    ###Getting slot size###    
    #--------------------##
    embeddings_dict_cat = categorify_op.get_embedding_sizes(CATEGORICAL_COLUMNS)
    embeddings_dict_cross = cross_cat_op.get_embedding_sizes(CROSS_COLUMNS)
    embeddings = [embeddings_dict_cross[c][0] for c in CROSS_COLUMNS] + [embeddings_dict_cat[c][0] for c in CATEGORICAL_COLUMNS]
    
    print(embeddings)
    ##--------------------##

    logging.info('Valid Datasets Preprocessing.....')

    if output_format == 'hugectr':
        workflow.transform(valid_ds_iterator).to_hugectr(
                cats=CATEGORICAL_COLUMNS + CROSS_COLUMNS,
                conts=conts,
                labels=LABEL_COLUMNS,
                output_path=val_output,
                shuffle=shuffle,
                out_files_per_proc=args.out_files_per_proc,
                num_threads=args.num_io_threads)
    else:
        workflow.transform(valid_ds_iterator).to_parquet(
                output_path=val_output,
                dtypes=dict_dtypes,
                cats=CATEGORICAL_COLUMNS + CROSS_COLUMNS,
                conts=conts,
                labels=LABEL_COLUMNS,
                shuffle=shuffle,
                out_files_per_proc=args.out_files_per_proc,
                num_threads=args.num_io_threads)

    embeddings_dict_cat = categorify_op.get_embedding_sizes(CATEGORICAL_COLUMNS)
    embeddings_dict_cross = cross_cat_op.get_embedding_sizes(CROSS_COLUMNS)
    embeddings = [embeddings_dict_cross[c][0] for c in CROSS_COLUMNS] + [embeddings_dict_cat[c][0] for c in CATEGORICAL_COLUMNS]
    
    print(embeddings)
    ##--------------------##

    ## Shutdown clusters
    client.close()
    logging.info('NVTabular processing done')

    runtime = time.time() - runtime

    print("\nDask-NVTabular Criteo Preprocessing")
    print("--------------------------------------")
    print(f"data_path          | {args.data_path}")
    print(f"output_path        | {args.out_path}")
    print(f"partition size     | {'%.2f GB'%bytesto(int(args.part_mem_frac * device_size),'g')}")
    print(f"protocol           | {args.protocol}")
    print(f"device(s)          | {args.devices}")
    print(f"rmm-pool-frac      | {(args.device_pool_frac)}")
    print(f"out-files-per-proc | {args.out_files_per_proc}")
    print(f"num_io_threads     | {args.num_io_threads}")
    print(f"shuffle            | {args.shuffle}")
    print("======================================")
    print(f"Runtime[s]         | {runtime}")
    print("======================================\n")


def parse_args():
    parser = argparse.ArgumentParser(description=("Multi-GPU Criteo Preprocessing"))

    #
    # System Options
    #

    parser.add_argument("--data_path", type=str, help="Input dataset path (Required)")
    parser.add_argument("--out_path", type=str, help="Directory path to write output (Required)")
    parser.add_argument(
        "-d",
        "--devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        type=str,
        help='Comma-separated list of visible devices (e.g. "0,1,2,3"). '
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
        help="Worker device-memory limit as a fraction of GPU capacity (Default 0.8). "
    )
    parser.add_argument(
        "--device_pool_frac",
        default=0.9,
        type=float,
        help="RMM pool size for each worker  as a fraction of GPU capacity (Default 0.9). "
        "The RMM pool frac is the same for all GPUs, make sure each one has enough memory size",
    )
    parser.add_argument(
        "--num_io_threads",
        default=0,
        type=int,
        help="Number of threads to use when writing output data (Default 0). "
        "If 0 is specified, multi-threading will not be used for IO.",
    )

    #
    # Data-Decomposition Parameters
    #

    parser.add_argument(
        "--part_mem_frac",
        default=0.125,
        type=float,
        help="Maximum size desired for dataset partitions as a fraction "
        "of GPU capacity (Default 0.125)",
    )
    parser.add_argument(
        "--out_files_per_proc",
        default=8,
        type=int,
        help="Number of output files to write on each worker (Default 8)",
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

    parser.add_argument(
        "--feature_cross_list", default=None, type=str, help="List of feature crossing cols (e.g. C1_C2, C3_C4)"
    )

    #
    # Diagnostics Options
    #

    parser.add_argument(
        "--profile",
        metavar="PATH",
        default=None,
        type=str,
        help="Specify a file path to export a Dask profile report (E.g. dask-report.html)."
        "If this option is excluded from the command, not profile will be exported",
    )
    parser.add_argument(
        "--dashboard_port",
        default="8787",
        type=str,
        help="Specify the desired port of Dask's diagnostics-dashboard (Default `3787`). "
        "The dashboard will be hosted at http://<IP>:<PORT>/status",
    )

    #
    # Format
    #

    parser.add_argument('--criteo_mode', type=int, default=0)
    parser.add_argument('--parquet_format', type=int, default=1)
    parser.add_argument('--dataset_type', type=str, default='train')

    args = parser.parse_args()
    args.n_workers = len(args.devices.split(","))
    return args
if __name__ == '__main__':

    args = parse_args()

    process_NVT(args)
