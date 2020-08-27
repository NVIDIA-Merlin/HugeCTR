import nvtabular as nvt
import os
import sys
import argparse
import glob
import cudf
from cudf.io.parquet import ParquetWriter
import numpy as np
import pandas as pd
import concurrent.futures as cf
from concurrent.futures import as_completed
import shutil
import cudf

#%load_ext memory_profiler

import logging
logging.basicConfig(format='%(asctime)s %(message)s')
logging.root.setLevel(logging.NOTSET)


DATA_DIR = os.environ.get('DATA_DIR', './hugectr')
PREPROCESS_DIR = os.path.join(DATA_DIR, 'criteo')
PREPROCESS_DIR_temp = os.path.join(PREPROCESS_DIR, 'temp_csv')

CATEGORICAL_COLUMNS=["C" + str(x) for x in range(1, 27)]
CONTINUOUS_COLUMNS=["I" + str(x) for x in range(1, 14)]
LABEL_COLUMNS = ['label']
COLUMNS =  LABEL_COLUMNS + CONTINUOUS_COLUMNS +  CATEGORICAL_COLUMNS
criteo_COLUMN=LABEL_COLUMNS +  CATEGORICAL_COLUMNS

NUM_INTEGER_COLUMNS = 13
NUM_CATEGORICAL_COLUMNS = 26
NUM_TOTAL_COLUMNS = 1 + NUM_INTEGER_COLUMNS + NUM_CATEGORICAL_COLUMNS


def _wait_futures_and_reset(futures):
    for future in as_completed(futures):
        print(future.result())
    futures = list()

def convert_label(path):
    for filename in os.listdir(path):
        if filename.endswith(".parquet"):
            print(os.path.join(path, filename))
            df = cudf.read_parquet(os.path.join(path, filename))
            #df = df.astype({"label": np.float32})
            df["label"] = df['label'].astype('float32')
            df.to_parquet(os.path.join(path, filename))
    
    
def process_NVT(train_paths,output_path):
    train_paths = glob.glob(os.path.join(train_paths, "*.csv"))
    output_file_num=len(train_paths)*4
    COLUMNS =  LABEL_COLUMNS + CONTINUOUS_COLUMNS +  CATEGORICAL_COLUMNS 
    if criteo_mode==0:
        proc = nvt.Workflow(cat_names=CATEGORICAL_COLUMNS,cont_names=CONTINUOUS_COLUMNS,label_name=LABEL_COLUMNS)
        logging.info('Fillmissing processing')
        proc.add_cont_feature(nvt.ops.FillMissing())
        logging.info('Nomalization processing')
        proc.add_cont_preprocess(nvt.ops.Normalize())
    else:
        proc = nvt.Workflow(cat_names=CATEGORICAL_COLUMNS,cont_names=[],label_name=LABEL_COLUMNS)
    logging.info('Categorify processing')
    proc.add_cat_preprocess(nvt.ops.Categorify(freq_threshold=6))
    proc.finalize() # prepare to load the config
        
    GPU_MEMORY_FRAC = 0.6
    output_format='hugectr'
    shuffle = False
    if parquet_format:
        output_format='parquet'
        shuffle = True
    
    # just for criteo model
    if criteo_mode==1:
        train_ds_iterator = nvt.io.Dataset(train_paths, engine='csv', gpu_memory_frac=GPU_MEMORY_FRAC,columns=criteo_COLUMN)
    else:
        train_ds_iterator = nvt.io.Dataset(train_paths, engine='csv', gpu_memory_frac=GPU_MEMORY_FRAC,columns=COLUMNS)
    proc.apply(
        train_ds_iterator,
        output_path=output_path,
        out_files_per_proc=output_file_num,
        output_format=output_format,
        shuffle=shuffle,
        num_io_threads=2,
    )
    embeddings=nvt.ops.get_embedding_sizes(proc)
    slot_size=[]
    #Output slot_size for each categorical feature
    for item in CATEGORICAL_COLUMNS:
        slot_size.append(embeddings[item][0])
    print(slot_size)
    #conver label type to fl32
    if parquet_format:
        convert_label(output_path)
    
    
def _process_chunks(chunk,path,feature_cross,feature_pairs):
    if feature_cross:
        logging.info('feature cross for chunk file:{} '.format(path))
        for idx, pair in enumerate(feature_pairs):
            col0 = pair[0]
            col1 = pair[1]
            chunk.insert(NUM_INTEGER_COLUMNS + 1 + idx, col0+'_'+col1, chunk[col0]+chunk[col1])
    chunk.to_csv(path, index=None)
    logging.info('output temp chunk file done:{}'.format(path))

def preprocess(src_csv_path, dst_csv_path, normalize_dense=1, feature_cross=1):
    PREPROCESS_DIR_temp = os.path.join(dst_csv_path, 'temp_csv')
    if os.path.exists(PREPROCESS_DIR_temp):
        shutil.rmtree(PREPROCESS_DIR_temp)
    os.mkdir(PREPROCESS_DIR_temp)
    
    feature_pairs = [('C1', 'C2'), ('C3', 'C4')]
    if feature_cross:
        for idx, pair in enumerate(feature_pairs):
            col0 = pair[0]
            col1 = pair[1]
            CATEGORICAL_COLUMNS.insert(idx,col0+'_'+col1)
    
    
    logging.info('Create a temp CSV folder:{}'.format(PREPROCESS_DIR_temp))
    with cf.ThreadPoolExecutor(max_workers=100) as executor:
        logging.info('read a CSV file')
        reader = pd.read_csv(src_csv_path, sep='\t',names=COLUMNS,chunksize=10000000)

        logging.info('split data to csv format')
        futures = list()
        num=0
        for chunk in reader:
            logging.info('read chunk:{}'.format(num))
            path=os.path.join(PREPROCESS_DIR_temp, '{}_chunk.csv'.format(num))
            future = executor.submit(_process_chunks,chunk,path,feature_cross,feature_pairs)
            futures.append(future)
            logging.info('read chunk:{} done'.format(num))
            num+=1
        _wait_futures_and_reset(futures)
    
    
    logging.info('NVTabular processing')
    process_NVT(PREPROCESS_DIR_temp,dst_csv_path)
    logging.info('NVTabular processing done')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Preprocssing Criteo Dataset')

    arg_parser.add_argument('--src_csv_path', type=str, required=True)
    arg_parser.add_argument('--dst_csv_path', type=str, required=True)
    arg_parser.add_argument('--normalize_dense', type=int, default=1)
    arg_parser.add_argument('--feature_cross', type=int, default=1)
    arg_parser.add_argument('--criteo_mode', type=int, default=0)
    arg_parser.add_argument('--parquet_format', type=int, default=1)
    

    args = arg_parser.parse_args()

    src_csv_path = args.src_csv_path
    dst_csv_path = args.dst_csv_path

    normalize_dense = args.normalize_dense
    feature_cross = args.feature_cross
    criteo_mode = args.criteo_mode
    parquet_format = args.parquet_format

    if os.path.exists(src_csv_path) == False:
        sys.exit('ERROR: the file \'{}\' doesn\'t exist'.format(src_csv_path))

    if os.path.exists(dst_csv_path) == False:
        sys.exit('ERROR: the folder \'{}\' does\'t exists'.format(dst_csv_path))

    preprocess(src_csv_path, dst_csv_path, normalize_dense, feature_cross)