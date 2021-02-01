import argparse
import sys
import numpy as np
import pandas as pd
import json
import pickle

def parse_config(src_config):
    try:
        with open(src_config, 'r') as data_json:
            j_data = json.load(data_json)
            dense_dim = j_data["dense"]
            categorical_dim = j_data["categorical"]
            slot_size = j_data["slot_size"]
        assert(categorical_dim == np.sum(slot_size))
        return dense_dim, categorical_dim, slot_size
    except:
        print("Invalid data configuration file!")
    
def convert(src_csv, src_config, dst, batch_size,segmentation):
    dense_dim, categorical_dim, slot_size = parse_config(src_config)
    total_columns = 1 + dense_dim + categorical_dim
    cols = []
    for i in range(total_columns):
        if i == 0:
            cols.append("label")
        else:
            cols.append("I"+str(i)) if i <= dense_dim else cols.append("C"+str(i-dense_dim))
    df = pd.read_csv(src_csv, names=cols, sep=' ', nrows=batch_size)
    slot_num = len(slot_size)
    row_ptrs = [0 for _ in range(batch_size*slot_num + 1)]
    for i in range(1, len(row_ptrs)):
        row_ptrs[i] = row_ptrs[i-1] + slot_size[(i-1)%slot_num]
    label_df =  pd.DataFrame(df['label'].values.reshape(1,batch_size))
    dense_df = pd.DataFrame(df[['I'+str(i+1) for i in range(dense_dim)]].values.reshape(1, batch_size*dense_dim))
    embedding_columns_df = pd.DataFrame(df[['C'+str(i+1) for i in range(categorical_dim)]].values.reshape(1, batch_size*categorical_dim))
    row_ptrs_df = pd.DataFrame(np.array(row_ptrs).reshape(1, batch_size*slot_num + 1))
    with open(dst, 'w') as dst_txt:
        label_df.to_csv(dst_txt, mode='a+', sep = segmentation, index=False,  header=False)
        dense_df.to_csv(dst_txt, mode='a+', sep = segmentation, index=False,  header=False)
        embedding_columns_df.to_csv(dst_txt, mode='a+', sep = segmentation, index=False,  header=False)
        row_ptrs_df.to_csv(dst_txt, mode='a+', sep = segmentation, index=False,  header=False)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Convert Preprocessed Criteo Data to Inference Format')
    arg_parser.add_argument('--src_csv_path', type=str, required=True)
    arg_parser.add_argument('--src_config_path', type=str, required=True)
    arg_parser.add_argument('--dst_path', type=str, required=True)
    arg_parser.add_argument('--batch_size', type=int, default=128)
    arg_parser.add_argument('--segmentation', type=str, default=' ')
    args = arg_parser.parse_args()
    src_csv_path = args.src_csv_path
    segmentation = args.segmentation
    src_config_path = args.src_config_path
    dst_path = args.dst_path
    batch_size = args.batch_size
    convert(src_csv_path, src_config_path, dst_path, batch_size, segmentation)