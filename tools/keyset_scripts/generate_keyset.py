import sys
import os
import argparse
import logging
import numpy as np
import cudf
import glob

logging.basicConfig(format='%(asctime)s %(message)s')
logging.root.setLevel(logging.NOTSET)

CRITEO_CAT_POS = [c for c in range(14, 40)]

def generate_keyset_for_single_file(file, cat_features_pos, cum_slot_size_array):
    df = cudf.read_parquet(file)
    keysets = []
    for i in range(len(cat_features_pos)):
        unique_keys = df.iloc[:, cat_features_pos[i]].unique() + cum_slot_size_array[i]
        keysets.append(set(unique_keys.to_pandas()))
        del unique_keys
    return keysets

def generate_keyset(src_dir_path, dst_dir_path, cat_features_pos, slot_size_array, int32_keyset=False):
    filelist = []
    all_keysets = []
    for _ in range(len(cat_features_pos)):
        all_keysets.append(set())
    cum_slot_size_array = np.cumsum(np.array([0] + slot_size_array[:-1]))
    for file in glob.glob(src_dir_path + '/*.parquet'):
        filelist.append(file)
    for file in filelist:
        cur_keysets = generate_keyset_for_single_file(file, cat_features_pos, cum_slot_size_array)
        for i in range(len(all_keysets)):
            all_keysets[i] = all_keysets[i].union(cur_keysets[i])
    with open(dst_dir_path, "wb") as f:
        int_size = 4 if int32_keyset else 8
        for all_keys in all_keysets:
            for k in all_keys:
                f.write(int(k).to_bytes(int_size, "little", signed=True))
    logging.info('Extracted keyset from {}'.format(src_dir_path))
        
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Creating keyset from parquet data')

    arg_parser.add_argument('--src_dir_path', type=str, required=True)
    arg_parser.add_argument('--keyset_path', type=str, required=True)
    arg_parser.add_argument('--cat_features_pos', nargs="*", type=int, required=False)
    arg_parser.add_argument('--slot_size_array', nargs="*", type=int, required=False)
    arg_parser.add_argument('--int32_keyset', type=bool, required=False, default=False)

    args = arg_parser.parse_args()

    src_dir_path = args.src_dir_path
    keyset_path = args.keyset_path
    cat_features_pos = args.cat_features_pos if args.cat_features_pos else CRITEO_CAT_POS
    slot_size_array = args.slot_size_array if args.slot_size_array else [0] * len(cat_features_pos)
    int32_keyset = args.int32_keyset if args.int32_keyset else False

    if os.path.exists(src_dir_path) == False:
        sys.exit('ERROR: the directory \'{}\' doesn\'t exist'.format(src_dir_path))
    
    if len(cat_features_pos) != len(slot_size_array):
        sys.exit('ERROR: the cat_features_pos and slot_size_array do not have the same dimension')

    generate_keyset(src_dir_path, keyset_path, cat_features_pos, slot_size_array, int32_keyset)