from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import sys
import tempfile

from six.moves import urllib
import urllib.request 

import sys
import os
import math
import time
import logging
import concurrent.futures as cf
from traceback import print_exc

import numpy as np
import pandas as pd
import sklearn.preprocessing as skp

logging.basicConfig(format='%(asctime)s %(message)s')
logging.root.setLevel(logging.NOTSET)

NUM_INTEGER_COLUMNS = 13
NUM_CATEGORICAL_COLUMNS = 26
NUM_TOTAL_COLUMNS = 1 + NUM_INTEGER_COLUMNS + NUM_CATEGORICAL_COLUMNS

MAX_NUM_WORKERS = NUM_TOTAL_COLUMNS

INT_NAN_VALUE = np.iinfo(np.int32).min
CAT_NAN_VALUE = '80000000'

def idx2key(idx):
    if idx == 0:
        return 'label'
    return 'I' + str(idx) if idx <= NUM_INTEGER_COLUMNS else 'C' + str(idx - NUM_INTEGER_COLUMNS)

def _fill_missing_features_and_split(chunk, series_list_dict):
    for cid, col in enumerate(chunk.columns):
        NAN_VALUE = INT_NAN_VALUE if cid <= NUM_INTEGER_COLUMNS else CAT_NAN_VALUE
        result_series = chunk[col].fillna(NAN_VALUE)
        series_list_dict[col].append(result_series)

def _merge_and_transform_series(src_series_list, col, dense_cols,
                                normalize_dense):
    result_series = pd.concat(src_series_list)

    if col != 'label':
        unique_value_counts = result_series.value_counts()
        unique_value_counts = unique_value_counts.loc[unique_value_counts >= 6]
        unique_value_counts = set(unique_value_counts.index.values)
        NAN_VALUE = INT_NAN_VALUE if col.startswith('I') else CAT_NAN_VALUE
        result_series = result_series.apply(
                lambda x: x if x in unique_value_counts else NAN_VALUE)

    if col == 'label' or col in dense_cols:
        result_series = result_series.astype(np.int64)
        le = skp.LabelEncoder()
        result_series = pd.DataFrame(le.fit_transform(result_series))
        if col != 'label':
           result_series = result_series + 1
    else:
        oe = skp.OrdinalEncoder(dtype=np.int64)
        result_series = pd.DataFrame(oe.fit_transform(pd.DataFrame(result_series)))
        result_series = result_series + 1


    if normalize_dense != 0:
        if col in dense_cols:
           mms = skp.MinMaxScaler(feature_range=(0,1))
           result_series = pd.DataFrame(mms.fit_transform(result_series))

    result_series.columns = [col]

    min_max = (np.int64(result_series[col].min()), np.int64(result_series[col].max()))
    if col != 'label':
        logging.info('column {} [{}, {}]'.format(col, str(min_max[0]),str(min_max[1])))

    return [result_series, min_max]

def _merge_columns_and_feature_cross(series_list, min_max, feature_pairs,
                                     feature_cross):
    name_to_series = dict()
    for series in series_list:
        name_to_series[series.columns[0]] = series.iloc[:,0]
    df = pd.DataFrame(name_to_series)
    cols = [idx2key(idx) for idx in range(0, NUM_TOTAL_COLUMNS)]
    df = df.reindex(columns=cols)

    offset = np.int64(0)
    for col in cols:
        if col != 'label' and col.startswith('I') == False:
            df[col] += offset
            logging.info('column {} offset {}'.format(col, str(offset)))
            offset += min_max[col][1]

    if feature_cross != 0:
        for idx, pair in enumerate(feature_pairs):
            col0 = pair[0]
            col1 = pair[1]

            col1_width = int(min_max[col1][1] - min_max[col1][0] + 1)

            crossed_column_series = df[col0] * col1_width + df[col1]
            oe = skp.OrdinalEncoder(dtype=np.int64)
            crossed_column_series = pd.DataFrame(oe.fit_transform(pd.DataFrame(crossed_column_series)))
            crossed_column_series = crossed_column_series + 1

            crossed_column = col0 + '_' + col1
            df.insert(NUM_INTEGER_COLUMNS + 1 + idx, crossed_column, crossed_column_series)
            crossed_column_max_val = np.int64(df[crossed_column].max())
            logging.info('column {} [{}, {}]'.format(
                crossed_column,
                str(df[crossed_column].min()),
                str(crossed_column_max_val)))
            df[crossed_column] += offset
            logging.info('column {} offset {}'.format(crossed_column, str(offset)))
            offset += crossed_column_max_val

    return df

def _wait_futures_and_reset(futures):
    for future in futures:
        result = future.result()
        if result:
            print(result)
    futures = list()

def _process_chunks(executor, chunks_to_process, op, *argv):
    futures = list()
    for chunk in chunks_to_process:
        argv_list = list(argv)
        argv_list.insert(0, chunk)
        new_argv = tuple(argv_list)
        future = executor.submit(op, *new_argv)
        futures.append(future)
    _wait_futures_and_reset(futures)

def preprocess(src_txt_name, dst_txt_name, normalize_dense, feature_cross):
    cols = [idx2key(idx) for idx in range(0, NUM_TOTAL_COLUMNS)]
    series_list_dict = dict()

    with cf.ThreadPoolExecutor(max_workers=MAX_NUM_WORKERS) as executor:
        logging.info('read a CSV file')
        reader = pd.read_csv(src_txt_name, sep='\t',
                             names=cols,
                             chunksize=131072)

        logging.info('_fill_missing_features_and_split')
        for col in cols:
            series_list_dict[col] = list()
        _process_chunks(executor, reader, _fill_missing_features_and_split,
                        series_list_dict)

    with cf.ProcessPoolExecutor(max_workers=MAX_NUM_WORKERS) as executor:
        logging.info('_merge_and_transform_series')
        futures = list()
        dense_cols = [idx2key(idx+1) for idx in range(NUM_INTEGER_COLUMNS)]
        dst_series_list = list()
        min_max = dict()
        for col, src_series_list in series_list_dict.items():
            future = executor.submit(_merge_and_transform_series,
                                     src_series_list, col, dense_cols,
                                     normalize_dense)
            futures.append(future)

        for future in futures:
            col = None
            for idx, ret in enumerate(future.result()):
                try:
                    if idx == 0:
                        col = ret.columns[0]
                        dst_series_list.append(ret)
                    else:
                        min_max[col] = ret
                except:
                    print_exc()
        futures = list()

        logging.info('_merge_columns_and_feature_cross')
        feature_pairs = [('C1', 'C2'), ('C3', 'C4')]
        df = _merge_columns_and_feature_cross(dst_series_list, min_max, feature_pairs,
                                              feature_cross)

        
        logging.info('_convert_to_string')
        df = pd.DataFrame(df.values.astype(str))

        logging.info('write to a CSV file')
        df.to_csv(dst_txt_name, sep=' ', header=False, index=False)

        logging.info('done!')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Preprocssing Criteo Dataset')

    arg_parser.add_argument('--src_csv_path', type=str, required=True)
    arg_parser.add_argument('--dst_csv_path', type=str, required=True)
    arg_parser.add_argument('--normalize_dense', type=int, default=1)
    arg_parser.add_argument('--feature_cross', type=int, default=1)

    args = arg_parser.parse_args()

    src_csv_path = args.src_csv_path
    dst_csv_path = args.dst_csv_path

    normalize_dense = args.normalize_dense
    feature_cross = args.feature_cross

    if os.path.exists(src_csv_path) == False:
        sys.exit('ERROR: the file \'{}\' doesn\'t exist'.format(src_csv_path))

    if os.path.exists(dst_csv_path) == True:
        sys.exit('ERROR: the file \'{}\' exists'.format(dst_csv_path))

    preprocess(src_csv_path, dst_csv_path, normalize_dense, feature_cross)

