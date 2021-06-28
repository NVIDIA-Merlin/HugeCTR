"""
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import sys
import logging
import tensorflow as tf
import os

NUM_INTEGER_COLUMNS = 13
NUM_CATEGORICAL_COLUMNS = 26
NUM_TOTAL_COLUMNS = 1 + NUM_INTEGER_COLUMNS + NUM_CATEGORICAL_COLUMNS

CROSSED_COLUMN_NAMES = ['C1_C2', 'C3_C4']
NUM_CROSSED_COLUMNS = len(CROSSED_COLUMN_NAMES)

EX_NUM_TOTAL_COLUMNS = NUM_TOTAL_COLUMNS + NUM_CROSSED_COLUMNS

def idx2key(idx, feature_crossed):
    if feature_crossed:
        if idx == 0:
            return 'label'
        elif idx <= NUM_INTEGER_COLUMNS:
            return 'I' + str(idx)
        elif idx <= (NUM_INTEGER_COLUMNS + NUM_CROSSED_COLUMNS):
            return CROSSED_COLUMN_NAMES[idx - (NUM_INTEGER_COLUMNS + NUM_CROSSED_COLUMNS)]
        else:
            return 'C' + str(idx - (NUM_INTEGER_COLUMNS + NUM_CROSSED_COLUMNS))
    else:
        if idx == 0:
            return 'label'
        elif idx <= NUM_INTEGER_COLUMNS:
            return 'I' + str(idx)
        else:
            return 'C' + str(idx - NUM_INTEGER_COLUMNS)


def line2fea_dict(idx, line, crossed, normalized):
    vals = line.split()
    vals = [v for v in vals]
    if idx == 0 and len(vals) == EX_NUM_TOTAL_COLUMNS:
        crossed = True
    fea_dict = dict()
    for idx in range(0, len(vals)):
        key = idx2key(idx, crossed)
        val = vals[idx]
        fea = None
        if idx == 0:
            val = [int(val)]
            fea = tf.train.Feature(int64_list=tf.train.Int64List(value=val))
        elif idx <= NUM_INTEGER_COLUMNS:
            if normalized == 1:
                val = [float(val)]
                fea = tf.train.Feature(float_list=tf.train.FloatList(value=val))
            else:
                val = [int(val)]
                fea = tf.train.Feature(int64_list=tf.train.Int64List(value=val))
        else:
            val = [int(val)]
            fea = tf.train.Feature(int64_list=tf.train.Int64List(value=val))
        fea_dict[key] = fea

    return fea_dict


def _convert_per_process(src_txt_name, dst_tfrecord_name, normalized, skip_lines, samples_num):
    with open(src_txt_name) as f:
        with tf.io.TFRecordWriter(dst_tfrecord_name) as writer:
            crossed = False
            for idx, line in enumerate(f):
                if idx < skip_lines:
                    continue
                elif (idx >= skip_lines + samples_num):
                    break
                else:
                    fea_dict = line2fea_dict(idx, line, crossed, normalized)
                    example = tf.train.Example(features = tf.train.Features(feature = fea_dict))
                    writer.write(example.SerializeToString())


def txt2tfrecord(src_txt_name, dst_tfrecord_name, normalized, shard_num=1, use_multi_process=0):
    lines_each_shard = [-1 for _ in range(shard_num)]
    # num of samples range [left, right)
    left_range = [0 for _ in range(shard_num)]
    right_range = [0 for _ in range(shard_num)]
    if shard_num > 1:
        # decide the number of samples in each shard 
        lines_num = 0
        with open(src_txt_name, "rb") as f:
            while True:
                buffer = f.read(1024 * 8192)
                if not buffer:
                    break
                lines_num += buffer.count(bytes("\n", encoding='utf8'))     
        if lines_num % shard_num != 0:
            left = lines_num % shard_num
            lines_each_shard = [(lines_num // shard_num) + (1 if i < left else 0) 
                                for i in range(shard_num)]
        else:
            lines_each_shard = [(lines_num // shard_num) for _ in range(shard_num)]
        logging.info('There are {} lines in {}, and shard_num is {}'.format(lines_num, src_txt_name, shard_num))
        logging.info("And the number of samples in each shard is:")
        for i, lines_shard in enumerate(lines_each_shard):
            logging.info("{}".format(lines_shard))
            if i == 0:
                left_range[i] = 0
                right_range[i] = lines_shard
            else:
                left_range[i] = right_range[i-1]
                right_range[i] = right_range[i-1] + lines_shard

    if shard_num == 1:
        with open(src_txt_name) as f:
            with tf.io.TFRecordWriter(dst_tfrecord_name) as writer:
                crossed = False
                for idx, line in enumerate(f):

                    fea_dict = line2fea_dict(idx, line, crossed, normalized)

                    example = tf.train.Example(features = tf.train.Features(feature = fea_dict))
                    writer.write(example.SerializeToString())
    else:
        if use_multi_process == 0: # Single Process to do converting.
            dst = dst_tfrecord_name.split(".tfrecord")
            names =[dst[0] + "_" + str(shard_idx) + ".tfrecord"
                    for shard_idx in range(shard_num)]
            writers = [tf.io.TFRecordWriter(names[shard_idx])
                    for shard_idx in range(shard_num)]

            shard_idx = 0
            logging.info('Writing samples into {}'.format(names[shard_idx]))
            with open(src_txt_name) as f:
                crossed = False
                for idx, line in enumerate(f):
                    if idx >= right_range[shard_idx]:
                        shard_idx += 1
                        logging.info('Writing samples into {}'.format(names[shard_idx]))

                    fea_dict = line2fea_dict(idx, line, crossed, normalized)

                    example = tf.train.Example(features = tf.train.Features(feature = fea_dict))
                    writers[shard_idx].write(example.SerializeToString())

            for writer in writers:
                writer.close()
        
        else: # multi-process to do converting.
            from multiprocessing import Pool
            logging.info("Use multi-processing to do converting..")

            dst = dst_tfrecord_name.split(".tfrecord")
            names =[dst[0] + "_" + str(shard_idx) + ".tfrecord"
                    for shard_idx in range(shard_num)]

            process_pool = Pool()
            for shard_idx in range(shard_num):
                process_pool.apply_async(_convert_per_process, (src_txt_name, names[shard_idx], normalized, 
                                                        left_range[shard_idx], lines_each_shard[shard_idx]))
            process_pool.close()
            process_pool.join()




if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.root.setLevel('INFO')

    arg_parser = argparse.ArgumentParser(description='Convert txt to TfRecord file.')
    arg_parser.add_argument('--src_txt_name', type=str, required=True)
    arg_parser.add_argument('--dst_tfrecord_name', type=str, required=True)
    arg_parser.add_argument('--normalized', type=int, required=True)
    arg_parser.add_argument('--shard_num', type=int, required=False, default=1)
    arg_parser.add_argument('--use_multi_process', type=int, required=False, default=0, choices=[0, 1])

    args = arg_parser.parse_args()

    logging.info('Convert {} to {}'.format(args.src_txt_name, args.dst_tfrecord_name))
    txt2tfrecord(args.src_txt_name, args.dst_tfrecord_name, args.normalized, args.shard_num,
                 args.use_multi_process)
    logging.info('Done!')
