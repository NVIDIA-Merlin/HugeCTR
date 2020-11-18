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

# import nvtx.plugins.tf as nvtx_tf

import txt2tfrecord as utils
import tensorflow as tf
import sys
sys.path.append("../python")
import hugectr_tf_ops
from model import DeepFM_PluginEmbedding, DeepFM_OriginalEmbedding
import argparse
import logging
import time

tf.debugging.set_log_device_placement(False)
devices = tf.config.list_physical_devices("GPU")
for dev in devices:
    tf.config.experimental.set_memory_growth(dev, True)



@tf.function(input_signature=(tf.TensorSpec(shape=(), dtype=tf.int64), 
                              tf.TensorSpec(shape=(), dtype=tf.int32), 
                              tf.TensorSpec(shape=(), dtype=tf.int32)))
def _distributed_map_fn(elem, gpu_count, dev_id):
    return tf.cond(tf.math.logical_and(elem != -1, tf.cast(elem, dtype=tf.int32) % gpu_count == dev_id), lambda: elem, 
                    lambda: tf.ones_like(elem) * -1)

@tf.function(input_signature=(tf.TensorSpec(shape=[None, 26, 1], dtype=tf.int64),
                              tf.TensorSpec(shape=(), dtype=tf.int32)))
def _distribute_keys_for_distributed(all_keys, gpu_count):
    """
    This function convert dense keys to CSR formats. Used in distributed embedding type.
    """
    all_keys_flat = tf.reshape(all_keys, [-1])

    row_offsets = tf.TensorArray(dtype=tf.int64, size=gpu_count, dynamic_size=False, clear_after_read=True)
    value_tensors = tf.TensorArray(dtype=tf.int64, size=gpu_count, dynamic_size=False,
                                    element_shape=all_keys_flat.shape, clear_after_read=True)
    nnz_array = tf.TensorArray(dtype=tf.int64, size=gpu_count, dynamic_size=False, clear_after_read=True)

    for dev_id in tf.range(gpu_count, dtype=tf.int32):
        # erase keys which do not belong to this device
        vectorized_keys = tf.vectorized_map(lambda elem: _distributed_map_fn(elem, gpu_count, dev_id), all_keys_flat)
        # convert to CSR
        vectorized_keys = tf.reshape(vectorized_keys, [-1, all_keys.shape[-1]])
        indices = tf.where(vectorized_keys != -1)
        csr_sparse_matrix = tf.raw_ops.DenseToCSRSparseMatrix(dense_input=tf.cast(vectorized_keys, dtype=tf.float64), 
                                                                indices=indices)
        row_ptrs, col_inds, values = tf.raw_ops.CSRSparseMatrixComponents(csr_sparse_matrix=csr_sparse_matrix, 
                                                                            index=0, 
                                                                            type=tf.float64)
        row_ptrs = tf.cast(row_ptrs, dtype=tf.int64)
        values = tf.cast(values, dtype=tf.int64)
        values = tf.pad(values, paddings=[[0, tf.shape(all_keys_flat)[0] - tf.shape(values)[0]]])

        # return row_ptrs, col_inds, values
        row_offsets = row_offsets.write(dev_id, row_ptrs)
        value_tensors = value_tensors.write(dev_id, values)
        nnz_array = nnz_array.write(dev_id, row_ptrs[-1])

    return row_offsets.stack(), value_tensors.stack(), nnz_array.stack()

@tf.function
def _distribute_keys_for_distributed_coo(all_keys, gpu_count, batch_size, slot_num):
    """
    This function convert coo to CSR format. Used in distributed embedding type.
    tf.SparseTensor can be viewed as a coo format matrix. In order to convert it to csr format.
    1. calculate binary index based on value % gpu_count == dev_id
    2. choose coo->row_index, coo->col_index, values based on binary index
    3. create a new sparsetensor from coo->row_index and coo->col_index and values.
    4. convert sparsetensor to csr format. (Acctually, only coo->row_index needed to be convert to CSR.)
    """
    # convert dense tensor to sparse tensor
    all_keys_flat = tf.reshape(all_keys, [-1])
    reshape_keys = tf.reshape(all_keys, [-1, all_keys.shape[-1]]) #[batchsize * slot_num, max_nnz]
    valid_indices = tf.where(reshape_keys != -1) # [[row_index, col_index]]
    valid_keys = tf.gather_nd(reshape_keys, valid_indices) # [values]
    coo_indices = tf.transpose(valid_indices, perm=[1, 0]) #[[row_index], [col_index]]

    row_offsets = tf.TensorArray(dtype=tf.int64, size=gpu_count, dynamic_size=False, clear_after_read=True)
    value_tensors = tf.TensorArray(dtype=tf.int64, size=gpu_count, dynamic_size=False,
                                    element_shape=all_keys_flat.shape, 
                                    clear_after_read=True)
    nnz_array = tf.TensorArray(dtype=tf.int64, size=gpu_count, dynamic_size=False, clear_after_read=True)

    for dev_id in tf.range(gpu_count, dtype=tf.int32):
        # generate binary flag based on valid_keys
        dev_keys_indices = tf.where(tf.cast(valid_keys, dtype=tf.int32) % gpu_count == dev_id)
        row_indexes = tf.gather_nd(coo_indices[0], dev_keys_indices)
        col_indexes = tf.gather_nd(coo_indices[1], dev_keys_indices)
        dev_keys = tf.gather_nd(valid_keys, dev_keys_indices)

        sparse_indices = tf.transpose(tf.stack([row_indexes, col_indexes]), perm=[1, 0])
        csr_sparse_matrix = tf.raw_ops.SparseTensorToCSRSparseMatrix(indices=sparse_indices, 
                                                                     values=tf.cast(dev_keys, dtype=tf.float64), 
                                    dense_shape=(tf.shape(all_keys)[0] * tf.shape(all_keys)[1], tf.shape(all_keys)[2]))        
        row_ptrs, _, _ = tf.raw_ops.CSRSparseMatrixComponents(csr_sparse_matrix=csr_sparse_matrix, 
                                                                            index=0, 
                                                                            type=tf.float64)
        row_ptrs = tf.cast(row_ptrs, dtype=tf.int64)
        values = dev_keys
        # values = tf.cast(values, dtype=tf.int64)
        values = tf.pad(values, paddings=[[0, tf.shape(all_keys_flat)[0] - tf.shape(values)[0]]])

        # return row_ptrs, col_inds, values
        row_offsets = row_offsets.write(dev_id, row_ptrs)
        value_tensors = value_tensors.write(dev_id, values)
        nnz_array = nnz_array.write(dev_id, row_ptrs[-1])        
        
    return row_offsets.stack(), value_tensors.stack(), nnz_array.stack()


@tf.function
def _distribute_keys_for_localized(all_keys, gpu_count, batch_size, slot_num):
    """
    tf.SparseTensor is coo format. In order to convert coo to csr.
    1. calculate slot_idx = row_idex % min(slot_num, gpu_count)
    2. calculate binary flag based on slot_idx == dev_id
    3. choose coo->row_index based on binary flag. and then new_row_idx = coo->row_index // min(slot_num, gpu_count)
    4. choose coo->col_index and values based on binary flag.
    5. form a sparsetensor and convert to csr. (acctually, only new_row_idx need to be convert to csr.)
    """
    # convert dense tensor to sparse tensor
    slot_mod = slot_num % gpu_count
    all_keys_flat = tf.reshape(all_keys, [-1])
    reshape_keys = tf.reshape(all_keys, [-1, all_keys.shape[-1]]) #[batchsize * slot_num, max_nnz]

    valid_indices = tf.where(reshape_keys != -1) # N * [row_index, col_index]
    valid_keys = tf.gather_nd(reshape_keys, valid_indices) # N [values]
    coo_indices = tf.transpose(valid_indices, perm=[1, 0]) # [N * [row_index], N * [col_index]]

    min_mod = tf.cast(tf.math.minimum(slot_num, gpu_count), dtype=tf.int64) # the minimum in slot_num and gpu_count
    slot_dev_idx = coo_indices[0] % min_mod

    row_offsets = tf.TensorArray(dtype=tf.int64, size=gpu_count, dynamic_size=False, 
                                 clear_after_read=False,
                                 element_shape=[batch_size * slot_num + 1])
    value_tensors = tf.TensorArray(dtype=tf.int64, size=gpu_count, dynamic_size=False,
                                    element_shape=[batch_size * ((slot_num // gpu_count) + 1)], 
                                    clear_after_read=False)
    nnz_array = tf.TensorArray(dtype=tf.int64, size=gpu_count, dynamic_size=False, clear_after_read=False)

    for dev_id in tf.range(gpu_count, dtype=tf.int32):
        # generate binary flag based on valid_row_index
        dev_keys_indices = tf.where(tf.cast(slot_dev_idx, dtype=tf.int32) == dev_id)
        row_indexes = tf.gather_nd(coo_indices[0], dev_keys_indices)
        row_indexes = row_indexes // min_mod # convert to row_index in new matrix
        col_indexes = tf.gather_nd(coo_indices[1], dev_keys_indices)
        dev_keys = tf.gather_nd(valid_keys, dev_keys_indices)

        sparse_indices = tf.transpose(tf.stack([row_indexes, col_indexes]), perm=[1, 0])
        csr_sparse_matrix = tf.raw_ops.SparseTensorToCSRSparseMatrix(indices=sparse_indices, 
                                                                     values=tf.cast(dev_keys, dtype=tf.float64), 
                                    dense_shape= tf.cond(dev_id < slot_mod, 
                                        lambda: (tf.shape(all_keys)[0] * ((slot_num // gpu_count) + 1), tf.shape(all_keys)[2]),
                                        lambda: (tf.shape(all_keys)[0] * (slot_num // gpu_count), tf.shape(all_keys)[2])))        
        row_ptrs, _, _ = tf.raw_ops.CSRSparseMatrixComponents(csr_sparse_matrix=csr_sparse_matrix, 
                                                                          index=0, 
                                                                          type=tf.float64)
        values = dev_keys                                                                 
        # return row_ptrs, col_inds, values
        row_ptrs = tf.cast(row_ptrs, dtype=tf.int64)
        nnz_array = nnz_array.write(dev_id, row_ptrs[-1])
        row_ptrs = tf.pad(row_ptrs, paddings=[[0, tf.shape(reshape_keys)[0] + 1 - tf.shape(row_ptrs)[0]]])
        # values = tf.cast(values, dtype=tf.int64)
        values = tf.pad(values, paddings=[[0, batch_size * ((slot_num // gpu_count) + 1) - tf.shape(values)[0]]])
        row_offsets = row_offsets.write(dev_id, row_ptrs)
        value_tensors = value_tensors.write(dev_id, values)

    return row_offsets.stack(), value_tensors.stack(), nnz_array.stack()


@tf.function
def _distribute_kyes(all_keys, gpu_count, embedding_type, batch_size, slot_num):
    return tf.cond(tf.equal("distributed", embedding_type), 
                #    lambda: _distribute_keys_for_distributed_coo(all_keys, gpu_count, batch_size, slot_num),
                   lambda: _distribute_keys_for_distributed(all_keys, gpu_count),
                   lambda: _distribute_keys_for_localized(all_keys, gpu_count, batch_size, slot_num))

def create_dataset(dataset_names, feature_desc, batch_size, n_epochs=-1, 
                    distribute_keys=False, gpu_count=1, embedding_type='localized'):
    """
    This function is used to get batch of data from tfrecords file.
    #arguments:
        dataset_names: list of strings
        feature_des: feature description of the features in one sample.
    """
    # FIXME:
    if (embedding_type == "localized"):
        raise RuntimeError("there will be memory corruption error when calling _distribute_keys_for_localized."+\
                        " Therefore do not call it.")

    # num_threads = tf.data.experimental.AUTOTUNE
    num_threads = 32
    dataset = tf.data.TFRecordDataset(filenames=dataset_names, compression_type=None,
                                      buffer_size=100 * 1024 * 1024, num_parallel_reads=32)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(n_epochs)

    # data preprocessing
    @tf.function
    def _parse_fn(serialized, feature_desc,
                  distribute_keys=False, 
                  embedding_type="localized"):
        with tf.name_scope("datareader_map"):
            features = tf.io.parse_example(serialized, feature_desc)
            # split into label + dense + cate
            label = features['label']
            dense = tf.TensorArray(dtype=tf.int64, size=utils.NUM_INTEGER_COLUMNS, dynamic_size=False, 
                                    element_shape=(batch_size,))
            cate = tf.TensorArray(dtype=tf.int64, size=utils.NUM_CATEGORICAL_COLUMNS, dynamic_size=False,
                                    element_shape=(batch_size, 1))

            for idx in range(utils.NUM_INTEGER_COLUMNS):
                dense = dense.write(idx, features["I" + str(idx+1)])
            
            for idx in range(utils.NUM_CATEGORICAL_COLUMNS):
                cate = cate.write(idx, features["C" + str(idx+1)])
                
            dense = tf.transpose(dense.stack(), perm=[1, 0]) # [batchsize, dense_dim]
            cate = tf.transpose(cate.stack(), perm=[1, 0, 2]) # [batchsize, slot_num, nnz]

            # distribute cate-keys to each GPU
            if distribute_keys:
                # convert cate to SparseTensor
                # indices = tf.where(cate != -1)
                # values = tf.gather_nd(cate, indices)
                # row_offsets, value_tensors, nnz_array = hugectr_tf_ops.distribute_keys(indices, values, cate.shape,
                #                                             gpu_count = gpu_count, embedding_type='localized', max_nnz=1)
                row_offsets, value_tensors, nnz_array = _distribute_kyes(all_keys=cate, 
                                                    gpu_count=gpu_count,
                                                    embedding_type=tf.convert_to_tensor(embedding_type, dtype=tf.string),
                                                    batch_size=batch_size,
                                                    slot_num=26)
                place_holder = tf.sparse.SparseTensor([[0,0]], tf.constant([0], dtype=tf.int64), 
                                                      [batch_size * utils.NUM_CATEGORICAL_COLUMNS,1])
                return label, dense, row_offsets, value_tensors, nnz_array, place_holder
            else:
                reshape_cate = tf.reshape(cate, [-1, cate.shape[-1]])
                indices = tf.where(reshape_cate != -1)
                values = tf.gather_nd(reshape_cate, indices)
                sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=reshape_cate.shape)
                place_holder = tf.constant(1, dtype=tf.int64)
                return label, dense, place_holder, place_holder, place_holder, sparse_tensor

    dataset = dataset.map(lambda serialized: _parse_fn(serialized, 
                                                       feature_desc, 
                                                       tf.convert_to_tensor(distribute_keys, dtype=tf.bool),
                                                       tf.convert_to_tensor(embedding_type, dtype=tf.string)),
                          num_parallel_calls=32, # tf.data.experimental.AUTOTUNE
                          deterministic=False)
    dataset = dataset.prefetch(buffer_size=16) # tf.data.experimental.AUTOTUNE

    return dataset


def test_read_data(embedding_type, batch_size, display_steps, distribute_keys):
    cols = [utils.idx2key(idx, False) for idx in range(0, utils.NUM_TOTAL_COLUMNS)]
    feature_desc = dict()
    for col in cols:
        if col == 'label' or col.startswith("I"):
            feature_desc[col] = tf.io.FixedLenFeature([], tf.int64) # scaler
        else: 
            feature_desc[col] = tf.io.FixedLenFeature([1], tf.int64) # [slot_num, nnz]

    dataset_names = ["train.tfrecord"]
    dataset = create_dataset(dataset_names=dataset_names,
                             feature_desc=feature_desc,
                             batch_size=batch_size,
                             n_epochs=1,
                             distribute_keys=tf.constant(distribute_keys, dtype=tf.bool),
                             gpu_count=tf.constant(4, dtype=tf.int32),
                             embedding_type=tf.constant(embedding_type, dtype=tf.string))

    total_steps = 0                      
    total_begin_time = time.time()   
    begin_time = total_begin_time
    for step, datas in enumerate(dataset):
        total_steps += 1
        a = datas
        if step % display_steps == 0 and step != 0:
            end_time = time.time()
            tf.print("Elapsed time: %.5f for %d steps." %(end_time - begin_time, display_steps))
            begin_time = time.time()
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_begin_time
    print("Total elapsed time: %.5f seconds for %d steps. Average elapsed time: %.5f / step." 
            %(total_elapsed_time, total_steps, (total_elapsed_time / total_steps)))



if __name__ == "__main__":
    test_read_data(embedding_type='distributed', batch_size=16384, display_steps=100, distribute_keys=True)
