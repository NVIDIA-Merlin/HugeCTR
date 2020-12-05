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

embedding_name = None
def create_dataset(dataset_names, feature_desc, batch_size, n_epochs=-1, 
                    distribute_keys=False, gpu_count=1, embedding_type='localized',
                    use_which_device='cpu'):
    """
    This function is used to get batch of data from tfrecords file.
    #arguments:
        dataset_names: list of strings
        feature_des: feature description of the features in one sample.
    """
    # num_threads = tf.data.experimental.AUTOTUNE
    if use_which_device == 'gpu':
        global embedding_name
        hugectr_tf_ops.init(visiable_gpus=[i for i in range(gpu_count)], key_type='int64', value_type='float', 
                        batch_size=batch_size, batch_size_eval=gpu_count)
        embedding_name = hugectr_tf_ops.create_embedding(init_value=False, embedding_type=embedding_type,
                                                        opt_hparams=[1.0] * 4, slot_num=26, max_nnz=1,
                                                        max_feature_num=26 * 1, name_='hugectr_embedding')

    num_threads = 32
    dataset = tf.data.TFRecordDataset(filenames=dataset_names, compression_type=None,
                                      buffer_size=100 * 1024 * 1024, num_parallel_reads=1)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(n_epochs)

    # data preprocessing
    @tf.function
    def _parse_fn(serialized, feature_desc,
                  distribute_keys=False):
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
                def _distribute_keys_cpu(cate):
                    # --------------------- convert cate to CSR on CPU ---------------------------- #
                    indices = tf.where(cate != -1)
                    values = tf.gather_nd(cate, indices)
                    row_offsets, value_tensors, nnz_array = hugectr_tf_ops.distribute_keys(indices, values, cate.shape,
                                                                gpu_count = gpu_count, embedding_type=embedding_type, max_nnz=1)

                    place_holder = tf.sparse.SparseTensor([[0,0]], tf.constant([0], dtype=tf.int64), 
                                                          [batch_size * utils.NUM_CATEGORICAL_COLUMNS,1])
                    return label, dense, tf.stack(row_offsets), tf.stack(value_tensors), nnz_array, place_holder

                def _distribute_keys_gpu(cate):
                    # ---------------------- convert cate to CSR On GPU ------------------------ #
                    cate = tf.reshape(cate, [-1, 1])
                    indices = tf.where(cate != -1)
                    row_indices = tf.transpose(indices, perm=[1, 0])[0]
                    values = tf.gather_nd(cate, indices)

                    nnz_array = tf.constant(0, dtype=tf.int64)

                    place_holder = tf.sparse.SparseTensor([[0,0]], tf.constant([0], dtype=tf.int64), 
                                                        [batch_size * utils.NUM_CATEGORICAL_COLUMNS,1])
                    return label, dense, row_indices, values, nnz_array, place_holder

                if "cpu" == use_which_device:
                    return _distribute_keys_cpu(cate)
                else:
                    return _distribute_keys_gpu(cate)

            else:
                reshape_cate = tf.reshape(cate, [-1, cate.shape[-1]])
                indices = tf.where(reshape_cate != -1)
                values = tf.gather_nd(reshape_cate, indices)
                sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=reshape_cate.shape)
                place_holder = tf.constant(1, dtype=tf.int64)
                return label, dense, place_holder, place_holder, place_holder, sparse_tensor

    dataset = dataset.map(lambda serialized: _parse_fn(serialized, 
                                                       feature_desc, 
                                                       tf.convert_to_tensor(distribute_keys, dtype=tf.bool)),
                          num_parallel_calls=1, # tf.data.experimental.AUTOTUNE
                          deterministic=False)
    # dataset = dataset.prefetch(buffer_size=16) # tf.data.experimental.AUTOTUNE

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


# wrap above functions into a class
class CreateDataset(object):
    def __init__(self, 
                 dataset_names,
                 feature_desc,
                 batch_size,
                 n_epochs,
                 slot_num,
                 max_nnz,
                 convert_to_csr=False,
                 gpu_count=1,
                 embedding_type='localized',
                 get_row_indices=False):
        self.dataset_names = dataset_names
        self.feature_desc = feature_desc
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.slot_num = slot_num
        self.max_nnz = max_nnz
        self.convert_to_csr = convert_to_csr
        self.gpu_count = gpu_count
        self.embedding_type = embedding_type
        self.get_row_indices = get_row_indices

        if (self.convert_to_csr and self.get_row_indices):
            raise RuntimeError("convert_to_csr and get_row_indices could not be True at the same time.")

        self.num_threads = 32


    def __call__(self):
        dataset = tf.data.TFRecordDataset(filenames=self.dataset_names, compression_type=None,
                                            buffer_size=100 * 1024 * 1024, num_parallel_reads=self.num_threads)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.repeat(self.n_epochs)

        dataset = dataset.map(lambda serialized: self._parse_fn(serialized),
                              num_parallel_calls=1,
                              deterministic=False)

        dataset = dataset.prefetch(buffer_size=16)
        
        return dataset

    
    @tf.function
    def _parse_fn(self, serialized):
        with tf.name_scope("datareader_map"):
            features = tf.io.parse_example(serialized, self.feature_desc)

            label = features['label']
            dense = tf.TensorArray(dtype=tf.int64, size=utils.NUM_INTEGER_COLUMNS, dynamic_size=False,
                                   element_shape=(self.batch_size,))
            cate = tf.TensorArray(dtype=tf.int64, size=utils.NUM_CATEGORICAL_COLUMNS, dynamic_size=False,
                                    element_shape=(self.batch_size, 1))

            for idx in range(utils.NUM_INTEGER_COLUMNS):
                dense = dense.write(idx, features['I' + str(idx + 1)])

            for idx in range(utils.NUM_CATEGORICAL_COLUMNS):
                cate = cate.write(idx, features['C' + str(idx + 1)])

            dense = tf.transpose(dense.stack(), perm=[1, 0])
            cate = tf.transpose(cate.stack(), perm=[1, 0, 2])

            if self.convert_to_csr:
                row_offsets, value_tensors, nnz_array = self._distribute_keys(all_keys=cate)
                
                place_holder = tf.sparse.SparseTensor([[0,0]], tf.constant([0], dtype=tf.int64), 
                                                      [self.batch_size * utils.NUM_CATEGORICAL_COLUMNS,1])

                return label, dense, row_offsets, value_tensors, nnz_array, place_holder
            else:
                reshape_keys = tf.reshape(cate, [-1, self.max_nnz])
                indices = tf.where(reshape_keys != -1)
                values = tf.gather_nd(reshape_keys, indices)
                sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=reshape_keys.shape)
                place_holder = tf.constant(1, dtype=tf.int64)

                if self.get_row_indices:
                    row_indices = tf.transpose(indices, perm=[1, 0])[0]
                    return label, dense, row_indices, values, place_holder, sparse_tensor
                else:
                    return label, dense, place_holder, place_holder, place_holder, sparse_tensor

    @tf.function
    def _distribute_keys(self, all_keys):
        return tf.cond(tf.equal("distributed", self.embedding_type),
                        lambda: self._distribute_keys_for_distributed(all_keys),
                        lambda: self._distribute_keys_for_localized(all_keys)) 

    @tf.function
    def _localized_recompute_row_indices(self, row_indices, slot_mod, dev_id):
        batch_idx = row_indices // self.slot_num
        slot_idx = row_indices % self.slot_num
        dev_slot_idx = slot_idx // self.gpu_count
        dev_slot_num = tf.cast(self.slot_num // self.gpu_count + (1 if dev_id < slot_mod else 0), dtype=batch_idx.dtype)
        dev_row_indices = batch_idx * dev_slot_num + dev_slot_idx
        return dev_row_indices

    @tf.function
    def _distribute_keys_for_localized(self, all_keys):
        slot_mod = tf.cast(self.slot_num % self.gpu_count, dtype=tf.int32)
        reshape_keys = tf.reshape(all_keys, [self.batch_size * self.slot_num, self.max_nnz])
        
        valid_indices = tf.where(reshape_keys != -1)
        valid_keys = tf.gather_nd(reshape_keys, valid_indices)
        coo_indices = tf.transpose(valid_indices, perm=[1, 0])

        slot_dev_idx = tf.cast(coo_indices[0] % self.slot_num, dtype=tf.int32)

        roof_slot_num_gpu_count = self.slot_num // self.gpu_count
        roof_slot_num_gpu_count += (1 if self.slot_num % self.gpu_count != 0 else 0)

        row_offsets = tf.TensorArray(dtype=tf.int64, size=self.gpu_count, dynamic_size=False,
                                     clear_after_read=False, 
                                     element_shape=[self.batch_size * (roof_slot_num_gpu_count) + 1])
        value_tensors = tf.TensorArray(dtype=tf.int64, size=self.gpu_count, dynamic_size=False,
                                    clear_after_read=False,
                                    element_shape=[self.batch_size * (roof_slot_num_gpu_count) * self.max_nnz])
        nnz_array = tf.TensorArray(dtype=tf.int64, size=self.gpu_count, dynamic_size=False, clear_after_read=False)

        for dev_id in tf.range(self.gpu_count, dtype=tf.int32):
            flag_indices = tf.where(slot_dev_idx % self.gpu_count == dev_id)
            row_indexes = tf.gather_nd(coo_indices[0], flag_indices)

            # recompute dev row_idexes in each GPU
            row_indexes = self._localized_recompute_row_indices(row_indexes, slot_mod, dev_id)

            col_indexes = tf.gather_nd(coo_indices[1], flag_indices)
            dev_keys = tf.gather_nd(valid_keys, flag_indices)

            sparse_indices = tf.transpose(tf.stack([row_indexes, col_indexes]), perm=[1, 0])
            csr_sparse_matrix = tf.raw_ops.SparseTensorToCSRSparseMatrix(indices=sparse_indices, 
                                                                         values=tf.cast(dev_keys, dtype=tf.float64),
                                dense_shape=tf.cond(dev_id < slot_mod,
                                    lambda: (self.batch_size * ((self.slot_num // self.gpu_count) + 1), self.max_nnz),
                                    lambda: (self.batch_size * (self.slot_num // self.gpu_count), self.max_nnz)))

            row_ptrs, _, _ = tf.raw_ops.CSRSparseMatrixComponents(csr_sparse_matrix=csr_sparse_matrix,
                                                                  index=0,
                                                                  type=tf.float64)

            row_ptrs = tf.cast(row_ptrs, dtype=tf.int64)
            nnz_array = nnz_array.write(dev_id, row_ptrs[-1])
            row_ptrs = tf.pad(row_ptrs, paddings=[[0, self.batch_size * (roof_slot_num_gpu_count) + 1 - tf.shape(row_ptrs)[0]]])
            values = tf.pad(dev_keys, paddings=[[0, self.batch_size * (roof_slot_num_gpu_count) * self.max_nnz - tf.shape(dev_keys)[0]]])
            row_offsets = row_offsets.write(dev_id, row_ptrs)
            value_tensors = value_tensors.write(dev_id, values)

        return row_offsets.stack(), value_tensors.stack(), nnz_array.stack()
        

    @tf.function
    def _distribute_keys_for_distributed(self, all_keys):
        reshape_keys = tf.reshape(all_keys, [self.batch_size * self.slot_num, self.max_nnz])

        valid_indices = tf.where(reshape_keys != -1)
        valid_values = tf.gather_nd(reshape_keys, valid_indices)
        coo_indices = tf.transpose(valid_indices, perm=[1, 0])

        row_offsets = tf.TensorArray(dtype=tf.int64, size=self.gpu_count, dynamic_size=False,
                                    clear_after_read=True)
        value_tensors = tf.TensorArray(dtype=tf.int64, size=self.gpu_count, dynamic_size=False,
                                    element_shape=[self.batch_size * self.slot_num * self.max_nnz],
                                    clear_after_read=True)
        nnz_array = tf.TensorArray(dtype=tf.int64, size=self.gpu_count, dynamic_size=False, 
                                   clear_after_read=True)

        for dev_id in tf.range(self.gpu_count, dtype=tf.int32):
            binary_indices = tf.where(tf.cast(valid_values % self.gpu_count, dtype=tf.int32) == dev_id)
            row_indexes = tf.gather_nd(coo_indices[0], binary_indices)
            col_indexes = tf.gather_nd(coo_indices[1], binary_indices)
            dev_values = tf.gather_nd(valid_values, binary_indices)

            sparse_indices = tf.transpose(tf.stack([row_indexes, col_indexes]), perm=[1, 0])
            csr_sparse_matrix = tf.raw_ops.SparseTensorToCSRSparseMatrix(indices=sparse_indices,
                                                        values=tf.cast(dev_values, dtype=tf.float64),
                                            dense_shape=(self.batch_size * self.slot_num, self.max_nnz))

            row_ptrs, _, _ = tf.raw_ops.CSRSparseMatrixComponents(csr_sparse_matrix=csr_sparse_matrix, index=0, type=tf.float64)
            dev_values = tf.pad(dev_values, paddings=[[0, self.batch_size * self.slot_num * self.max_nnz - tf.shape(dev_values)[0]]])

            row_ptrs = tf.cast(row_ptrs, dtype=tf.int64)
            row_offsets = row_offsets.write(dev_id, row_ptrs)
            value_tensors = value_tensors.write(dev_id, dev_values)
            nnz_array = nnz_array.write(dev_id, row_ptrs[-1])

        return row_offsets.stack(), value_tensors.stack(), nnz_array.stack()


if __name__ == "__main__":
    # test_read_data(embedding_type='distributed', batch_size=16384, display_steps=100, distribute_keys=True)

    cols = [utils.idx2key(idx, False) for idx in range(0, utils.NUM_TOTAL_COLUMNS)]
    feature_desc = dict()
    for col in cols:
        if col == 'label' or col.startswith("I"):
            feature_desc[col] = tf.io.FixedLenFeature([], tf.int64) # scaler
        else: 
            feature_desc[col] = tf.io.FixedLenFeature([1], tf.int64) # [slot_num, nnz]


    embedding_type_list = ['localized'] #['localized', 'distributed']
    batch_size_list = [65536] #[512, 1024, 16384, 65536]
    gpu_count_list = [8] #[1, 2, 4, 8]
    iterations = 10

    for embedding_type in embedding_type_list:
        for gpu_count in gpu_count_list:
            for batch_size in batch_size_list:

                dataset_0 = create_dataset(dataset_names=['./train.tfrecord'],
                                        feature_desc=feature_desc,
                                        batch_size=batch_size,
                                        n_epochs=1,
                                        distribute_keys=True,
                                        gpu_count=gpu_count,
                                        embedding_type=embedding_type,
                                        use_which_device='gpu')

                dataset_1 = CreateDataset(dataset_names=['./train.tfrecord'],
                                        feature_desc=feature_desc,
                                        batch_size=batch_size,
                                        n_epochs=1,
                                        slot_num=26,
                                        max_nnz=1,
                                        convert_to_csr=True,
                                        gpu_count=gpu_count,
                                        embedding_type=embedding_type)()

                dataset_0 = iter(dataset_0)
                dataset_1 = iter(dataset_1)

                for iter_i in range(iterations):
                    datas_0 = next(dataset_0)
                    row_indices, values, nnz_array_0 = datas_0[2:5]
                    # TODO: ONLY FOR CORRECTNESS VERFIFYING.
                    row_offsets_0, value_tensors_0, nnz_array_0 = hugectr_tf_ops.distribute_keys_gpu(row_indices=row_indices,
                                                                            values=values, embedding_name=embedding_name,
                                                                            embedding_type=embedding_type,
                                                                            batch_size=batch_size,
                                                                            slot_num=26,
                                                                            gpu_count=gpu_count,
                                                                            max_nnz=1)

                    
                    datas_1 = next(dataset_1)
                    row_offsets_1, value_tensors_1, nnz_array_1 = datas_1[2:5]

                    try:
                        tf.debugging.assert_equal(row_offsets_0[:, 0:row_offsets_1.shape[1]], row_offsets_1)
                        tf.debugging.assert_equal(value_tensors_0[:, 0:value_tensors_1.shape[1]], value_tensors_1)
                        tf.debugging.assert_equal(nnz_array_0, nnz_array_1)
                    except tf.errors.InvalidArgumentError as error:
                        print("plugin_ops:\n", row_offsets_0) # row_offsets_0[1, 10:20]
                        print("tf_ops:\n", row_offsets_1) # row_offsets_1[1, 10:20]
                        raise RuntimeError("Error in %s, gpu_count %d, batch_size %d." %(embedding_type, gpu_count, batch_size),
                                        error.message)

                    print("For %s and gpu_count: %d, batch_size: %d, iteration: %d results is the same." 
                            %(embedding_type, gpu_count, batch_size, iter_i))
