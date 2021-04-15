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

"""
This script is only used for input format description.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

class CreateDataset(object):
    """
    This class is used to preprocess input data format of embedding plugin.
    """
    def __init__(self, 
                 dataset_names,              # list of tfrecord file names
                 feature_desc,               # its feature_description, please refer to TensorFlow documents
                 batch_size,                 # the generated batch_size
                 n_epochs,                   # number of epochs
                 slot_num,                   # number of slots (feature-fields)
                 max_nnz,                    # the max of non-zeros in all slots
                 convert_to_csr=False,       # whether converte input dense tensor to CSR format
                 gpu_count=1,                # how many GPUs are used for training
                 embedding_type='localized', # which embedding, could be 'distributed' or 'localized'
                 get_row_indices=False):     # whether converte input dense tensor to COO format
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
        """
        This function is used to read tfrecords with tf.data module.
        Please refer to TensorFlow documents for more details.
        """
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

            # cate stored all input keys, and it is a dense tensor whose shape is [batchsize, slot_num, max_nnz]
            cate = tf.transpose(cate.stack(), perm=[1, 0, 2]) 

            if self.convert_to_csr:
                """
                If need to be converted to CSR format
                """
                # this function will convert it to CSR
                row_offsets, value_tensors, nnz_array = self._distribute_keys(all_keys=cate) 
                
                place_holder = tf.sparse.SparseTensor([[0,0]], tf.constant([0], dtype=tf.int64), 
                                                      [self.batch_size * utils.NUM_CATEGORICAL_COLUMNS,1])

                return label, dense, row_offsets, value_tensors, nnz_array, place_holder
            else:
                reshape_keys = tf.reshape(cate, [-1, self.max_nnz]) # reshape cate to [batchsize * slot_num, max_nnz]
                indices = tf.where(reshape_keys != -1) # get the indices of all valid values, -1 represents invalid value
                values = tf.gather_nd(reshape_keys, indices) # get all valid values
                sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=reshape_keys.shape) # form a tf.SparseTensor
                place_holder = tf.constant(1, dtype=tf.int64)

                if self.get_row_indices:
                    """
                    If need to be converted to COO format
                    """
                    # get the row-indices of all valid values, column-indices will be ignored.
                    row_indices = tf.transpose(indices, perm=[1, 0])[0]
                    return label, dense, row_indices, values, place_holder, sparse_tensor
                else:
                    return label, dense, place_holder, place_holder, place_holder, sparse_tensor

    @tf.function
    def _distribute_keys(self, all_keys):
        """
        use different function based on embedding_type
        # arguments:
            all_keys: dense tensor, whose shape is [batchsize, slot_num, max_nnz]
        """
        return tf.cond(tf.equal("distributed", self.embedding_type),
                        lambda: self._distribute_keys_for_distributed(all_keys),
                        lambda: self._distribute_keys_for_localized(all_keys)) 

    @tf.function
    def _localized_recompute_row_indices(self, row_indices, slot_mod, dev_id):
        """
        helper function for localized embedding
        """
        batch_idx = row_indices // self.slot_num
        slot_idx = row_indices % self.slot_num
        dev_slot_idx = slot_idx // self.gpu_count
        dev_slot_num = tf.cast(self.slot_num // self.gpu_count + (1 if dev_id < slot_mod else 0), dtype=batch_idx.dtype)
        dev_row_indices = batch_idx * dev_slot_num + dev_slot_idx
        return dev_row_indices

    @tf.function
    def _distribute_keys_for_localized(self, all_keys):
        """
        convert dense tensor to CSR format for 'localized' embedding.
        # arguments:
            all_keys: a dense tensor, whose shape is [batchsize, slot_num, max_nnz]
        # returns:
            row_offsets: a list of dense tensors. Each element represents CSR.row_offset for one specific GPU.
                For example, row_offsets[0] is the CSR.row_offset for GPU 0.
            value_tensors: a list of dense tensors. Each element represents CSR.values for one specific GPU.
                For example, value_tensors[0] is the CSR.values for GPU 0.
            nnz_array: a list of scalers. Each element equals to the number of elements for one specific value_tensor.
                For example, nnz_array[0] = NumElements(value_tensors[0]) 
        """
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
        """
        convert dense tensor to CSR format for 'distributed' embedding.
        # arguments:
            all_keys: a dense tensor, whose shape is [batchsize, slot_num, max_nnz]
        # returns:
            row_offsets: a list of dense tensors. Each element represents CSR.row_offset for one specific GPU.
                For example, row_offsets[0] is the CSR.row_offset for GPU 0.
            value_tensors: a list of dense tensors. Each element represents CSR.values for one specific GPU.
                For example, value_tensors[0] is the CSR.values for GPU 0.
            nnz_array: a list of scalers. Each element equals to the number of elements for one specific value_tensor.
                For example, nnz_array[0] = NumElements(value_tensors[0]) 
        """
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
