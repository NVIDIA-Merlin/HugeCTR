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
This script is a demo about how to use hugectr's embedding plugin.
"""
import tensorflow as tf
import sys
sys.path.append("./python")
import hugectr
import numpy as np
import scipy


tf.debugging.set_log_device_placement(False)
devices = tf.config.list_physical_devices("GPU")
for dev in devices:
    tf.config.experimental.set_memory_growth(dev, True)

def test():
    with tf.GradientTape() as tape:
        with tf.device("/gpu:0"):
            
            vocabulary_size = 8
            slot_num = 3
            embedding_vec_size = 4

            init_value = np.float32([i for i in range(1, vocabulary_size * embedding_vec_size + 1)]).reshape(vocabulary_size, embedding_vec_size)
            # init_value = False
            # print(init_value)

            hugectr.init(visiable_gpus=[0,1,3,4], seed=123, key_type='uint32', value_type='float', batch_size=4, batch_size_eval=4)
            embedding_name = hugectr.create_embedding(init_value=init_value, opt_hparams=[0.1, 0.9, 0.99, 1e-3], name_='test_embedding',
                                                    max_vocabulary_size_per_gpu=5, slot_num=slot_num, embedding_vec_size=embedding_vec_size,
                                                    max_feature_num=4, embedding_type='localized', max_nnz=2)
            # print(embedding_name)
            # embedding_name = hugectr.create_embedding(init_value=init_value, opt_hparams=[0.001, 0.9, 0.99, 1e-3], name_='test_embedding',
            #                                           max_vocabulary_size_per_gpu=5, slot_num=slot_num, embedding_vec_size=embedding_vec_size,
            #                                           max_feature_num=4)
            # print(embedding_name)
            # embedding_name = hugectr.create_embedding(init_value=init_value, opt_hparams=[0.001, 0.9, 0.99, 1e-3], name_='test_embedding',
            #                                           max_vocabulary_size_per_gpu=5, slot_num=slot_num, embedding_vec_size=embedding_vec_size,
            #                                           max_feature_num=4)
            # print(embedding_name)

            keys = np.array([[[0, -1, -1, -1],
                            [1, -1, -1, -1],
                            [2, 6, -1, -1]],

                            [[0, -1, -1, -1],
                            [1, -1, -1, -1],
                            [-1, -1, -1, -1]],
                            
                            [[0, -1, -1, -1],
                            [1, -1, -1, -1],
                            [6, -1, -1, -1]],
                            
                            [[0, -1, -1, -1],
                            [1, -1, -1, -1],
                            [2, -1, -1, -1]]], dtype=np.int64)

            sparse_indices = tf.where(keys != -1) #[N, ndims]
            values = tf.gather_nd(keys, sparse_indices) # [N]
            # print("sparse_indices = ", sparse_indices)
            # print("values = ", values)

            bp_trigger = tf.Variable(initial_value=[1.0, 2.0], trainable=True, dtype=tf.float32, 
                                    name='embedding_plugin_bprop_trigger') # must be trainable

            forward_result = hugectr.fprop(embedding_name=embedding_name, sparse_indices=sparse_indices, values=values, dense_shape=keys.shape,
                                            output_type=tf.float32, is_training=True, bp_trigger=bp_trigger)
            print("first step: \n", forward_result)

            grads = tape.gradient(forward_result, bp_trigger)

            forward_result = hugectr.fprop(embedding_name=embedding_name, sparse_indices=sparse_indices, values=values, dense_shape=keys.shape,
                                            output_type=tf.float32, is_training=False, bp_trigger=bp_trigger)
            print("second step: \n", forward_result)


            # tf embedding lookup op
            # new_keys = np.reshape(keys, newshape=(-1, keys.shape[-1]))

            # indices = tf.where(new_keys != -1)
            # values = tf.gather_nd(new_keys, indices)
            # sparse_tensor = tf.sparse.SparseTensor(indices, values, new_keys.shape)

            # tf_forward = tf.nn.embedding_lookup_sparse(init_value, sparse_tensor,
            #                                            sp_weights=None, combiner = "sum")
            # print("tf: \n", tf_forward)
            

def test_v2():
    with tf.GradientTape() as tape:
        with tf.device("/gpu:0"):
            
            vocabulary_size = 8
            slot_num = 3
            embedding_vec_size = 4

            init_value = np.float32([i for i in range(1, vocabulary_size * embedding_vec_size + 1)]).reshape(vocabulary_size, embedding_vec_size)
            # init_value = False
            # print(init_value)

            hugectr.init(visiable_gpus=[0,1,3,4], seed=123, key_type='int64', value_type='float', batch_size=4, batch_size_eval=4)
            embedding_name = hugectr.create_embedding(init_value=init_value, opt_hparams=[0.1, 0.9, 0.99, 1e-3], name_='test_embedding',
                                                    max_vocabulary_size_per_gpu=1737710, slot_num=slot_num, embedding_vec_size=embedding_vec_size,
                                                    max_feature_num=4, embedding_type='localized', max_nnz=2)

            keys = np.array([[[0, -1, -1, -1],
                            [1, -1, -1, -1],
                            [2, 6, -1, -1]],

                            [[0, -1, -1, -1],
                            [1, -1, -1, -1],
                            [-1, -1, -1, -1]],
                            
                            [[0, -1, -1, -1],
                            [1, -1, -1, -1],
                            [6, -1, -1, -1]],
                            
                            [[0, -1, -1, -1],
                            [1, -1, -1, -1],
                            [2, -1, -1, -1]]], dtype=np.int64)

            sparse_indices = tf.where(keys != -1) #[N, ndims]
            values = tf.gather_nd(keys, sparse_indices) # [N]

            row_offsets, value_tensors, nnz_array = hugectr.distribute_keys(sparse_indices, values, keys.shape,
                                    gpu_count = 4, embedding_type='localized', max_nnz=2)
            print("row_offsets = ", row_offsets, "\n")
            print("value_tensors = ", value_tensors, "\n")
            print("nnz_array = ", nnz_array, "\n")

            bp_trigger = tf.Variable(initial_value=[1.0, 2.0], trainable=True, dtype=tf.float32, 
                                    name='embedding_plugin_bprop_trigger') # must be trainable

            forward_result = hugectr.fprop_v2(embedding_name=embedding_name, row_offsets=row_offsets, nnz_array=nnz_array,
                                              value_tensors=value_tensors, is_training=True, bp_trigger=bp_trigger,
                                              output_shape=[4, slot_num, embedding_vec_size])
            print("first step: \n", forward_result)

            grads = tape.gradient(forward_result, bp_trigger)

            forward_result = hugectr.fprop_v2(embedding_name=embedding_name, row_offsets=row_offsets, nnz_array=nnz_array,
                                              value_tensors=value_tensors, is_training=False, bp_trigger=bp_trigger,
                                              output_shape=[4, slot_num, embedding_vec_size])
            print("second step: \n", forward_result)

def test_distribute_keys(embedding_type):
    keys = np.array([[[0, -1],
                      [1, -1],
                      [2,  6]],

                     [[0, -1],
                      [1, -1],
                      [-1,-1]],
                    
                     [[0, -1],
                      [1, -1],
                      [6, -1]],
                    
                     [[0, -1],
                      [1, -1],
                      [2, -1]]], dtype=np.int64)
    indices = tf.where(keys != -1)
    values = tf.gather_nd(keys, indices)
    
    row_offsets, value_tensors, nnz_array = hugectr.distribute_keys(indices, values, keys.shape,
                                                gpu_count=4, embedding_type=embedding_type, max_nnz=2)
    print("\n")
    print("row_offsets:", row_offsets, "\n")
    print("value_tensors:", value_tensors, "\n")
    print("nnz_array:", nnz_array)

    # distribute keys v2
    # row_offsets, value_tensors, nnz_array = hugectr.distribute_keys_v2(all_keys=keys, gpu_count=4, 
    #                                                                    embedding_type=embedding_type, max_nnz=2,
    #                                                                    batch_size=4, slot_num=3)
    # print("\n")
    # print("row_offset v2: ", row_offsets, "\n")
    # print("value_tensors v2: ", value_tensors, "\n")
    # print("nnz_array v2: ", nnz_array)


def test_forward_distribute_keys_v2(embedding_type):
    with tf.GradientTape() as tape:
        with tf.device("/gpu:0"):
            
            vocabulary_size = 8
            slot_num = 3
            embedding_vec_size = 4

            init_value = np.float32([i for i in range(1, vocabulary_size * embedding_vec_size + 1)]).reshape(vocabulary_size, embedding_vec_size)
            # init_value = False
            # print(init_value)

            hugectr.init(visiable_gpus=[0,1,3,4], seed=123, key_type='int64', value_type='float', batch_size=4, batch_size_eval=4)
            embedding_name = hugectr.create_embedding(init_value=init_value, opt_hparams=[1.0, 0.9, 0.99, 1e-3], name_='test_embedding',
                                                    max_vocabulary_size_per_gpu=1737710, slot_num=slot_num, embedding_vec_size=embedding_vec_size,
                                                    max_feature_num=4, embedding_type=embedding_type, max_nnz=2)

            keys = np.array([[[0, -1],
                            [1, -1],
                            [2,  6]],

                            [[0, -1],
                            [1, -1],
                            [-1,-1]],
                            
                            [[0, -1],
                            [1, -1],
                            [6, -1]],
                            
                            [[0, -1],
                            [1, -1],
                            [2, -1]]], dtype=np.int64)

            sparse_indices = tf.where(keys != -1) #[N, ndims]
            values = tf.gather_nd(keys, sparse_indices) # [N]

            row_offsets, value_tensors, nnz_array = hugectr.distribute_keys_v2(all_keys=keys, gpu_count=4, 
                                                                       embedding_type=embedding_type, max_nnz=2,
                                                                       batch_size=4, slot_num=3)
            print("row_offsets = ", row_offsets, "\n")
            print("value_tensors = ", value_tensors, "\n")
            print("nnz_array = ", nnz_array, "\n")

            bp_trigger = tf.Variable(initial_value=[1.0, 2.0], trainable=True, dtype=tf.float32, 
                                    name='embedding_plugin_bprop_trigger') # must be trainable

            forward_result = hugectr.fprop_v2(embedding_name=embedding_name, row_offsets=row_offsets, nnz_array=nnz_array,
                                              value_tensors=value_tensors, is_training=True, bp_trigger=bp_trigger,
                                              output_shape=[4, slot_num, embedding_vec_size])
            print("first step: \n", forward_result)

            grads = tape.gradient(forward_result, bp_trigger)

            forward_result = hugectr.fprop_v2(embedding_name=embedding_name, row_offsets=row_offsets, nnz_array=nnz_array,
                                              value_tensors=value_tensors, is_training=False, bp_trigger=bp_trigger,
                                              output_shape=[4, slot_num, embedding_vec_size])
            print("second step: \n", forward_result)


def test_forward_distribute_keys_v3(embedding_type):
    with tf.GradientTape() as tape:
        with tf.device("/gpu:0"):
            
            vocabulary_size = 8
            slot_num = 3
            embedding_vec_size = 4

            init_value = np.float32([i for i in range(1, vocabulary_size * embedding_vec_size + 1)]).reshape(vocabulary_size, embedding_vec_size)
            # init_value = False
            # print(init_value)

            hugectr.init(visiable_gpus=[0,1,3,4], seed=123, key_type='int64', value_type='float', batch_size=4, batch_size_eval=4)
            embedding_name = hugectr.create_embedding(init_value=init_value, opt_hparams=[1.0, 0.9, 0.99, 1e-3], name_='test_embedding',
                                                    max_vocabulary_size_per_gpu=1737710, slot_num=slot_num, embedding_vec_size=embedding_vec_size,
                                                    max_feature_num=int(1e6), embedding_type=embedding_type, max_nnz=2)

            keys = np.array([[[0, -1],
                            [1, -1],
                            [2,  6]],

                            [[0, -1],
                            [1, -1],
                            [-1,-1]],
                            
                            [[0, -1],
                            [1, -1],
                            [6, -1]],
                            
                            [[0, -1],
                            [1, -1],
                            [2, -1]]], dtype=np.int64)

            sparse_indices = tf.where(keys != -1) #[N, ndims]
            values = tf.gather_nd(keys, sparse_indices) # [N]

            row_offsets, value_tensors, nnz_array = hugectr.distribute_keys_v3(keys, 
                                                                               unique_name="distribute_keys_1",
                                                                               embedding_type=embedding_type,
                                                                               gpu_count=4,
                                                                               batch_size=4,
                                                                               slot_num=3,
                                                                               max_nnz=2)
            print("row_offsets = ", row_offsets, "\n")
            print("value_tensors = ", value_tensors, "\n")
            print("nnz_array = ", nnz_array, "\n")

            bp_trigger = tf.Variable(initial_value=[1.0, 2.0], trainable=True, dtype=tf.float32, 
                                    name='embedding_plugin_bprop_trigger') # must be trainable

            forward_result = hugectr.fprop_v2(embedding_name=embedding_name, row_offsets=row_offsets, nnz_array=nnz_array,
                                              value_tensors=value_tensors, is_training=True, bp_trigger=bp_trigger,
                                              output_shape=[4, slot_num, embedding_vec_size])
            print("first step: \n", forward_result)

            grads = tape.gradient(forward_result, bp_trigger)

            forward_result = hugectr.fprop_v2(embedding_name=embedding_name, row_offsets=row_offsets, nnz_array=nnz_array,
                                              value_tensors=value_tensors, is_training=False, bp_trigger=bp_trigger,
                                              output_shape=[4, slot_num, embedding_vec_size])
            print("second step: \n", forward_result)


def test_forward_distribute_keys_v4(embedding_type):
    with tf.GradientTape() as tape:
        with tf.device("/gpu:0"):
            
            vocabulary_size = 8
            slot_num = 3
            embedding_vec_size = 4

            init_value = np.float32([i for i in range(1, vocabulary_size * embedding_vec_size + 1)]).reshape(vocabulary_size, embedding_vec_size)
            # init_value = False
            # print(init_value)

            hugectr.init(visiable_gpus=[0,1,3,4], seed=123, key_type='int64', value_type='float', batch_size=4, batch_size_eval=4)
            embedding_name = hugectr.create_embedding(init_value=init_value, opt_hparams=[1.0, 0.9, 0.99, 1e-3], name_='test_embedding',
                                                    max_vocabulary_size_per_gpu=1737710, slot_num=slot_num, embedding_vec_size=embedding_vec_size,
                                                    max_feature_num=4, embedding_type=embedding_type, max_nnz=2)

            keys = np.array([[[0, -1],
                            [1, -1],
                            [2,  6]],

                            [[0, -1],
                            [1, -1],
                            [-1,-1]],
                            
                            [[0, -1],
                            [1, -1],
                            [6, -1]],
                            
                            [[0, -1],
                            [1, -1],
                            [2, -1]]], dtype=np.int64)

            sparse_indices = tf.where(keys != -1) #[N, ndims]
            values = tf.gather_nd(keys, sparse_indices) # [N]

            row_offsets, value_tensors, nnz_array = hugectr.distribute_keys_v4(all_keys=keys, gpu_count=4, 
                                                                       embedding_type=embedding_type, max_nnz=2,
                                                                       batch_size=4, slot_num=3)
            print("row_offsets = ", row_offsets, "\n")
            print("value_tensors = ", value_tensors, "\n")
            print("nnz_array = ", nnz_array, "\n")

            bp_trigger = tf.Variable(initial_value=[1.0, 2.0], trainable=True, dtype=tf.float32, 
                                    name='embedding_plugin_bprop_trigger') # must be trainable

            forward_result = hugectr.fprop_v2(embedding_name=embedding_name, row_offsets=row_offsets, nnz_array=nnz_array,
                                              value_tensors=value_tensors, is_training=True, bp_trigger=bp_trigger,
                                              output_shape=[4, slot_num, embedding_vec_size])
            print("first step: \n", forward_result)

            grads = tape.gradient(forward_result, bp_trigger)

            forward_result = hugectr.fprop_v2(embedding_name=embedding_name, row_offsets=row_offsets, nnz_array=nnz_array,
                                              value_tensors=value_tensors, is_training=False, bp_trigger=bp_trigger,
                                              output_shape=[4, slot_num, embedding_vec_size])
            print("second step: \n", forward_result)

@tf.function(input_signature=(tf.TensorSpec(shape=(), dtype=tf.int64), 
                              tf.TensorSpec(shape=(), dtype=tf.int32), 
                              tf.TensorSpec(shape=(), dtype=tf.int32)))
def _distributed_map_fn(elem, gpu_count, dev_id):
    print("_distributed_map_fn is traced.s, on Line 40")
    return tf.cond(tf.math.logical_and(elem != -1, tf.cast(elem, dtype=tf.int32) % gpu_count == dev_id), lambda: elem, 
                    lambda: tf.ones_like(elem) * -1)

@tf.function(input_signature=(tf.TensorSpec(shape=[None, 3, 2], dtype=tf.int64),
                              tf.TensorSpec(shape=(), dtype=tf.int32)))
def _distribute_keys_for_distributed(all_keys, gpu_count):
    print("_distribute_keys_for_distributed, on Line 46")
    all_keys_flat = tf.reshape(all_keys, [-1])

    row_offsets = tf.TensorArray(dtype=tf.int64, size=gpu_count, dynamic_size=True, clear_after_read=False)
    value_tensors = tf.TensorArray(dtype=tf.int64, size=gpu_count, dynamic_size=True,
                                    element_shape=all_keys_flat.shape, clear_after_read=False)
    nnz_array = tf.TensorArray(dtype=tf.int64, size=gpu_count, dynamic_size=True, clear_after_read=False)

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


@tf.function(input_signature=(tf.TensorSpec(shape=[None, 3, 2], dtype=tf.int64),
                              tf.TensorSpec(shape=(), dtype=tf.int32)))
def _distribute_keys_for_localized(all_keys, gpu_count):
    # currently not implemented
    print("_distribute_keys_for_localized, on Line 80")
    return _distribute_keys_for_distributed(all_keys, gpu_count)


@tf.function(input_signature=(tf.TensorSpec(shape=[None, 3, 2], dtype=tf.int64),
                              tf.TensorSpec(shape=(), dtype=tf.int32),
                              tf.TensorSpec(shape=(), dtype=tf.string)))
def _distribute_kyes(all_keys, gpu_count, embedding_type):
    print("_distribute_kyes, on Line 85")
    return tf.cond(tf.equal("distributed", embedding_type), 
                   lambda: _distribute_keys_for_distributed(all_keys, gpu_count),
                   lambda: _distribute_keys_for_localized(all_keys, gpu_count))



def tf_distribute_keys_fprop_v3(embedding_type):
    with tf.GradientTape() as tape:
        with tf.device("/gpu:0"):
            
            vocabulary_size = 8
            slot_num = 3
            embedding_vec_size = 4

            init_value = np.float32([i for i in range(1, vocabulary_size * embedding_vec_size + 1)]).reshape(vocabulary_size, embedding_vec_size)
            # init_value = False
            # print(init_value)

            hugectr.init(visiable_gpus=[0,1,3,4], seed=123, key_type='int64', value_type='float', batch_size=4, batch_size_eval=4)
            embedding_name = hugectr.create_embedding(init_value=init_value, opt_hparams=[1.0, 0.9, 0.99, 1e-3], name_='test_embedding',
                                                    max_vocabulary_size_per_gpu=1737710, slot_num=slot_num, embedding_vec_size=embedding_vec_size,
                                                    max_feature_num=4, embedding_type=embedding_type, max_nnz=2)

            keys = np.array([[[0, -1],
                            [1, -1],
                            [2,  6]],

                            [[0, -1],
                            [1, -1],
                            [-1,-1]],
                            
                            [[0, -1],
                            [1, -1],
                            [6, -1]],
                            
                            [[0, -1],
                            [1, -1],
                            [2, -1]]], dtype=np.int64)

            row_offsets, value_tensors, nnz_array = _distribute_kyes(tf.convert_to_tensor(keys), gpu_count=4, 
                                                                     embedding_type=embedding_type)
            print("row_ptrs", row_offsets)
            print("\nvalues", value_tensors)
            print("\n", nnz_array)

            row_offsets, value_tensors, nnz_array = _distribute_kyes(tf.convert_to_tensor(keys), gpu_count=4, embedding_type=embedding_type)
            print("\nrow_ptrs", row_offsets)
            print("\nvalues", value_tensors)
            print("\n", nnz_array)
            # print("\n", _distribute_kyes.pretty_printed_concrete_signatures(), "\n")

            bp_trigger = tf.Variable(initial_value=[1.0, 2.0], trainable=True, dtype=tf.float32, 
                                    name='embedding_plugin_bprop_trigger') # must be trainable

            forward_result = hugectr.fprop_v3(embedding_name=embedding_name, row_offsets=row_offsets, nnz_array=nnz_array,
                                                value_tensors=value_tensors, is_training=True, bp_trigger=bp_trigger,
                                                output_shape=[4, slot_num, embedding_vec_size])
            print("first step: \n", forward_result)

            grads = tape.gradient(forward_result, bp_trigger)

            forward_result = hugectr.fprop_v3(embedding_name=embedding_name, row_offsets=row_offsets, nnz_array=nnz_array,
                                                value_tensors=value_tensors, is_training=False, bp_trigger=bp_trigger,
                                                output_shape=[4, slot_num, embedding_vec_size])
            print("second step: \n", forward_result)

if __name__ == "__main__":
    # test()
    # test_v2()
    # test_distribute_keys("distributed")
    # test_forward_distribute_keys_v2("distributed")
    # test_forward_distribute_keys_v3("distributed")
    # test_forward_distribute_keys_v4("distributed")
    tf_distribute_keys_fprop_v3("distributed")

    pass