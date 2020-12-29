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
import sys
sys.path.append("./python")
sys.path.append("./performance_profile")
import txt2tfrecord as utils
import tensorflow as tf
import hugectr_tf_ops
from model import OriginalEmbedding
from read_data import create_dataset, CreateDataset
import argparse
import logging
import time
import numpy as np

tf.debugging.set_log_device_placement(False)
devices = tf.config.list_physical_devices("GPU")
for dev in devices:
    tf.config.experimental.set_memory_growth(dev, True)

cols = [utils.idx2key(idx, False) for idx in range(0, utils.NUM_TOTAL_COLUMNS)]
feature_desc = dict()
for col in cols:
    if col == 'label' or col.startswith("I"):
        feature_desc[col] = tf.io.FixedLenFeature([], tf.int64) # scaler
    else: 
        feature_desc[col] = tf.io.FixedLenFeature([1], tf.int64) # [slot_num, nnz]

def Convert_to_csr_test(batch_size, gpu_count, embedding_type, iterations=10):
    def _plugin_CPU_op_VS_tf_ops():
        """
        Compare the result of converting to CSR between plugin CPU ops and tf ops.
        """
        dataset_names = ['./performance_profile/train.tfrecord']
        dataset_cpu = create_dataset(dataset_names=dataset_names, 
                                     feature_desc=feature_desc,
                                     batch_size=batch_size, 
                                     n_epochs=1,
                                     distribute_keys=True,
                                     gpu_count=gpu_count,
                                     embedding_type=embedding_type,
                                     use_which_device='cpu')
        dataset_tf = CreateDataset(dataset_names=dataset_names,
                                    feature_desc=feature_desc,
                                    batch_size=batch_size,
                                    n_epochs=1,
                                    slot_num=26,
                                    max_nnz=1,
                                    convert_to_csr=True,
                                    gpu_count=gpu_count,
                                    embedding_type=embedding_type)()

        dataset_cpu = iter(dataset_cpu)
        dataset_tf = iter(dataset_tf)

        for iter_i in range(iterations):
            row_offsets_cpu, value_tensor_cpu, nnz_array_cpu = next(dataset_cpu)[2:5]
            row_offsets_tf, value_tensor_tf, nnz_array_tf = next(dataset_tf)[2:5]

            try:
                tf.debugging.assert_equal(row_offsets_cpu[:, 0:row_offsets_tf.shape[1]], row_offsets_tf)
                tf.debugging.assert_equal(value_tensor_cpu[:, 0:value_tensor_tf.shape[1]], value_tensor_tf)
                tf.debugging.assert_equal(nnz_array_cpu, nnz_array_tf)
            except tf.errors.InvalidArgumentError as error:
                raise RuntimeError("Error in %s, gpu_count %d, batch_size %d." %(embedding_type, gpu_count, batch_size),
                                error.message)

            print("[INFO]: For %s and gpu_count: %d, batch_size: %d, iteration: %d results is the same." 
                    %(embedding_type, gpu_count, batch_size, iter_i))

    def _plugin_GPU_op_VS_tf_ops():
        """
        Compare the result of converting to CSR between plugin GPU ops and tf ops.
        """
        dataset_names = ['./performance_profile/train.tfrecord']
        dataset_gpu = create_dataset(dataset_names=dataset_names, 
                                     feature_desc=feature_desc,
                                     batch_size=batch_size, 
                                     n_epochs=1,
                                     distribute_keys=True,
                                     gpu_count=gpu_count,
                                     embedding_type=embedding_type,
                                     use_which_device='gpu')
        dataset_tf = CreateDataset(dataset_names=dataset_names,
                                    feature_desc=feature_desc,
                                    batch_size=batch_size,
                                    n_epochs=1,
                                    slot_num=26,
                                    max_nnz=1,
                                    convert_to_csr=True,
                                    gpu_count=gpu_count,
                                    embedding_type=embedding_type)()

        dataset_gpu = iter(dataset_gpu)
        dataset_tf = iter(dataset_tf)

        for iter_i in range(iterations):
            row_indices, values, nnz_array_gpu = next(dataset_gpu)[2:5]
            row_offsets_gpu, value_tensor_gpu, nnz_array_gpu = hugectr_tf_ops.distribute_keys_gpu(row_indices=row_indices,
                                                                                                  values=values,
                                                                                                  embedding_name='hugectr_embedding',
                                                                                                  embedding_type=embedding_type,
                                                                                                  batch_size=batch_size,
                                                                                                  slot_num=26,
                                                                                                  gpu_count=gpu_count,
                                                                                                  max_nnz=1)

            row_offsets_tf, value_tensor_tf, nnz_array_tf = next(dataset_tf)[2:5]

            try:
                tf.debugging.assert_equal(row_offsets_gpu[:, 0:row_offsets_tf.shape[1]], row_offsets_tf)
                tf.debugging.assert_equal(value_tensor_gpu[:, 0:value_tensor_tf.shape[1]], value_tensor_tf)
                tf.debugging.assert_equal(nnz_array_gpu, nnz_array_tf)
            except tf.errors.InvalidArgumentError as error:
                raise RuntimeError("Error in %s, gpu_count %d, batch_size %d." %(embedding_type, gpu_count, batch_size),
                                error.message)

            print("[INFO]: For %s and gpu_count: %d, batch_size: %d, iteration: %d results is the same." 
                    %(embedding_type, gpu_count, batch_size, iter_i))

        hugectr_tf_ops.reset()



    # # check convert to CSR via CPU and tf ops
    # for batch_size in [1024, 16384, 65536]:
    #     for gpu_count in [1, 2, 4, 8]:
    #         for embedding_type in ['localized', 'distributed']:
    #             _plugin_CPU_op_VS_tf_ops()

    # check GPU via tf ops, TODO: write shell script to do multiple testing.
    _plugin_GPU_op_VS_tf_ops()
    _plugin_GPU_op_VS_tf_ops()


def Embedding_ops_test(vocabulary_size, slot_num, max_nnz, embedding_vec_size, batch_size, gpus, embedding_type):
    """
    test forward propagation result with tf embedding layer.
    And do backward, then check forward propagation again.
    """
    def _fprop_VS_tf():
        print("[INFO]: Testing fprop vs tf...")
        if vocabulary_size < slot_num:
            raise RuntimeError("vocabulary_size must > slot.")
        with tf.GradientTape(persistent=True) as tape:
            # initial embedding table
            init_value = np.float32(np.random.normal(loc=0, scale=1, size=(vocabulary_size, embedding_vec_size)))
            # input keys
            # TODO: Keys in different slots should be unique.
            input_keys = np.ones(shape=(batch_size, slot_num, max_nnz), dtype=np.int64) * -1
            each_slot = vocabulary_size // slot_num
            nnz_0_num = 0
            for batch_id in range(batch_size):
                for slot_id in range(slot_num):
                    nnz = np.random.randint(low=nnz_0_num, high=max_nnz+1, size=1)[0] # how many keys in this slot
                    if nnz == 0:
                        nnz_0_num = 1
                    if (embedding_type == 'distributed'):
                        keys = np.random.randint(low=slot_id * each_slot, high=(slot_id + 1) * each_slot, size=nnz)
                    elif (embedding_type == "localized"):
                        # TODO: key should belong to that slot.
                        keys = []
                        while len(keys) < nnz:
                            key = np.random.randint(low=slot_id * each_slot, high=(slot_id + 1) * each_slot, size=1)
                            if key % slot_num == slot_id:
                                keys.append(key)

                    input_keys[batch_id, slot_id, 0:nnz] = keys

            # hugectr ops
            hugectr_tf_ops.init(visiable_gpus=gpus, key_type='int64', value_type='float', 
                                batch_size=batch_size, batch_size_eval=len(gpus))
            embedding_name = hugectr_tf_ops.create_embedding(init_value=init_value, opt_hparams=[0.1, 0.9, 0.99, 1e-5],
                                        name_='hugectr_embedding', max_vocabulary_size_per_gpu= (vocabulary_size // len(gpus))* 2 + 1,
                                        slot_num=slot_num, embedding_vec_size=embedding_vec_size,
                                        max_feature_num=slot_num*max_nnz, embedding_type=embedding_type, 
                                        max_nnz=max_nnz, update_type='Global')
            indices = tf.where(input_keys != -1)
            values = tf.gather_nd(input_keys, indices)
            
            bp_trigger = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)
            
            hugectr_forward = hugectr_tf_ops.fprop(embedding_name=embedding_name, sparse_indices=indices, values=values,
                                                   dense_shape=input_keys.shape, output_type=tf.float32, is_training=True,
                                                   bp_trigger=bp_trigger)
            # print("hugectr_results=\n", hugectr_forward)

            # tf ops
            reshape_input_keys = np.reshape(input_keys, [-1, max_nnz])
            tf_indices = tf.where(reshape_input_keys != -1)
            tf_values = tf.gather_nd(reshape_input_keys, tf_indices)
            sparse_tensor = tf.sparse.SparseTensor(tf_indices, tf_values, reshape_input_keys.shape)

            # FIXME: if there are too more nnz=0 slots, tf.nn.embedding_lookup_sparse may get wrong results?
            tf_embedding_layer = OriginalEmbedding(vocabulary_size=vocabulary_size,
                                                   embedding_vec_size=embedding_vec_size,
                                                   initializer=init_value,
                                                   combiner='sum',
                                                   gpus=gpus)

            tf_forward = tf_embedding_layer(sparse_tensor, output_shape=[batch_size, slot_num, embedding_vec_size])
            # print("tf_results=\n", tf_forward)

            # compare first forward result
            try:
                tf.debugging.assert_near(hugectr_forward, tf_forward)
            except tf.errors.InvalidArgumentError as error:
                raise error
            
            print("[INFO]: The results from HugeCTR and tf in the first forward propagation are the same.")

        # backward
        hugectr_grads = tape.gradient(hugectr_forward, bp_trigger)

        tf_opt = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.99, epsilon=1e-5)
        tf_grads = tape.gradient(tf_forward, tf_embedding_layer.trainable_weights)
        tf_opt.apply_gradients(zip(tf_grads, tf_embedding_layer.trainable_weights))

        # compare second forward result
        hugectr_forward_2 = hugectr_tf_ops.fprop(embedding_name=embedding_name, sparse_indices=indices, values=values,
                                                dense_shape=input_keys.shape, output_type=tf.float32, is_training=True,
                                                bp_trigger=bp_trigger)
                                                
        tf_forward_2 = tf_embedding_layer(sparse_tensor, output_shape=[batch_size, slot_num, embedding_vec_size])

        # print("hugectr 2:\n", hugectr_forward_2)
        # print("tf 2:\n", tf_forward_2)
        try:
            tf.debugging.assert_near(hugectr_forward_2, tf_forward_2, rtol=1e-4, atol=1e-5)
        except tf.errors.InvalidArgumentError as error:
            raise error

        print("[INFO]: The results from HugeCTR and tf in the second forward propagation are the same.")
        hugectr_tf_ops.reset()

    def _fprop_v3_VS_tf():
        print("[INFO]: Testing fprop_v3 vs tf...")
        if vocabulary_size < slot_num:
            raise RuntimeError("vocabulary_size must > slot.")
        with tf.GradientTape(persistent=True) as tape:
            # initial embedding table
            init_value = np.float32(np.random.normal(loc=0, scale=1, size=(vocabulary_size, embedding_vec_size)))
            # input keys
            # TODO: Keys in different slots should be unique.
            input_keys = np.ones(shape=(batch_size, slot_num, max_nnz), dtype=np.int64) * -1
            each_slot = vocabulary_size // slot_num
            nnz_0_num = 0
            for batch_id in range(batch_size):
                for slot_id in range(slot_num):
                    nnz = np.random.randint(low=nnz_0_num, high=max_nnz+1, size=1)[0] # how many keys in this slot
                    if nnz == 0:
                        nnz_0_num = 1
                    if (embedding_type == 'distributed'):
                        keys = np.random.randint(low=slot_id * each_slot, high=(slot_id + 1) * each_slot, size=nnz)
                    elif (embedding_type == "localized"):
                        keys = []
                        while len(keys) < nnz:
                            key = np.random.randint(low=slot_id * each_slot, high=(slot_id + 1) * each_slot, size=1)
                            if key % slot_num == slot_id:
                                keys.append(key)

                    input_keys[batch_id, slot_id, 0:nnz] = keys

            # hugectr ops
            hugectr_tf_ops.init(visiable_gpus=gpus, key_type='int64', value_type='float', 
                                batch_size=batch_size, batch_size_eval=len(gpus))
            embedding_name = hugectr_tf_ops.create_embedding(init_value=init_value, opt_hparams=[0.1, 0.9, 0.99, 1e-5],
                                        name_='hugectr_embedding', max_vocabulary_size_per_gpu= (vocabulary_size // len(gpus))* 2 + 1,
                                        slot_num=slot_num, embedding_vec_size=embedding_vec_size,
                                        max_feature_num=slot_num*max_nnz, embedding_type=embedding_type, 
                                        max_nnz=max_nnz, update_type='Global')

            # use CreateDataset to do preprocessing
            dataset_utils = CreateDataset(dataset_names=None,
                                          feature_desc=None,
                                          batch_size=batch_size,
                                          n_epochs=1,
                                          slot_num=slot_num,
                                          max_nnz=max_nnz,
                                          convert_to_csr=None,
                                          gpu_count=len(gpus),
                                          embedding_type=embedding_type,
                                          get_row_indices=None)
            
            if ("distributed" == embedding_type):
                row_offsets, value_tensor, nnz_array = dataset_utils._distribute_keys_for_distributed(input_keys)
            elif ("localized" == embedding_type):
                row_offsets, value_tensor, nnz_array = dataset_utils._distribute_keys_for_localized(input_keys)
            else:
                raise RuntimeError("Not supported embedding_type %s" %embedding_type)

            bp_trigger = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)

            hugectr_forward = hugectr_tf_ops.fprop_v3(embedding_name=embedding_name, row_offsets=row_offsets,
                                                    value_tensors=value_tensor, nnz_array=nnz_array,
                                                    bp_trigger=bp_trigger, is_training=True,
                                                    output_shape=[batch_size, slot_num, max_nnz])

            # print("hugectr_results=\n", hugectr_forward)

            # tf ops
            reshape_input_keys = np.reshape(input_keys, [-1, max_nnz])
            tf_indices = tf.where(reshape_input_keys != -1)
            tf_values = tf.gather_nd(reshape_input_keys, tf_indices)
            sparse_tensor = tf.sparse.SparseTensor(tf_indices, tf_values, reshape_input_keys.shape)

            # FIXME: if there are too more nnz=0 slots, tf.nn.embedding_lookup_sparse may get wrong results?
            tf_embedding_layer = OriginalEmbedding(vocabulary_size=vocabulary_size,
                                                   embedding_vec_size=embedding_vec_size,
                                                   initializer=init_value,
                                                   combiner='sum',
                                                   gpus=gpus)

            tf_forward = tf_embedding_layer(sparse_tensor, output_shape=[batch_size, slot_num, embedding_vec_size])
            # print("tf_results=\n", tf_forward)

            # compare first forward result
            try:
                tf.debugging.assert_near(hugectr_forward, tf_forward)
            except tf.errors.InvalidArgumentError as error:
                raise error
            
            print("[INFO]: The results from HugeCTR and tf in the first forward propagation are the same.")

        # backward
        hugectr_grads = tape.gradient(hugectr_forward, bp_trigger)

        tf_opt = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.99, epsilon=1e-5)
        tf_grads = tape.gradient(tf_forward, tf_embedding_layer.trainable_weights)
        tf_opt.apply_gradients(zip(tf_grads, tf_embedding_layer.trainable_weights))

        # compare second forward result
        hugectr_forward_2 = hugectr_tf_ops.fprop_v3(embedding_name=embedding_name, row_offsets=row_offsets,
                                                    value_tensors=value_tensor, nnz_array=nnz_array,
                                                    bp_trigger=bp_trigger, is_training=True,
                                                    output_shape=[batch_size, slot_num, max_nnz])
                                                
        tf_forward_2 = tf_embedding_layer(sparse_tensor, output_shape=[batch_size, slot_num, embedding_vec_size])

        # print("hugectr 2:\n", hugectr_forward_2)
        # print("tf 2:\n", tf_forward_2)
        try:
            tf.debugging.assert_near(hugectr_forward_2, tf_forward_2, rtol=1e-4, atol=1e-5)
        except tf.errors.InvalidArgumentError as error:
            raise error

        print("[INFO]: The results from HugeCTR and tf in the second forward propagation are the same.")
        hugectr_tf_ops.reset()


    def _fprop_v4_VS_tf():
        print("[INFO]: Testing fprop_v4 vs tf...")
        if vocabulary_size < slot_num:
            raise RuntimeError("vocabulary_size must > slot.")
        with tf.GradientTape(persistent=True) as tape:
            # initial embedding table
            init_value = np.float32(np.random.normal(loc=0, scale=1, size=(vocabulary_size, embedding_vec_size)))
            # input keys
            # TODO: Keys in different slots should be unique.
            input_keys = np.ones(shape=(batch_size, slot_num, max_nnz), dtype=np.int64) * -1
            each_slot = vocabulary_size // slot_num
            nnz_0_num = 0
            for batch_id in range(batch_size):
                for slot_id in range(slot_num):
                    nnz = np.random.randint(low=nnz_0_num, high=max_nnz+1, size=1)[0] # how many keys in this slot
                    if nnz == 0:
                        nnz_0_num = 1
                    if (embedding_type == 'distributed'):
                        keys = np.random.randint(low=slot_id * each_slot, high=(slot_id + 1) * each_slot, size=nnz)
                    elif (embedding_type == "localized"):
                        # TODO: key should belong to that slot.
                        keys = []
                        while len(keys) < nnz:
                            key = np.random.randint(low=slot_id * each_slot, high=(slot_id + 1) * each_slot, size=1)
                            if key % slot_num == slot_id:
                                keys.append(key)

                    input_keys[batch_id, slot_id, 0:nnz] = keys

            # hugectr ops
            hugectr_tf_ops.init(visiable_gpus=gpus, key_type='int64', value_type='float', 
                                batch_size=batch_size, batch_size_eval=len(gpus))
            embedding_name = hugectr_tf_ops.create_embedding(init_value=init_value, opt_hparams=[0.1, 0.9, 0.99, 1e-5],
                                        name_='hugectr_embedding', max_vocabulary_size_per_gpu= (vocabulary_size // len(gpus))* 2 + 1,
                                        slot_num=slot_num, embedding_vec_size=embedding_vec_size,
                                        max_feature_num=slot_num*max_nnz, embedding_type=embedding_type, 
                                        max_nnz=max_nnz, update_type='Global')
            reshape_input_keys = np.reshape(input_keys, [-1, max_nnz])
            indices = tf.where(reshape_input_keys != -1)
            values = tf.gather_nd(reshape_input_keys, indices)
            row_indices = tf.transpose(indices, perm=[1, 0])[0]

            bp_trigger = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)

            hugectr_forward = hugectr_tf_ops.fprop_v4(embedding_name=embedding_name, row_indices=row_indices, 
                                                    values=values, bp_trigger=bp_trigger, is_training=True,
                                                    output_shape=[batch_size, slot_num, max_nnz])
            # print("hugectr_results=\n", hugectr_forward)

            # tf ops
            reshape_input_keys = np.reshape(input_keys, [-1, max_nnz])
            tf_indices = tf.where(reshape_input_keys != -1)
            tf_values = tf.gather_nd(reshape_input_keys, tf_indices)
            sparse_tensor = tf.sparse.SparseTensor(tf_indices, tf_values, reshape_input_keys.shape)

            # FIXME: if there are too more nnz=0 slots, tf.nn.embedding_lookup_sparse may get wrong results?
            tf_embedding_layer = OriginalEmbedding(vocabulary_size=vocabulary_size,
                                                   embedding_vec_size=embedding_vec_size,
                                                   initializer=init_value,
                                                   combiner='sum',
                                                   gpus=gpus)

            tf_forward = tf_embedding_layer(sparse_tensor, output_shape=[batch_size, slot_num, embedding_vec_size])
            # print("tf_results=\n", tf_forward)

            # compare first forward result
            try:
                tf.debugging.assert_near(hugectr_forward, tf_forward)
            except tf.errors.InvalidArgumentError as error:
                raise error
            
            print("[INFO]: The results from HugeCTR and tf in the first forward propagation are the same.")

        # backward
        hugectr_grads = tape.gradient(hugectr_forward, bp_trigger)

        tf_opt = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.99, epsilon=1e-5)
        tf_grads = tape.gradient(tf_forward, tf_embedding_layer.trainable_weights)
        tf_opt.apply_gradients(zip(tf_grads, tf_embedding_layer.trainable_weights))

        # compare second forward result
        hugectr_forward_2 = hugectr_tf_ops.fprop_v4(embedding_name=embedding_name, row_indices=row_indices, 
                                                    values=values, bp_trigger=bp_trigger, is_training=True,
                                                    output_shape=[batch_size, slot_num, max_nnz])
                                                
        tf_forward_2 = tf_embedding_layer(sparse_tensor, output_shape=[batch_size, slot_num, embedding_vec_size])

        # print("hugectr 2:\n", hugectr_forward_2)
        # print("tf 2:\n", tf_forward_2)
        try:
            tf.debugging.assert_near(hugectr_forward_2, tf_forward_2, rtol=1e-4, atol=1e-5)
        except tf.errors.InvalidArgumentError as error:
            raise error

        print("[INFO]: The results from HugeCTR and tf in the second forward propagation are the same.")
        hugectr_tf_ops.reset()

    # _fprop_VS_tf()
    _fprop_v3_VS_tf()
    _fprop_v4_VS_tf()

if __name__ == "__main__":
    # Convert_to_csr_test(batch_size=65536, gpu_count=8, embedding_type='localized')
    Embedding_ops_test(vocabulary_size=1024, slot_num=26, max_nnz=5, embedding_vec_size=128, batch_size=1024, 
                    gpus=[0,1,2,3,4,5,6,7], embedding_type='localized')