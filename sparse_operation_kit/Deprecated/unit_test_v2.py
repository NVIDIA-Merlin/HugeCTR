"""
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

import tensorflow as tf
import numpy as np
import sys
sys.path.append("./python")
import hugectr_tf_ops_v2
sys.path.append("./performance_profile")
from read_data import CreateDataset
from tensorflow.python.distribute.values import PerReplica
from model import OriginalEmbedding
import argparse

def Embedding_op_test(vocabulary_size, slot_num, max_nnz, embedding_vec_size, batch_size,
                        gpus, embedding_type):
    """
    test forward propagation result with tf embedding layer.
    And do backward, then check forward propagation again.
    """
    def generate_embedding_init_value_and_inputs():
        # initial value for embedding table
        init_value = np.float32(np.random.normal(loc=0, scale=1, size=(vocabulary_size, embedding_vec_size)))
        # input keys
        input_keys = np.ones(shape=(batch_size, slot_num, max_nnz), dtype=np.int64) * -1
        vocab_in_each_slot = vocabulary_size // slot_num
        nnz_0_num = 0
        for batch_id in range(batch_size):
            for slot_id in range(slot_num):
                # how many keys in this slot.
                nnz = np.random.randint(low=nnz_0_num, high=max_nnz+1, size=1)[0]
                if nnz == 0:
                    nnz_0_num = 1
                if (embedding_type == 'distributed'):
                    keys = np.random.randint(low=slot_id * vocab_in_each_slot, high=(slot_id + 1) * vocab_in_each_slot, size=nnz)
                elif (embedding_type == 'localized'):
                    # keys should be belong to this slot
                    keys = []
                    while len(keys) < nnz:
                        key = np.random.randint(low=slot_id * vocab_in_each_slot, high=(slot_id + 1) * vocab_in_each_slot, size=1)
                        if key % slot_num == slot_id:
                            keys.append(key)

                input_keys[batch_id, slot_id, 0:nnz] = keys
        return init_value, input_keys

    def _v2_fprop_v1_test():
        print("[INFO]: Testing plugin_v2 fprop_experimental vs tf..")
        if vocabulary_size < slot_num:
            raise ValueError("vocabulary_size must > slot_num.")

        # generate initial values
        init_value, input_keys = generate_embedding_init_value_and_inputs()

        # -------------------------------- hugectr ops ------------------------------------ #
        class TestModel(tf.keras.models.Model):
            def __init__(self,
                        init_value, 
                        name_,
                        embedding_type,
                        optimizer_type,
                        max_vocabulary_size_per_gpu,
                        opt_hparams,
                        update_type,
                        atomic_update,
                        scaler,
                        slot_num,
                        max_nnz,
                        max_feature_num,
                        embedding_vec_size,
                        combiner):
                super(TestModel, self).__init__()

                self.input_buffer_reset = True if "distributed" == embedding_type else False

                self.embedding_name = hugectr_tf_ops_v2.create_embedding(init_value=init_value, 
                                    name_=name_, embedding_type=embedding_type, optimizer_type=optimizer_type, 
                                    max_vocabulary_size_per_gpu=max_vocabulary_size_per_gpu, opt_hparams=opt_hparams,
                                    update_type=update_type, atomic_update=atomic_update, scaler=scaler, slot_num=slot_num,
                                    max_nnz=max_nnz, max_feature_num=max_feature_num, embedding_vec_size=embedding_vec_size, 
                                    combiner=combiner)
            
            def build(self, _):
                self.bp_trigger = self.add_weight(name="bp_trigger",
                                                shape=(1,), dtype=tf.float32, trainable=True)

            @tf.function
            def call(self, row_offset, values, nnz, training=True):
                replica_ctx = tf.distribute.get_replica_context()
                result = hugectr_tf_ops_v2.fprop_experimental(self.embedding_name, replica_ctx.replica_id_in_sync_group,
                                                row_offset, values, nnz, self.bp_trigger, 
                                                input_buffer_reset=self.input_buffer_reset)
                return result

        hugectr_tf_ops_v2.init(visible_gpus=gpus, seed=0, key_type='int64', value_type='float',
                                batch_size=batch_size, batch_size_eval=len(gpus))

        strategy = tf.distribute.MirroredStrategy(devices=['/GPU:' + str(i) for i in gpus])
        with strategy.scope():
            hugectr_model = TestModel(init_value=init_value, name_='test_embedding', 
                        embedding_type=embedding_type, optimizer_type='Adam', 
                        max_vocabulary_size_per_gpu=(vocabulary_size // len(gpus)) * 2 + 1, 
                        opt_hparams=[0.1, 0.9, 0.99, 1e-5],
                        update_type='Global', atomic_update=True, scaler=1.0, slot_num=slot_num,
                        max_nnz=max_nnz, max_feature_num=slot_num * max_nnz, 
                        embedding_vec_size=embedding_vec_size, combiner='sum')
            opt = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.99, epsilon=1e-5)

        # preprocess inputs
        dataset_utils = CreateDataset(dataset_names=None, 
                                        feature_desc=None,
                                        batch_size=batch_size,
                                        n_epochs=None,
                                        slot_num=slot_num,
                                        max_nnz=max_nnz,
                                        convert_to_csr=None,
                                        gpu_count=len(gpus),
                                        embedding_type=embedding_type,
                                        get_row_indices=None)
        if "distributed" == embedding_type:
            row_offsets, value_tensors, nnz_array = dataset_utils._distribute_keys_for_distributed(input_keys)
        elif "localized" == embedding_type:
            row_offsets, value_tensors, nnz_array = dataset_utils._distribute_keys_for_localized(input_keys)
        else:
            raise ValueError("Not supported embedding_type %s." %embedding_type)

        # forward function
        @tf.function
        def hugectr_train_step(row_offset, values, nnz):
            with tf.GradientTape() as tape:
                forward_result = hugectr_model(row_offset, values, nnz)

            grads = tape.gradient(forward_result, hugectr_model.trainable_weights)
            opt.apply_gradients(zip(grads, hugectr_model.trainable_weights))
            return forward_result

        # -------------------------------- tf ops ------------------------------------------- #
        reshape_input_keys = np.reshape(input_keys, [-1, max_nnz])
        tf_indices = tf.where(reshape_input_keys != -1)
        tf_values = tf.gather_nd(reshape_input_keys, tf_indices)
        sparse_tensor = tf.sparse.SparseTensor(tf_indices, tf_values, reshape_input_keys.shape)

        tf_embedding_layer = OriginalEmbedding(vocabulary_size=vocabulary_size,
                                               embedding_vec_size=embedding_vec_size,
                                               initializer=init_value,
                                               combiner='sum',
                                               gpus=gpus)
        
        tf_opt = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.99, epsilon=1e-5)

        @tf.function
        def tf_train_step(sparse_tensor):
            with tf.GradientTape() as tape:
                tf_forward = tf_embedding_layer(sparse_tensor, output_shape=[batch_size, slot_num, embedding_vec_size])

            grads = tape.gradient(tf_forward, tf_embedding_layer.trainable_weights)
            tf_opt.apply_gradients(zip(grads, tf_embedding_layer.trainable_weights))
            return tf_forward

        # ------------------ comparison ---------------------------------------------------- #
        for iteration in range(2):
            replica_row_offsets = PerReplica(row_offsets)
            replica_values = PerReplica(value_tensors)
            replica_nnz = PerReplica(nnz_array)

            hugectr_forward = strategy.run(hugectr_train_step, args=(replica_row_offsets, replica_values, replica_nnz))
            if len(gpus) > 1:
                hugectr_forward = tf.concat(hugectr_forward.values, axis=0)

            tf_forward = tf_train_step(sparse_tensor)

            try:
                tf.debugging.assert_near(hugectr_forward, tf_forward, rtol=1e-4, atol=1e-5)
            except tf.errors.InvalidArgumentError as error:
                raise error
            else:
                print("[INFO]: The results from HugeCTR and tf in %d iteration are the same" %(iteration + 1))

        # --------------------- release resources -------------------------------------- #
        hugectr_tf_ops_v2.reset()

    def _v2_fprop_v2_test():
        print("[INFO]: Testing plugin_v2 fprop vs tf..")
        if vocabulary_size < slot_num:
            raise ValueError("vocabulary_size must > slot_num.")

        # generate initial values
        init_value, input_keys = generate_embedding_init_value_and_inputs()

        # -------------------------------- hugectr ops ------------------------------------ #
        class TestModel(tf.keras.models.Model):
            def __init__(self,
                        init_value, 
                        name_,
                        embedding_type,
                        optimizer_type,
                        max_vocabulary_size_per_gpu,
                        opt_hparams,
                        update_type,
                        atomic_update,
                        scaler,
                        slot_num,
                        max_nnz,
                        max_feature_num,
                        embedding_vec_size,
                        combiner):
                super(TestModel, self).__init__()

                self.input_buffer_reset = True if "distributed" == embedding_type else False

                self.embedding_name = hugectr_tf_ops_v2.create_embedding(init_value=init_value, 
                                    name_=name_, embedding_type=embedding_type, optimizer_type=optimizer_type, 
                                    max_vocabulary_size_per_gpu=max_vocabulary_size_per_gpu, opt_hparams=opt_hparams,
                                    update_type=update_type, atomic_update=atomic_update, scaler=scaler, slot_num=slot_num,
                                    max_nnz=max_nnz, max_feature_num=max_feature_num, embedding_vec_size=embedding_vec_size, 
                                    combiner=combiner)
            
            def build(self, _):
                self.bp_trigger = self.add_weight(name="bp_trigger",
                                                shape=(1,), dtype=tf.float32, trainable=True)

            @tf.function
            def call(self, to_each_replica, training=True):
                replica_ctx = tf.distribute.get_replica_context()
                result = hugectr_tf_ops_v2.fprop(self.embedding_name, replica_ctx.replica_id_in_sync_group,
                                                to_each_replica, self.bp_trigger, is_training=training)

                return result

            @property
            def get_embedding_name(self):
                return self.embedding_name

        hugectr_tf_ops_v2.init(visible_gpus=gpus, seed=0, key_type='int64', value_type='float',
                                batch_size=batch_size, batch_size_eval=len(gpus))

        strategy = tf.distribute.MirroredStrategy(devices=['/GPU:' + str(i) for i in gpus])
        with strategy.scope():
            hugectr_model = TestModel(init_value=init_value, name_='test_embedding', 
                        embedding_type=embedding_type, optimizer_type='Adam', 
                        max_vocabulary_size_per_gpu=(vocabulary_size // len(gpus)) * 2 + 1, 
                        opt_hparams=[0.1, 0.9, 0.99, 1e-5],
                        update_type='Global', atomic_update=True, scaler=1.0, slot_num=slot_num,
                        max_nnz=max_nnz, max_feature_num=slot_num * max_nnz, 
                        embedding_vec_size=embedding_vec_size, combiner='sum')
            opt = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.99, epsilon=1e-5)

        # preprocess inputs
        reshape_input_keys = np.reshape(input_keys, [-1, max_nnz])
        indices = tf.where(reshape_input_keys != -1)
        values = tf.gather_nd(reshape_input_keys, indices)
        row_indices = tf.transpose(indices, perm=[1, 0])[0]

        # forward function
        @tf.function
        def hugectr_train_step(to_each_replica):
            with tf.GradientTape() as tape:
                forward_result = hugectr_model(to_each_replica)

            grads = tape.gradient(forward_result, hugectr_model.trainable_weights)
            opt.apply_gradients(zip(grads, hugectr_model.trainable_weights))
            return forward_result

        # -------------------------------- tf ops ------------------------------------------- #
        reshape_input_keys = np.reshape(input_keys, [-1, max_nnz])
        tf_indices = tf.where(reshape_input_keys != -1)
        tf_values = tf.gather_nd(reshape_input_keys, tf_indices)
        sparse_tensor = tf.sparse.SparseTensor(tf_indices, tf_values, reshape_input_keys.shape)

        tf_embedding_layer = OriginalEmbedding(vocabulary_size=vocabulary_size,
                                               embedding_vec_size=embedding_vec_size,
                                               initializer=init_value,
                                               combiner='sum',
                                               gpus=gpus)
        
        tf_opt = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.99, epsilon=1e-5)

        @tf.function
        def tf_train_step(sparse_tensor):
            with tf.GradientTape() as tape:
                tf_forward = tf_embedding_layer(sparse_tensor, output_shape=[batch_size, slot_num, embedding_vec_size])

            grads = tape.gradient(tf_forward, tf_embedding_layer.trainable_weights)
            tf_opt.apply_gradients(zip(grads, tf_embedding_layer.trainable_weights))
            return tf_forward

        # ------------------ comparison ---------------------------------------------------- #
        for iteration in range(2):
            to_each_replicas = hugectr_tf_ops_v2.broadcast_then_convert_to_csr(
                        hugectr_model.get_embedding_name,
                        row_indices, values, T = [tf.int32] * len(gpus))
            to_each_replicas = PerReplica(to_each_replicas)

            hugectr_forward = strategy.run(hugectr_train_step, args=(to_each_replicas,))
            if len(gpus) > 1:
                hugectr_forward = tf.concat(hugectr_forward.values, axis=0)

            tf_forward = tf_train_step(sparse_tensor)

            try:
                tf.debugging.assert_near(hugectr_forward, tf_forward, rtol=1e-4, atol=1e-5)
            except tf.errors.InvalidArgumentError as error:
                raise error
            else:
                print("[INFO]: The results from HugeCTR and tf in %d iteration are the same" %(iteration + 1))

        # --------------------- release resources -------------------------------------- #
        hugectr_tf_ops_v2.reset()  

    # do testing
    _v2_fprop_v1_test()
    _v2_fprop_v2_test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embedding plugin unit test v2')
    parser.add_argument("--fast_testing", type=int, help='whether to do the unit testing fastly?',
                        required=False, default=0, choices=[0, 1])
    args = parser.parse_args()

    if (args.fast_testing == 1):
        Embedding_op_test(vocabulary_size=1024, slot_num=26, max_nnz=5, embedding_vec_size=128, batch_size=1024,
                            gpus=[i for i in range(8)], embedding_type='localized')
    else:
        vocabulary_size = int(1e5)
        for slot_num in [16, 32]:
            for max_nnz in [1, 8, 16]:
                for embedding_vec_size in [16, 128]:
                    for batch_size in [16384, 65536]:
                        for gpus in [[i for i in range(gpu_count)] for gpu_count in [1, 2, 4, 8]]:
                            for embedding_type in ['distributed', 'localized']:
                                print(("[INFO]: vocabulary_size = %d, slot_num = %d, max_nnz = %d, embedding_vec_size = %d, " +\
                                        "batch_size = %d, gpu_count = %d, embedding_type = %s") 
                                        %(vocabulary_size, slot_num, max_nnz, embedding_vec_size, 
                                            batch_size, len(gpus), embedding_type))
                                Embedding_op_test(vocabulary_size, slot_num, max_nnz, embedding_vec_size, batch_size, 
                                                    gpus, embedding_type)