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
This script is only used for demo of fprop_v3
This version will be deprecated in near future, please update to fprop or fprop_experimental.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.python.distribute.values import PerReplica
import hugectr_tf_ops

""" 1. Define DNN model with fprop_v3, whole DNN model should be split into two sub-models."""
# define sparse model which contains embedding layer(s)
class PluginSparseModel(tf.keras.models.Model):
    def __init__(self, 
                 gpus, 
                 batch_size, 
                 embedding_type,
                 vocabulary_size,
                 slot_num,
                 embedding_vec_size,
                 embedding_type,
                 opt_hparam,
                 update_type,
                 atomic_update,
                 max_feature_num,
                 max_nnz,
                 combiner,
                 gpu_count):
        super(PluginSparseModel, self).__init__()

        self.vocabulary_size_each_gpu = (vocabulary_size // gpu_count) + 1 
        self.slot_num = slot_num
        self.embedding_vec_size = embedding_vec_size
        self.embedding_type = embedding_type
        self.optimizer_type = optimizer
        self.opt_hparam = opt_hparam
        self.update_type = update_type
        self.atomic_update = atomic_update
        self.max_feature_num = max_feature_num
        self.max_nnz = max_nnz
        self.combiner = combiner
        self.gpu_count = gpu_count

        # Make use init() only be called once. It will create resource manager for embedding_plugin.
        hugectr_tf_ops.init(visiable_gpus=gpus, seed=123, key_type='int64', value_type='float',
                     batch_size=batch_size, batch_size_eval=len(gpus))

        # create one embedding layer, and its embedding_name will be unique if there are more than one embedding layer.
        self.embedding_name = hugectr_tf_ops.create_embedding(initializer, name_=name, embedding_type=self.embedding_type, 
                                             optimizer_type=self.optimizer_type, 
                                             max_vocabulary_size_per_gpu=self.vocabulary_size_each_gpu,
                                             opt_hparams=self.opt_hparam, update_type=self.update_type,
                                             atomic_update=self.atomic_update, slot_num=self.slot_num,
                                             max_nnz=self.max_nnz, max_feature_num=self.max_feature_num,
                                             embedding_vec_size=self.embedding_vec_size, 
                                             combiner=self.combiner)
                
    def build(self, _):
        # this tf.Variable is used for embedding plugin.
        self.bp_trigger = self.add_weight(name='bp_trigger', shape=(1,), dtype=tf.float32, trainable=True)

    @tf.function
    def call(self, row_offsets, value_tensors, nnz_array, training=True):
        # forward propagtion of embedding layer
        return hugectr_tf_ops.fprop_v3(embedding_name=self.embedding_name, row_offsets=row_offsets, 
                                        value_tensors=value_tensors, nnz_array=nnz_array, bp_trigger=self.bp_trigger,
                                        is_training=training, 
                                        output_shape=[self.batch_size, self.slot_num, self.embedding_vec_size])

# define dense model which contains other parts of the DNN model
class DenseModel(tf.keras.models.Model):
    def __init__(self, num_layers):
        super(DenseModel, self).__init__()
        self.num_layers = num_layers

        self.dense_layers = []
        for _ in range(num_layers - 1):
            self.dense_layers.append(tf.keras.layers.Dense(units=1024, activation='relu'))

        self.out_layer = tf.keras.layers.Dense(units=1, activation='sigmoid', use_bias=True,
                                                kernel_initializer='glorot_normal', 
                                                bias_initializer='glorot_normal')

    @tf.function
    def call(self, inputs, training=True):
        hidden = tf.reshape(inputs, [tf.shape(inputs)[0], 26 * 32]) # [batchsize, slot_num * embedding_vec_size]

        for i in range(self.num_layers - 1):
            hidden = self.dense_layers[i](hidden)

        result = self.out_layer(hidden)
        return result


""" 2.Define training loop with the model mentioned above """
def main():
    # create MirroredStrategy with specified GPUs.
    strategy = tf.distribute.MirroredStrategy(devices=["/GPU:" + str(i) for i in range(gpu_count)])

    # create sparse model outside the scope of MirroredStrategy
    sparse_model = PluginSparseModel(...)
    sparse_opt = tf.keras.optimizers.SGD()

    # create dense model inside the scope of MirroredSrategy
    with strategy.scope():
        dense_model = DenseModel(...)
        dense_opt = tf.keras.optimizers.SGD()

    # define loss function for each replica
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    def _replica_loss(labels, logits):
        loss_value = loss_fn(labels, logits)
        return tf.nn.compute_average_loss(loss_value, global_batch_size=batch_size)

    # define dense model train step
    @tf.function
    def dense_train_step(dense_inputs, labels):
        with tf.GradientTape() as tape:
            # should watch inputs, in order to obtain gradients later
            tape.watch(dense_inputs)

            logits = dense_model(dense_inputs)
            replica_loss = _replica_loss(labels, logits)

        grads, input_grads = tape.gradient(replica_loss, [dense_model.trainable_weights, dense_inputs])
        dense_opt.apply_gradients(zip(grads, dense_model.trainable_weights))
        return replica_loss, input_grads

    # define whole model train step
    @tf.function
    def total_train_step(row_offsets, value_tensors, nnz_array, labels):
        with tf.GradientTape() as tape:
            # do embedding fprop
            embedding_results = sparse_model(row_offsets, value_tensors, nnz_array)

            # convert to PerReplica
            dense_inputs = tf.split(embedding_results, num_or_size_splits=gpu_count)
            dense_inputs = PerReplica(dense_inputs)
            labels = tf.expand_dims(labels, axis=1)
            labels = tf.split(labels, num_or_size_splits=gpu_count)
            labels = PerReplica(labels)

            replica_loss, input_grads = strategy.run(dense_train_step, args=(dense_inputs, labels))

            # gather all grads from dense replicas
            all_grads = tf.concat(input_grads.values, axis=0)

            # do embedding backward
            embedding_grads = tape.gradient(embedding_results, sparse_model.trainable_weights, output_gradients=all_grads)
            sparse_opt.apply_gradients(zip(embedding_grads, sparse_model.trainable_weights))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, replica_loss, axis=None)

    # create a tf.data.Dataset to read data
    dataset = ...

    # training loop
    for step, (row_offsets, value_tensors, nnz_array, labels) in enumerate(dataset):
        total_loss = total_train_step(row_offsets, value_tensors, nnz_array, labels)

        # you can save model, print loss or do sth. else.