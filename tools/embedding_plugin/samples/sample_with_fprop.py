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
This script is only used for demo of fprop
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.python.distribute.values import PerReplica
import hugectr_tf_ops_v2 


""" 1. Define DNN model with fprop """
class PluginSparseModel(tf.keras.models.Model):
    def __init__(self,
                batch_size,
                gpus,
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
                combiner,
                num_dense_layers,
                input_buffer_reset=False):
        super(PluginSparseModelV2, self).__init__()

        self.num_dense_layers = num_dense_layers
        self.input_buffer_reset = input_buffer_reset

        self.batch_size = batch_size
        self.slot_num = slot_num
        self.embedding_vec_size = embedding_vec_size
        self.gpus = gpus

        # Make use init() only be called once. It will create resource manager for embedding_plugin.
        hugectr_tf_ops_v2.init(visible_gpus=gpus, seed=0, key_type='int64', value_type='float',
                                batch_size=batch_size, batch_size_eval=len(gpus))

        # create one embedding layer, and its embedding_name will be unique if there are more than one embedding layer.
        self.embedding_name = hugectr_tf_ops_v2.create_embedding(init_value=init_value, 
                                name_=name_, embedding_type=embedding_type, optimizer_type=optimizer_type, 
                                max_vocabulary_size_per_gpu=max_vocabulary_size_per_gpu, opt_hparams=opt_hparams,
                                update_type=update_type, atomic_update=atomic_update, scaler=scaler, slot_num=slot_num,
                                max_nnz=max_nnz, max_feature_num=max_feature_num, embedding_vec_size=embedding_vec_size, 
                                combiner=combiner)

        # define other parts of this DNN model
        self.dense_layers = []
        for _ in range(self.num_dense_layers - 1):
            self.dense_layers.append(tf.keras.layers.Dense(units=1024, activation='relu'))

        self.out_layer = tf.keras.layers.Dense(units=1, activation='sigmoid', use_bias=True,
                                                kernel_initializer='glorot_normal', 
                                                bias_initializer='glorot_normal')

    def build(self, _):
        # this tf.Variable is used for embedding plugin.
        self.bp_trigger = self.add_weight(name='bp_trigger', shape=(1,), dtype=tf.float32, trainable=True)

    @tf.function
    def call(self, each_replica, training=True):
        # used to decide the replica_id of this call.
        replica_ctx = tf.distribute.get_replica_context()

        # forward propagation with fprop, its inputs are COO components, but don't pass COO here. 
        embedding_forward = hugectr_tf_ops_v2.fprop(self.embedding_name, replica_ctx.replica_id_in_sync_group,
                                            each_replica, self.bp_trigger, is_training=training)
            
        embedding_forward, dense_ctx = nvtx_tf.ops.start(embedding_forward, message='dense_fprop',
                                                         grad_message='dense_bprop')

        # forward propgation for other parts in this DNN model.
        hidden = tf.reshape(embedding_forward, [self.batch_size // len(self.gpus), self.slot_num * self.embedding_vec_size])

        for i in range(self.num_dense_layers - 1):
            hidden = self.dense_layers[i](hidden)
        
        logit = self.out_layer(hidden)

        logit = nvtx_tf.ops.end(logit, dense_ctx)

        return logit

    @property
    def get_embedding_name(self):
        return self.embedding_name

""" 2.Define training loop with the model mentioned above """
def main():
    # create MirroredStrategy with specified GPUs.
    strategy = tf.distribute.MirroredStrategy(devices=["/GPU:" + str(i) for i in range(gpu_count)])

    # create model instance inner the scope of MirroredStrategy
    with strategy.scope():
        model = PluginSparseModel(...)

        # define optimizer for the variables in DNN model except embedding layer
        opt = tf.keras.optimizers.SGD()

    # define loss function for each replica
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    def _replica_loss(labels, logits):
        loss_value = loss_fn(labels, logits)
        return tf.nn.compute_average_loss(loss_value, global_batch_size=batch_size)

    # define train step for one iteration in a replica
    @tf.function
    def _train_step(each_replica, label):
        with tf.GradientTape() as tape:
            label = tf.expand_dims(label, axis=1)
            logit = model(each_replica)
            replica_loss = _replica_loss(label, logit)

        replica_grads = tape.gradient(replica_loss, model.trainable_weights)
        opt.apply_gradients(zip(replica_grads, model.trainable_weights))
        return replica_loss

    # create a tf.data.Dataset to read data
    dataset = ...

    # training loop
    for step, (row_indices, values, labels) in enumerate(dataset):
        # use this API to broadcast input data to each GPU
        to_each_replicas = hugectr_tf_ops_v2.broadcast_then_convert_to_csr(
                        model.get_embedding_name, row_indices, values, T = [tf.int32] * gpu_count)
        to_each_replicas = PerReplica(to_each_replicas)
        labels = tf.split(labels, num_or_size_splits=gpu_count)
        labels = PerReplica(labels)

        # each replica iteration
        replica_loss = strategy.run(_train_step, args=(to_each_replicas, labels))

        # loss reduction in all replicas
        total_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, replica_loss, axis=None)

        # you can save model, print loss or do sth. else.
