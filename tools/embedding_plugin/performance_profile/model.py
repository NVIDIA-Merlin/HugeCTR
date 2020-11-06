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

import sys
sys.path.append("../python")
import hugectr
import tensorflow as tf



class OriginalEmbedding(tf.keras.layers.Layer):
    def __init__(self, 
                 vocabulary_size,
                 embedding_vec_size,
                 initializer='uniform',
                 combiner="sum",
                 gpus=[0]):
        super(OriginalEmbedding, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_vec_size = embedding_vec_size 
        if isinstance(initializer, str):
            self.initializer = tf.keras.initializers.get(initializer)
        else:
            self.initializer = initializer
        if combiner not in ["sum", "mean"]:
            raise RuntimeError("combiner must be one of \{'sum', 'mean'\}.")
        self.combiner = combiner
        if (not isinstance(gpus, list)) and (not isinstance(gpus, tuple)):
            raise RuntimeError("gpus must be a list or tuple.")
        self.gpus = gpus

    def build(self, _):
        if isinstance(self.initializer, tf.keras.initializers.Initializer):
            if len(self.gpus) > 1:
                self.embeddings_params = list()
                mod_size = self.vocabulary_size % len(self.gpus)
                vocabulary_size_each_gpu = [(self.vocabulary_size // len(self.gpus)) + (1 if dev_id < mod_size else 0)
                                            for dev_id in range(len(self.gpus))]

                for i, gpu in enumerate(self.gpus):
                    with tf.device("/gpu:%d" %gpu):
                        params_i = self.add_weight(name="embedding_" + str(gpu), 
                                                   shape=(vocabulary_size_each_gpu[i], self.embedding_vec_size),
                                                   initializer=self.initializer)
                    self.embeddings_params.append(params_i)

            else:
                self.embeddings_params = self.add_weight(name='embeddings', 
                                                        shape=(self.vocabulary_size, self.embedding_vec_size),
                                                        initializer=self.initializer)
        else:
            self.embeddings_params = self.initializer

    @tf.function
    def call(self, keys, output_shape):
        """
        uncomment these for debugging.
        # print("OriginalEmbedding, on Line 110")
        # if not isinstance(keys, tf.sparse.SparseTensor):
        #     raise RuntimeError("keys should be tf.sparse.SparseTensor")
        # if keys.shape.rank != 2:
        #     raise RuntimeError("keys' rank should be 2, while represents [batch_size * slot_num, max_nnz].")
        """
        result = tf.nn.embedding_lookup_sparse(self.embeddings_params, keys, 
                                             sp_weights=None, combiner=self.combiner)
        return tf.reshape(result, output_shape)



class PluginEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 vocabulary_size,
                 slot_num,
                 embedding_vec_size,
                 gpu_count,
                 initializer=False,
                 name='plugin_embedding',
                 embedding_type='localized',
                 optimizer='Adam',
                 opt_hparam=[0.1, 0.9, 0.99, 1e-3],
                 update_type='Local',
                 atomic_update=True,
                 max_feature_num=int(1e3),
                 max_nnz=1,
                 combiner='sum',
                 ):
        super(PluginEmbedding, self).__init__()

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

        self.name_ = hugectr.create_embedding(initializer, name_=name, embedding_type=self.embedding_type, 
                                             optimizer_type=self.optimizer_type, 
                                             max_vocabulary_size_per_gpu=self.vocabulary_size_each_gpu,
                                             opt_hparams=self.opt_hparam, update_type=self.update_type,
                                             atomic_update=self.atomic_update, slot_num=self.slot_num,
                                             max_nnz=self.max_nnz, max_feature_num=self.max_feature_num,
                                             embedding_vec_size=self.embedding_vec_size, 
                                             combiner=self.combiner)

    def build(self, _):
        self.bp_trigger = self.add_weight(name="bp_trigger",
                                          shape=(1,), dtype=tf.float32, trainable=True)

    # @tf.function(input_signature=(tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    #                               tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    #                               tf.TensorSpec(shape=(None,), dtype=tf.int64)))
    @tf.function
    def call(self, row_offsets, value_tensors, nnz_array, output_shape, training=False):
        return hugectr.fprop_v3(embedding_name=self.name_, row_offsets=row_offsets, value_tensors=value_tensors, 
                                nnz_array=nnz_array, bp_trigger=self.bp_trigger, is_training=training,
                                output_shape=output_shape)


class Multiply(tf.keras.layers.Layer):
    def __init__(self, out_units):
        super(Multiply, self).__init__()
        self.out_units = out_units

    def build(self, input_shape):
        self.w = self.add_weight(name='weight_vector', shape=(input_shape[1], self.out_units),
                                 initializer='glorot_uniform', trainable=True)
    
    def call(self, inputs):
        return inputs * self.w


class DeepFM_OriginalEmbedding(tf.keras.models.Model):
    def __init__(self, 
                 vocabulary_size, 
                 embedding_vec_size,
                 which_embedding,
                 dropout_rate, # list of float
                 deep_layers, # list of int
                 initializer,
                 gpus,
                 batch_size,
                 batch_size_eval,
                 embedding_type = 'localized',
                 slot_num=1,
                 seed=123):
        super(DeepFM_OriginalEmbedding, self).__init__()
        tf.keras.backend.clear_session()
        tf.compat.v1.set_random_seed(seed)

        self.vocabulary_size = vocabulary_size
        self.embedding_vec_size = embedding_vec_size
        self.which_embedding = which_embedding
        self.dropout_rate = dropout_rate
        self.deep_layers = deep_layers
        self.gpus = gpus
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval 
        self.slot_num = slot_num
        self.embedding_type = embedding_type

        self.original_embedding_layer = OriginalEmbedding(vocabulary_size=vocabulary_size, 
                                            embedding_vec_size=embedding_vec_size + 1, 
                                            initializer=initializer, gpus=gpus)
        self.deep_dense = []
        for i, deep_units in enumerate(self.deep_layers):
            self.deep_dense.append(tf.keras.layers.Dense(units=deep_units, activation=None, use_bias=True,
                                                         kernel_initializer='glorot_normal', 
                                                         bias_initializer='glorot_normal'))
            self.deep_dense.append(tf.keras.layers.Dropout(dropout_rate[i]))
        self.deep_dense.append(tf.keras.layers.Dense(units=1, activation=None, use_bias=True,
                                                     kernel_initializer='glorot_normal',
                                                     bias_initializer=tf.constant_initializer(0.01)))
        self.add_layer = tf.keras.layers.Add()
        self.y_act = tf.keras.layers.Activation(activation='sigmoid')

        self.dense_multi = Multiply(1)
        self.dense_embedding = Multiply(self.embedding_vec_size)

        self.concat_1 = tf.keras.layers.Concatenate()
        self.concat_2 = tf.keras.layers.Concatenate()

    @tf.function
    def call(self, dense_feature, sparse_feature, training=True):
        """
        forward propagation.
        #arguments:
            dense_feature: [batch_size, dense_dim]
            sparse_feature: for OriginalEmbedding, it is a SparseTensor, and the dense shape is [batch_size * slot_num, max_nnz];
                            for PluginEmbedding, it is a list of [row_offsets, value_tensors, nnz_array]. 
        """
        with tf.name_scope("embedding_and_slice"):
            dense_0 = tf.cast(tf.expand_dims(dense_feature, 2), dtype=tf.float32) # [batchsize, dense_dim, 1]
            dense_mul = self.dense_multi(dense_0) # [batchsize, dense_dim, 1]
            dense_emb = self.dense_embedding(dense_0) # [batchsize, dense_dim, embedding_vec_size]
            dense_mul = tf.reshape(dense_mul, [dense_mul.shape[0], -1]) # [batchsize, dense_dim * 1]
            dense_emb = tf.reshape(dense_emb, [dense_emb.shape[0], -1]) # [batchsize, dense_dim * embedding_vec_size]

            sparse = self.original_embedding_layer(sparse_feature, output_shape=[-1, self.slot_num, self.embedding_vec_size + 1])

            sparse_1 = tf.slice(sparse, [0, 0, self.embedding_vec_size], [-1, self.slot_num, 1]) #[batchsize, slot_num, 1]
            sparse_1 = tf.squeeze(sparse_1, 2) # [batchsize, slot_num]

            sparse_emb = tf.slice(sparse, [0, 0, 0], [-1, self.slot_num, self.embedding_vec_size]) #[batchsize, slot_num, embedding_vec_size]
            sparse_emb = tf.reshape(sparse_emb, [-1, self.slot_num * self.embedding_vec_size]) #[batchsize, slot_num * embedding_vec_size]
        
        with tf.name_scope("FM"):
            with tf.name_scope("first_order"):
                first = self.concat_1([dense_mul, sparse_1]) # [batchsize, dense_dim + slot_num]
                first_out = tf.reduce_sum(first, axis=-1, keepdims=True) # [batchsize, 1]
                
            with tf.name_scope("second_order"):
                hidden = self.concat_2([dense_emb, sparse_emb]) # [batchsize, (dense_dim + slot_num) * embedding_vec_size]
                second = tf.reshape(hidden, [-1, dense_feature.shape[1] + self.slot_num, self.embedding_vec_size])
                square_sum = tf.math.square(tf.math.reduce_sum(second, axis=1, keepdims=True)) # [batchsize, 1, embedding_vec_size]
                sum_square = tf.math.reduce_sum(tf.math.square(second), axis=1, keepdims=True) # [batchsize, 1, embedding_vec_size]
                
                second_out = 0.5 * (sum_square - square_sum) # [batchsize, 1, embedding_vec_size]
                second_out = tf.math.reduce_sum(second_out, axis=-1, keepdims=False) # [batchsize, 1]
                
        with tf.name_scope("Deep"):
            for i, layer in enumerate(self.deep_dense):
                if i % 2 == 0: # dense
                    hidden = layer(hidden)
                else: # dropout
                    hidden = layer(hidden, training)

        y = self.add_layer([hidden, first_out, second_out])
        y = self.y_act(y) # [batchsize, 1]

        return y
            

class DeepFM_PluginEmbedding(tf.keras.models.Model):
    def __init__(self, 
                 vocabulary_size, 
                 embedding_vec_size,
                 which_embedding,
                 dropout_rate, # list of float
                 deep_layers, # list of int
                 initializer,
                 gpus,
                 batch_size,
                 batch_size_eval,
                 embedding_type = 'localized',
                 slot_num=1,
                 seed=123):
        super(DeepFM_PluginEmbedding, self).__init__()
        tf.keras.backend.clear_session()
        tf.compat.v1.set_random_seed(seed)

        self.vocabulary_size = vocabulary_size
        self.embedding_vec_size = embedding_vec_size
        self.which_embedding = which_embedding
        self.dropout_rate = dropout_rate
        self.deep_layers = deep_layers
        self.gpus = gpus
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval 
        self.slot_num = slot_num
        self.embedding_type = embedding_type

        if isinstance(initializer, str):
            initializer = False
        hugectr.init(visiable_gpus=gpus, seed=seed, key_type='int64', value_type='float', 
                        batch_size=batch_size, batch_size_eval=batch_size_eval)
        self.plugin_embedding_layer = PluginEmbedding(vocabulary_size=vocabulary_size, slot_num=slot_num, 
                                            embedding_vec_size=embedding_vec_size + 1, 
                                            embedding_type=embedding_type,
                                            gpu_count=len(gpus), initializer=initializer)
        self.deep_dense = []
        for i, deep_units in enumerate(self.deep_layers):
            self.deep_dense.append(tf.keras.layers.Dense(units=deep_units, activation=None, use_bias=True,
                                                         kernel_initializer='glorot_normal', 
                                                         bias_initializer='glorot_normal'))
            self.deep_dense.append(tf.keras.layers.Dropout(dropout_rate[i]))
        self.deep_dense.append(tf.keras.layers.Dense(units=1, activation=None, use_bias=True,
                                                     kernel_initializer='glorot_normal',
                                                     bias_initializer=tf.constant_initializer(0.01)))
        self.add_layer = tf.keras.layers.Add()
        self.y_act = tf.keras.layers.Activation(activation='sigmoid')

        self.dense_multi = Multiply(1)
        self.dense_embedding = Multiply(self.embedding_vec_size)

        self.concat_1 = tf.keras.layers.Concatenate()
        self.concat_2 = tf.keras.layers.Concatenate()

    @tf.function
    def call(self, dense_feature, sparse_feature, training=True):
        """
        forward propagation.
        #arguments:
            dense_feature: [batch_size, dense_dim]
            sparse_feature: for OriginalEmbedding, it is a SparseTensor, and the dense shape is [batch_size * slot_num, max_nnz];
                            for PluginEmbedding, it is a list of [row_offsets, value_tensors, nnz_array]. 
        """
        with tf.name_scope("embedding_and_slice"):
            dense_0 = tf.cast(tf.expand_dims(dense_feature, 2), dtype=tf.float32) # [batchsize, dense_dim, 1]
            dense_mul = self.dense_multi(dense_0) # [batchsize, dense_dim, 1]
            dense_emb = self.dense_embedding(dense_0) # [batchsize, dense_dim, embedding_vec_size]
            dense_mul = tf.reshape(dense_mul, [dense_mul.shape[0], -1]) # [batchsize, dense_dim * 1]
            dense_emb = tf.reshape(dense_emb, [dense_emb.shape[0], -1]) # [batchsize, dense_dim * embedding_vec_size]

            sparse = self.plugin_embedding_layer(sparse_feature[0], sparse_feature[1], sparse_feature[2],
                                                output_shape=[self.batch_size, self.slot_num, self.embedding_vec_size + 1],
                                                training=training) # [batch_size, self.slot_num, self.embedding_vec_size + 1]

            sparse_1 = tf.slice(sparse, [0, 0, self.embedding_vec_size], [-1, self.slot_num, 1]) #[batchsize, slot_num, 1]
            sparse_1 = tf.squeeze(sparse_1, 2) # [batchsize, slot_num]

            sparse_emb = tf.slice(sparse, [0, 0, 0], [-1, self.slot_num, self.embedding_vec_size]) #[batchsize, slot_num, embedding_vec_size]
            sparse_emb = tf.reshape(sparse_emb, [-1, self.slot_num * self.embedding_vec_size]) #[batchsize, slot_num * embedding_vec_size]
        
        with tf.name_scope("FM"):
            with tf.name_scope("first_order"):
                first = self.concat_1([dense_mul, sparse_1]) # [batchsize, dense_dim + slot_num]
                first_out = tf.reduce_sum(first, axis=-1, keepdims=True) # [batchsize, 1]
                
            with tf.name_scope("second_order"):
                hidden = self.concat_2([dense_emb, sparse_emb]) # [batchsize, (dense_dim + slot_num) * embedding_vec_size]
                second = tf.reshape(hidden, [-1, dense_feature.shape[1] + self.slot_num, self.embedding_vec_size])
                square_sum = tf.math.square(tf.math.reduce_sum(second, axis=1, keepdims=True)) # [batchsize, 1, embedding_vec_size]
                sum_square = tf.math.reduce_sum(tf.math.square(second), axis=1, keepdims=True) # [batchsize, 1, embedding_vec_size]
                
                second_out = 0.5 * (sum_square - square_sum) # [batchsize, 1, embedding_vec_size]
                second_out = tf.math.reduce_sum(second_out, axis=-1, keepdims=False) # [batchsize, 1]
                
        with tf.name_scope("Deep"):
            for i, layer in enumerate(self.deep_dense):
                if i % 2 == 0: # dense
                    hidden = layer(hidden)
                else: # dropout
                    hidden = layer(hidden, training)

        y = self.add_layer([hidden, first_out, second_out])
        y = self.y_act(y) # [batchsize, 1]

        return y



if __name__ == "__main__":
    import numpy as np
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

    # --------------- test OriginalEmbedding ------------------------------ #
    # reshape_keys = np.reshape(keys, newshape=[-1, keys.shape[-1]])
    # sparse_indices = tf.where(reshape_keys != -1) #[N, ndims]
    # values = tf.gather_nd(reshape_keys, sparse_indices) # [N]

    # init_value = np.float32([i for i in range(1, 10 * 4 + 1)]).reshape(10, 4)
    # embedding_layer = OriginalEmbedding(vocabulary_size=10, embedding_vec_size=4, initializer="uniform", gpus=[0,1,3,4])
    
    # keys_input = tf.sparse.SparseTensor(indices=sparse_indices, values=values, dense_shape=reshape_keys.shape)
    # result = embedding_layer(keys_input)
    # print(result)
    # print(embedding_layer.trainable_weights)


    # ---------------- test PluginEmbedding ----------------------------- #
    # hugectr.init(visiable_gpus=[0,1,3,4], seed=123, key_type='int64', value_type='float', batch_size=4, batch_size_eval=4)

    # sparse_indices = tf.where(keys != -1) #[N, ndims]
    # values = tf.gather_nd(keys, sparse_indices) # [N]
    # row_offsets, value_tensors, nnz_array = hugectr.distribute_keys(sparse_indices, values, keys.shape,
    #                                 gpu_count = 4, embedding_type='localized', max_nnz=2)
                            
    # init_value = np.float32([i for i in range(1, 10 * 4 + 1)]).reshape(10, 4)
    # embedding_layer = PluginEmbedding(vocabulary_size=10, slot_num=3, embedding_vec_size=4, gpu_count=4, initializer=init_value)
    # result = embedding_layer(row_offsets, value_tensors, nnz_array, training=True)
    # print(result)
    # print(embedding_layer.trainable_weights)

    # ------------ test model ---------------------- #
    # model = DeepFM(vocabulary_size=10, embedding_vec_size=4, which_embedding='OriginalEmbedding',
    #                 dropout_rate=[0.5, 0.5], deep_layers=[512, 512], 
    #                 initializer='uniform', gpus=[0,1,3,4], batch_size=4, batch_size_eval=4,
    #                 slot_num=3)

    # reshape_keys = np.reshape(keys, newshape=[-1, keys.shape[-1]])
    # sparse_indices = tf.where(reshape_keys != -1) #[N, ndims]
    # values = tf.gather_nd(reshape_keys, sparse_indices) # [N]
    # sparse_feature = tf.sparse.SparseTensor(indices=sparse_indices, values=values, dense_shape=reshape_keys.shape)
    # result = model(np.ones(shape=(4, 13), dtype=np.float32), sparse_feature=sparse_feature, training=True)
    # print(result)
    # print(model.trainable_weights)

    model = DeepFM(vocabulary_size=10, embedding_vec_size=4, which_embedding='PluginEmbedding',
                    dropout_rate=[0.5, 0.5], deep_layers=[512, 512], 
                    initializer='uniform', gpus=[0,1,3,4], batch_size=4, batch_size_eval=4,
                    slot_num=3)

    indices = tf.where(keys != -1)
    values = tf.gather_nd(keys, indices)
    row_offsets, value_tensors, nnz_array = hugectr.distribute_keys(indices, values, keys.shape,
                                                gpu_count=4, embedding_type='localized', max_nnz=2)
    result = model(np.ones(shape=(4, 13), dtype=np.float32), [row_offsets, value_tensors, nnz_array], training=True)
    print("result = ", result)