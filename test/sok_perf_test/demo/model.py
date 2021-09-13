"""
 Copyright (c) 2021, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import tensorflow as tf
import sys, os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "../../../sparse_operation_kit/")))
import sparse_operation_kit as sok

class HashtableEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 max_vocabulary_size,
                 embedding_vec_size,
                 key_dtype=tf.int64,
                 value_dtype=tf.int64,
                 initializer='random_uniform',
                 serving_default_value=None 
                 ):
        super(HashtableEmbedding, self).__init__()
        self.max_vocabulary_size = max_vocabulary_size
        self.embedding_vec_size = embedding_vec_size
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype
        self.initializer = initializer
        self.serving_default_value = serving_default_value
        if (self.serving_default_value is not None
            and (not isinstance(self.serving_default_value, tf.Tensor)
            or not isinstance(self.serving_default_value, np.ndarray))):
                raise RuntimeError("serving_default_value must be None or tf.Tensor.")
        else:
            self.serving_default_value = tf.zeros(shape=[1, self.embedding_vec_size], dtype=tf.float32)

        self.minimum = -9223372036854775808
        self.maximum = 9223372036854775807

        self.default_value = tf.constant(self.minimum, dtype=self.value_dtype)

        if isinstance(self.initializer, str):
            self.initializer = tf.keras.initializers.get(self.initializer)
            initial_value = self.initializer(shape=[self.max_vocabulary_size, self.embedding_vec_size], dtype=tf.float32)
        elif isinstance(self.initializer, tf.keras.initializers.Initializer):
            initial_value = self.initializer(shape=[self.max_vocabulary_size, self.embedding_vec_size], dtype=tf.float32)
        elif isinstance(self.initializer, np.ndarray):
            initial_value = self.initializer
        else:
            raise RuntimeError("Not supported initializer.")

        self.hash_table = tf.lookup.experimental.DenseHashTable(
            key_dtype=self.key_dtype, value_dtype=self.value_dtype, default_value=self.default_value,
            empty_key=self.maximum, deleted_key=self.maximum - 1)
        self.counter = tf.Variable(initial_value=0, trainable=False, dtype=self.value_dtype, name="hashtable_counter")
        self.embedding_var = tf.Variable(initial_value=initial_value, dtype=tf.float32, name='embedding_variables')
        
        # used for inference, as the default embedding vector.
        self.default_embedding = tf.Variable(initial_value=tf.convert_to_tensor(self.serving_default_value, dtype=tf.float32),
                                                name='default_embedding_vector', trainable=False)

    def get_insert(self, flatten_ids, length):
        hash_ids = self.hash_table.lookup(flatten_ids)
        default_ids = tf.gather_nd(flatten_ids, tf.where(hash_ids == self.default_value))
        unique_default_ids, _ = tf.unique(default_ids)
        unique_default_ids_num = tf.size(unique_default_ids, out_type=self.value_dtype)
        if 0 != unique_default_ids_num:
            # TODO: check counter < max_vocabulary_size
            inserted_values = tf.range(start=self.counter, limit=self.counter + unique_default_ids_num, delta=1, dtype=self.value_dtype)
            self.counter.assign_add(unique_default_ids_num, read_value=False)
            self.hash_table.insert(unique_default_ids, inserted_values)
            hash_ids = self.hash_table.lookup(flatten_ids)

        return hash_ids

    def get(self, flatten_ids, length):
        hash_ids = self.hash_table.lookup(flatten_ids)
        hash_ids = tf.where(hash_ids == self.default_value, 
                            tf.constant(self.max_vocabulary_size, dtype=self.value_dtype),
                            hash_ids)
        return hash_ids

    @property
    def hashtable(self):
        return self.hash_table

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None], dtype=tf.int64), 
                                  tf.TensorSpec(dtype=tf.bool, shape=[])))
    def call(self, ids, training=True):
        flatten_ids = tf.reshape(ids, [-1])
        length = tf.size(flatten_ids)
        if training:
            hash_ids = self.get_insert(flatten_ids, length)
        else:
            hash_ids = self.get(flatten_ids, length)

        hash_ids = tf.reshape(hash_ids, tf.shape(ids))
        embedding = tf.nn.embedding_lookup([self.embedding_var, self.default_embedding], hash_ids)

        return embedding


class DemoModel(tf.keras.layers.Layer):
    def __init__(self,
                 max_vocabulary_size_per_gpu,
                 embedding_vec_size,
                 slot_num,
                 nnz_per_slot,
                 num_dense_layers,
                 use_sok = False,
                 **kwargs):
        super(DemoModel, self).__init__(**kwargs)

        self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        self.embedding_vec_size = embedding_vec_size
        self.slot_num = slot_num
        self.nnz_per_slot = nnz_per_slot
        self.num_dense_layers = num_dense_layers
        self.use_sok = use_sok

        if self.use_sok:
            self.embedding_layer = sok.All2AllDenseEmbedding(
                            max_vocabulary_size_per_gpu=self.max_vocabulary_size_per_gpu,
                            embedding_vec_size=self.embedding_vec_size,
                            slot_num=self.slot_num,
                            nnz_per_slot=self.nnz_per_slot)
        else:
            self.embedding_layer = HashtableEmbedding(
                            max_vocabulary_size=self.max_vocabulary_size_per_gpu,
                            embedding_vec_size=self.embedding_vec_size)

        self.dense_layers = list()
        for _ in range(self.num_dense_layers):
            self._layer = tf.keras.layers.Dense(units=1024, activation='relu')
            self.dense_layers.append(self._layer)

        self.out_layer = tf.keras.layers.Dense(units=1, activation=None)

    def call(self, inputs, training):
        embedding_vector = self.embedding_layer(inputs, training=training)

        embedding_vector = tf.reshape(embedding_vector,
                    shape=[-1, self.slot_num * self.nnz_per_slot * self.embedding_vec_size])

        hidden = embedding_vector
        for layer in self.dense_layers:
            hidden = layer(hidden)

        logit = self.out_layer(hidden)
        return logit