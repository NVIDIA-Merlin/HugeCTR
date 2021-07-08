# 
# Copyright (c) 2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


try:
    # tensorflow 2.x
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    # tensorflow 1.x
    import tensorflow as tf

import numpy as np


def embedding_layer(input_keys, init_values, combiner=0):
    """
    input_keys: [batch, slot_num, max_nnz_per_slot]
    """
    vocabulary_size, embedding_vec_size = init_values.shape

    # map -1 to zeros embedding-feature
    zeros = np.zeros(shape=(1, embedding_vec_size), dtype=np.float32)
    init_values = np.concatenate((init_values, zeros), axis=0)
    
    embedding_table = tf.get_variable(name='embedding-table', shape=init_values.shape, 
                                      dtype=tf.float32, 
                                      initializer=tf.constant_initializer(value=init_values))

    embedding_feature = tf.nn.embedding_lookup(embedding_table, input_keys)
    if combiner == 0:
        embedding_feature = tf.reduce_sum(embedding_feature, axis=-2)
    elif combiner == 1:
        embedding_feature = tf.reduce_mean(embedding_feature, axis=-2)

    return embedding_feature


def slice_layer(x, offsets, lengths):
    y = []
    for i in zip(offsets, lengths):
        y.append(tf.slice(x, [0, i[0]], [-1, i[1]]))
    return y


def multicross_layer(x, w, b, layers):
    y = []
    for i in range(layers):
        v = tf.linalg.matvec(x if i == 0 else y[i - 1], tf.Variable(w[i]))
        v = tf.transpose(v)
        m = tf.multiply(x, v)
        m = tf.add(m, x if i == 0 else y[i - 1])
        m = tf.add(m, tf.Variable(b[i]))
        y.append(m)
    return y

def innerproduct_layer(x, w, b):
    return tf.matmul(x, tf.Variable(w)) + tf.Variable(b)


if __name__ == "__main__":

    batch_size = 2
    slot_num = 3
    max_nnz_per_slot = 4

    vocabulary_size = 10
    embedding_vec_size = 5

    keys = np.ones(shape=(batch_size, slot_num, max_nnz_per_slot), dtype=np.int64) * -1
    print(keys)

    keys = tf.convert_to_tensor(keys)

    init_values = np.reshape(np.array([i for i in range(0, vocabulary_size * embedding_vec_size)], dtype=np.float32),
                        newshape=(vocabulary_size, embedding_vec_size))
    
    embedding_feature = embedding_layer(keys, init_values)
    print(embedding_feature)
