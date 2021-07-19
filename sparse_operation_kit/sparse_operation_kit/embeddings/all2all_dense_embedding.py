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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sparse_operation_kit.core.embedding_variable import EmbeddingVariable
from sparse_operation_kit.kit_lib import create_embedding_dense, plugin_dense_fprop

import tensorflow as tf

class All2AllDenseEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 max_vocabulary_size_per_gpu,
                 embedding_vec_size, 
                 slot_num,
                 nnz_per_slot,
                 **kwargs):
        super(All2AllDenseEmbedding, self).__init__(**kwargs)

        self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        self.embedding_vec_size = embedding_vec_size
        self.slot_num = slot_num
        self.nnz_per_slot = nnz_per_slot

        self.var = EmbeddingVariable.CreateInstances(shape=[self.max_vocabulary_size_per_gpu, self.embedding_vec_size],
                                                     trainable=True)

        self.emb = create_embedding_dense(self.var.values[0].emb_handle,
                                          input_dispatcher="All2AllInput",
                                          embedding_lookuper="dense_gather",
                                          output_dispatcher="All2AllOutput",
                                          slot_num=self.slot_num,
                                          nnz_per_slot=self.nnz_per_slot)

    @property
    def embedding_variable(self):
        return self.var

    @tf.function
    def call(self, inputs, training=True):
        """
        Inputs must be a dense tensor.
        """
        replica_ctx = tf.distribute.get_replica_context()

        emb_vector = plugin_dense_fprop(self.emb,
                                        self.var,
                                        values=inputs,
                                        global_replica_id=replica_ctx.replica_id_in_sync_group,
                                        training=training, vector_dtype=tf.float32,
                                        unique_op_name="1")
        return emb_vector

