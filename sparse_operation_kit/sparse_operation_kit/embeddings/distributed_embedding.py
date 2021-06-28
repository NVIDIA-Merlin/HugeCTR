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

import tensorflow as tf

from sparse_operation_kit.kit_lib import create_var, create_embedding_sparse, plugin_sparse_fprop
from sparse_operation_kit.core.embedding_variable import EmbeddingVariable

class DistributedEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 combiner,
                 max_vocabulary_size_per_gpu,
                 embedding_vec_size,
                 slot_num,
                 max_nnz,
                 max_feature_num = 1,
                 **kwargs):
        super(DistributedEmbedding, self).__init__(**kwargs)

        self.combiner = combiner
        self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        self.embedding_vec_size = embedding_vec_size
        self.slot_num = slot_num
        self.max_nnz = max_nnz
        self.max_feature_num = max_feature_num

        self.var = EmbeddingVariable.CreateInstances(shape=[self.max_vocabulary_size_per_gpu, self.embedding_vec_size],
                                                     trainable=True)

        self.emb = create_embedding_sparse(self.var.values[0].emb_handle,
                                           input_dispatcher="all_gather_dispatcher",
                                           input_dispatcher_subsequent_ops=["csr_conversion_distributed"],
                                           embedding_executor="distributed",
                                           output_dispatcher="reduce_scatter_dispatcher",
                                           slot_num=self.slot_num, 
                                           max_nnz=self.max_nnz,
                                           max_feature_num=self.max_feature_num,
                                           combiner=self.combiner)

    @property
    def embedding_variable(self):
        return self.var

    def get_config(self):
        config = super(DistributedEmbedding, self).get_config()
        config.update({})
        return config

    def build(self, input_shape):
        pass

    @tf.function
    def call(self, inputs, training=True):
        """
        inputs must be a SparseTensor, and its rank must be 2,
        which represents [row-indice, column-indice].
        """
        if not isinstance(inputs, tf.SparseTensor):
            raise TypeError("inputs must be SparseTensor")

        values = inputs.values
        row_indices = tf.transpose(inputs.indices, perm=[1, 0])[0]

        replica_ctx = tf.distribute.get_replica_context()

        # option 2, return grad for self.emb
        emb_vector = plugin_sparse_fprop(self.emb, 
                                         self.var,
                                         values, row_indices, 
                                         replica_ctx.replica_id_in_sync_group,
                                         training=training, vector_dtype=tf.float32, 
                                         unique_op_name="1") 
        return emb_vector

