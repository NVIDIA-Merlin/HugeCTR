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
from hierarchical_parameter_server.core import lookup_ops


class LookupLayer(tf.keras.layers.Layer):
    """
    Abbreviated as ``hps.LookupLayer(*args, **kwargs)``.

    This is a wrapper class for HPS lookup layer.
    It can be used to create a dense embedding layer which will distribute
    keys based on `gpu_id = key % gpu_num` to each GPU.

    Parameters
    ----------
    model_name: string
            the name of the model that has embedding table(s)
    table_id: integer
            the index of the embedding table for the model specified by
            model_name
    emb_vec_size: integer
            the embedding vector size for the embedding table specified
            by model_name and table_id
    emb_vec_dtype: tensorflow.python.framework.dtypes.DType
            the data type of embedding vectors which must be tf.float32

    Examples
    --------
    .. code-block:: python

        lookup_layer = hps.LookupLayer(model_name = args.model_name,
                                        table_id = args.table_id,
                                        emb_vec_size = args.embed_vec_size,
                                        emb_vec_dtype = tf.float32)

        @tf.function
        def _infer_step(inputs):
            embedding_vector = lookup_layer(inputs)
            ...

        for i, (inputs, labels) in enumerate(dataset):
            _infer_step(inputs)
    """

    def __init__(self, model_name, table_id, emb_vec_size, emb_vec_dtype, **kwargs):
        super(LookupLayer, self).__init__(**kwargs)
        self.model_name = model_name
        self.table_id = table_id
        self.emb_vec_size = emb_vec_size
        self.emb_vec_dtype = emb_vec_dtype

    def call(self, inputs, training=False):
        """
        The forward logic of this wrapper class.

        Parameters
        ----------
        inputs: tf.Tensor
                keys are stored in Tensor. The data type must be tf.int64.

        training: boolean
                whether training or not. Only False is valid.

        Returns
        -------
        emb_vector: tf.float32
                the embedding vectors for the input keys. Its shape is
                *inputs.get_shape() + emb_vec_size*
        """
        emb_vector = lookup_ops.lookup(
            values=inputs,
            model_name=self.model_name,
            table_id=self.table_id,
            emb_vec_size=self.emb_vec_size,
            emb_vec_dtype=self.emb_vec_dtype,
        )
        output_shape = inputs.get_shape() + self.emb_vec_size
        emb_vector.set_shape(output_shape)
        return emb_vector
