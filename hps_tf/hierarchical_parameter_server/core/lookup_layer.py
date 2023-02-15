"""
 Copyright (c) 2023, NVIDIA CORPORATION.

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

    This is a wrapper class for HPS lookup layer, which basically performs
    the same function as ``tf.nn.embedding_lookup``. Note that ``ps_config_file``
    and ``global_batch_size`` should be specified in the constructor if you want
    to use implicit HPS initialization.

    Parameters
    ----------
    model_name: str
            The name of the model that has embedding tables.
    table_id: int
            The index of the embedding table for the model specified by
            model_name.
    emb_vec_size: int
            The embedding vector size for the embedding table specified
            by model_name and table_id.
    emb_vec_dtype:
            The data type of embedding vectors which must be ``tf.float32``.
    ps_config_file: str
            The JSON configuration file for HPS initialization.
    global_batch_size: int
            The global batch size for HPS that is deployed on multiple GPUs.

    Examples
    --------
    .. code-block:: python

        import hierarchical_parameter_server as hps

        lookup_layer = hps.LookupLayer(model_name = args.model_name,
                                      table_id = args.table_id,
                                      emb_vec_size = args.embed_vec_size,
                                      emb_vec_dtype = tf.float32,
                                      ps_config_file = args.ps_config_file,
                                      global_batch_size = args.global_batch_size)

        @tf.function
        def _infer_step(inputs):
            embedding_vector = lookup_layer(inputs)
            ...

        for i, (inputs, labels) in enumerate(dataset):
            _infer_step(inputs)
    """

    def __init__(
        self,
        model_name,
        table_id,
        emb_vec_size,
        emb_vec_dtype,
        ps_config_file="",
        global_batch_size=1,
        **kwargs
    ):
        super(LookupLayer, self).__init__(**kwargs)
        self.model_name = model_name
        self.table_id = table_id
        self.emb_vec_size = emb_vec_size
        self.emb_vec_dtype = emb_vec_dtype
        self.ps_config_file = ps_config_file
        self.global_batch_size = global_batch_size

    def call(self, ids, max_norm=None):
        """
        The forward logic of this wrapper class.

        Parameters
        ----------
        ids:
                Keys are stored in Tensor. The supported data types are ``tf.int32`` and ``tf.int64``.
        max_norm:
            if not ``None``, each embedding is clipped if its l2-norm is larger
            than this value.

        Returns
        -------
        emb_vector: ``tf.Tensor`` of float32
                the embedding vectors for the input keys. Its shape is
                *ids.get_shape() + emb_vec_size*.
        """
        emb_vector = lookup_ops.lookup(
            ids=ids,
            model_name=self.model_name,
            table_id=self.table_id,
            emb_vec_size=self.emb_vec_size,
            emb_vec_dtype=self.emb_vec_dtype,
            ps_config_file=self.ps_config_file,
            global_batch_size=self.global_batch_size,
            max_norm=max_norm,
        )
        output_shape = ids.get_shape() + self.emb_vec_size
        emb_vector.set_shape(output_shape)
        return emb_vector
