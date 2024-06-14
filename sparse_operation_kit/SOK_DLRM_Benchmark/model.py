import os
import math
from typing import Dict, List, Optional

import sparse_operation_kit as sok
import tensorflow as tf
import numpy as np
import horovod.tensorflow as hvd
from typing import List, Optional

try:
    from tensorflow_dot_based_interact.python.ops import dot_based_interact_ops
except:
    pass


class SOKEmbedding(tf.keras.layers.Layer):
    def __init__(
        self, vocab_sizes: List[int], embedding_vec_size: int, num_gpus: int = 1, **kwargs
    ):
        def next_power_of_2(n):
            exponent = math.ceil(math.log2(n))
            return 2**exponent

        super(SOKEmbedding, self).__init__(**kwargs)
        self._vocab_sizes = vocab_sizes
        self._embedding_vec_size = embedding_vec_size

        self._sok_embedding = []
        for i in range(len(vocab_sizes)):
            max_capacity = next_power_of_2(math.ceil(vocab_sizes[i] / num_gpus))
            init_capacity = 1024
            if max_capacity < init_capacity:
                init_capacity = max_capacity
            self._sok_embedding.append(
                sok.DynamicVariable(
                    var_type="hybrid",
                    dimension=self._embedding_vec_size,  # 128 in Criteo Terabyte Dataset
                    init_capacity=init_capacity,
                    max_capacity=max_capacity,
                )
            )

    def call(self, inputs, combiners, training=True):
        """
        Compute the output of embedding layer.
        Args:
            inputs: [batch_size, 26] int64 tensor
        Returns:
            emb_vectors: [batch_size, 26, embedding_vec_size] float32 tensor
        """

        emb_vectors = sok.lookup_sparse(self._sok_embedding, inputs, combiners=combiners)
        stacked_emb_vectors = tf.stack(emb_vectors, axis=1)

        return stacked_emb_vectors


class TFEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_sizes, embedding_vec_size, num_gpus=1, **kwargs):
        super(TFEmbedding, self).__init__(**kwargs)
        assert num_gpus == 1
        self._vocab_sizes = vocab_sizes
        self._embedding_vec_size = embedding_vec_size

        self._tf_embeddings = []
        for size in vocab_sizes:
            emb = tf.keras.layers.Embedding(
                input_dim=size,
                output_dim=self._embedding_vec_size,
            )
            self._tf_embeddings.append(emb)

    def call(self, inputs, training=True):
        fused_inputs = tf.split(inputs, num_or_size_splits=len(self._vocab_sizes), axis=1)
        emb_vectors = []
        for i in range(len(self._tf_embeddings)):
            out = self._tf_embeddings[i](fused_inputs[i])
            emb_vectors.append(out)
        emb_vectors = tf.concat(emb_vectors, axis=1)

        return emb_vectors


class MLP(tf.keras.layers.Layer):
    """Sequential multi-layer perceptron (MLP) block."""

    def __init__(
        self,
        units: List[int],
        use_bias: bool = True,
        activation="relu",
        final_activation=None,
        **kwargs,
    ):
        """Initializes the MLP layer.
        Args:
        units: Sequential list of layer sizes. List[int].
        use_bias: Whether to include a bias term.
        activation: Type of activation to use on all except the last layer.
        final_activation: Type of activation to use on last layer.
        **kwargs: Extra args passed to the Keras Layer base class.
        """
        super().__init__(**kwargs)

        self._sublayers = []
        for i, num_units in enumerate(units):
            kernel_init = tf.keras.initializers.GlorotNormal()
            bias_init = tf.keras.initializers.RandomNormal(stddev=math.sqrt(1.0 / num_units))

            self._sublayers.append(
                tf.keras.layers.Dense(
                    num_units,
                    use_bias=use_bias,
                    kernel_initializer=kernel_init,
                    bias_initializer=bias_init,
                    activation=activation if i < (len(units) - 1) else final_activation,
                )
            )

    def call(self, x: tf.Tensor):
        """Performs the forward computation of the block."""
        for layer in self._sublayers:
            x = layer(x)
        return x


class LowRankCrossNetLayer(tf.keras.layers.Layer):
    def __init__(self, in_features, num_layers, low_rank, **kwargs):
        super(LowRankCrossNetLayer, self).__init__(**kwargs)
        self.in_features = in_features
        self.num_layers = num_layers
        self.low_rank = low_rank
        self.V_kernels = []
        self.W_kernels = []
        self.biases = []

        for _ in range(num_layers):
            self.V_kernels.append(
                self.add_weight(
                    shape=(in_features, self.low_rank), initializer="glorot_normal", trainable=True
                )
            )
            self.W_kernels.append(
                self.add_weight(
                    shape=(self.low_rank, in_features), initializer="glorot_normal", trainable=True
                )
            )
            self.biases.append(
                self.add_weight(shape=(in_features,), initializer="zeros", trainable=True)
            )

    def call(self, inputs):
        x_0 = inputs
        x_l = x_0

        for layer in range(self.num_layers):
            x_l_v = tf.matmul(x_l, self.V_kernels[layer])
            x_l_w = tf.matmul(x_l_v, self.W_kernels[layer])
            x_l = x_0 * (x_l_w + self.biases[layer]) + x_l

        return x_l


class InteractionDCNArch(tf.keras.layers.Layer):
    def __init__(self, num_sparse_features, crossnet, **kwargs):
        super(InteractionDCNArch, self).__init__(**kwargs)
        self.F = num_sparse_features
        self.crossnet = crossnet

    def call(self, inputs):
        """
        Args:
            inputs: a list of two tensors [dense_features, sparse_features]
                    dense_features is a tensor of shape (B, D)
                    sparse_features is a tensor of shape (B, F, D)

        Returns:
            TensorFlow tensor of shape (B, F*D + D)
        """
        dense_features, sparse_features = inputs
        if self.F <= 0:
            return dense_features

        B, D = tf.shape(dense_features)[0], tf.shape(dense_features)[1]

        # Concatenating dense_features and sparse_features
        combined_values = tf.concat([tf.expand_dims(dense_features, 1), sparse_features], axis=1)

        # Reshaping to size (B, -1) to flatten the features
        combined_values_reshaped = tf.reshape(combined_values, [B, -1])

        # Passing through the crossnet
        output = self.crossnet(combined_values_reshaped)

        return output


class DLRM(tf.keras.models.Model):
    def __init__(
        self,
        vocab_sizes: List[int],
        # num_features: int,
        num_dense_features: int,
        num_sparse_features: int,
        embedding_vec_size: int,
        dcn_num_layers: int,
        dcn_low_rank_dim: int,
        bottom_stack_units: List[int],
        top_stack_units: List[int],
        num_gpus=1,
        **kwargs,
    ):
        super(DLRM, self).__init__(**kwargs)
        self._vocab_sizes = vocab_sizes
        self._num_dense_features = num_dense_features
        self._embedding_vec_size = embedding_vec_size
        self._combinears = ["sum" for i in range(num_sparse_features)]

        self._embedding_layer = SOKEmbedding(self._vocab_sizes, self._embedding_vec_size, num_gpus)
        self._bottom_stack = MLP(units=bottom_stack_units, final_activation="relu")

        self._crossnet = LowRankCrossNetLayer(
            (num_sparse_features + 1) * self._embedding_vec_size, dcn_num_layers, dcn_low_rank_dim
        )
        self._feature_interaction = InteractionDCNArch(num_sparse_features, self._crossnet)
        self._top_stack = MLP(units=top_stack_units, final_activation=None)

    def call(self, inputs: List[tf.Tensor], training=True):
        dense_input = inputs[0]
        sparse_input = inputs[1]
        if amp:
            dense_input = tf.cast(dense_input, tf.float16)
        dense_embedding_vec = self._bottom_stack(dense_input)
        sparse_embeddings = self._embedding_layer(sparse_input, self._combinears, training=training)
        inter_acrh_inputs = [dense_embedding_vec, sparse_embeddings]
        inter_arch_outputs = self._feature_interaction(inter_acrh_inputs)

        prediction = self._top_stack(inter_arch_outputs)
        return prediction
