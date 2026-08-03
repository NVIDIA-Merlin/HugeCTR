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

import os
import struct
import sys
import numpy as np
import json

FLOAT_BYTES = 4
INT64_BYTES = 8
DEFAULT_MAX_TENSOR_BYTES = 8 * 1024 * 1024 * 1024
MAX_TENSOR_BYTES_ENV = "HUGECTR2ONNX_MAX_TENSOR_BYTES"

ONNX_LAYER_TYPES = {
    "Add",
    "BatchNorm",
    "LayerNorm",
    "Concat",
    "Dropout",
    "ElementwiseMultiply",
    "ELU",
    "FmOrder2",
    "InnerProduct",
    "FusedInnerProduct",
    "MLP",
    "FusedReshapeConcat",
    "Interaction",
    "MatrixMultiply",
    "MultiHeadAttention",
    "MultiCross",
    "PReLU_Dice",
    "ReduceMean",
    "ReduceSum",
    "ReLU",
    "Reshape",
    "Scale",
    "Sigmoid",
    "Slice",
    "Softmax",
    "Sub",
    "WeightMultiply",
    "BinaryCrossEntropyLoss",
    "CrossEntropyLoss",
    "MultiCrossEntropyLoss",
    "SequenceMask",
}

EXEMPTION_LAYER_TYPES = {"Cast", "FusedReshapeConcatGeneral", "GRU", "Gather", "ReLUHalf", "Select"}


def get_tensor_names(clause):
    if isinstance(clause, list):
        return clause
    elif isinstance(clause, str):
        return [clause]
    else:
        return []


class LayerParams(object):
    def __init__(self):
        """Create LayerParams for HugeCTR"""
        self.layer_type = ""
        # Input Layer
        self.label_name = ""
        self.label_dim = 0
        self.dense_name = ""
        self.dense_dim = 0
        self.sparse_names = []
        self.sparse_dims = []
        # Embdding Layer
        self.combiner = 0
        # Dense Layer
        self.bottom_names = []
        self.top_names = []
        self.factor = 1.0
        self.eps = 0.00001
        self.dropout_rate = 0.9
        self.elu_alpha = 1.0
        self.prelu_alpha = 1.0
        self.prelu_eps = 0.00001
        self.scale_axis = 0
        self.scale_factor = 1
        self.num_output = 1
        self.num_layers = 0
        self.leading_dim = 1
        self.reshape_time_step = 0
        self.selected = False
        self.selected_slots = []
        self.ranges = []
        self.weight_dims = []
        self.out_dim = 0
        self.axis = 1
        self.max_sequence_len_from = 1
        self.max_sequence_len_to = 1
        self.num_attention_heads = 1
        self.transpose_b = True
        # MLP Layer
        self.activation = "Relu"
        self.activations = []
        self.num_outputs = []
        self.use_bias = True
        self.biases = []


class HugeCTRLoader(object):
    def __init__(
        self, graph_config, dense_model, convert_embedding=False, sparse_models=None, ntp_file=None
    ):
        """Create HugeCTRLoader
        Args:
            graph_config: str, model graph configuration JSON file
            dense_model: str, dense model file
            convert_embedding: boolean, whether converting sparse embedding models to ONNX
            sparse_models: List[str], sparse model files
            ntp_file: str, file that stores non-trainable parameters
        """
        self.__graph_config = graph_config
        self.__dense_model = dense_model
        self.__convert_embeddding = convert_embedding
        self.__sparse_models = sparse_models
        self.__ntp_file = ntp_file
        self.__max_tensor_bytes = self.__load_max_tensor_bytes()
        self.__dense_model_size = None
        with open(graph_config, "rb") as file:
            self.__layers_config = json.load(file)["layers"]
        self.__layers = len(self.__layers_config)
        self.__index = 0
        self.__embedding_counter = 0
        if self.__ntp_file != None and len(self.__ntp_file) > 0:
            with open(self.__ntp_file, "rb") as file:
                self.__ntp_config = json.load(file)["layers"]
        else:
            self.__ntp_config = None
        self.__ntp_counter = 0
        self.__dimensions = {}
        self.__offset = 0
        self.__vocab_size_all_tables = 0
        self.__key_to_indice_hash_all_tables = []
        self.__key_to_indice_hash_table_sizes = []
        for i in range(self.layers):
            layer_config = self.__layers_config[i]
            layer_type = layer_config["type"]
            if (
                layer_type == "DistributedSlotSparseEmbeddingHash"
                or layer_type == "LocalizedSlotSparseEmbeddingHash"
            ):
                max_vocab_size_global = layer_config["sparse_embedding_hparam"][
                    "max_vocabulary_size_global"
                ]
                max_vocab_size_global = self.__require_positive_int(
                    max_vocab_size_global,
                    "sparse_embedding_hparam.max_vocabulary_size_global",
                )
                self.__vocab_size_all_tables = self.__checked_sum(
                    "total sparse vocabulary size",
                    self.__vocab_size_all_tables,
                    max_vocab_size_global,
                )
                self.__key_to_indice_hash_table_sizes.append(self.__vocab_size_all_tables)

    @staticmethod
    def __load_max_tensor_bytes():
        max_tensor_bytes = os.environ.get(MAX_TENSOR_BYTES_ENV)
        if max_tensor_bytes is None:
            return DEFAULT_MAX_TENSOR_BYTES
        try:
            max_tensor_bytes = int(max_tensor_bytes)
        except ValueError as error:
            raise ValueError(
                MAX_TENSOR_BYTES_ENV + " must be a positive integer byte count"
            ) from error
        if max_tensor_bytes <= 0:
            raise ValueError(MAX_TENSOR_BYTES_ENV + " must be a positive integer byte count")
        return max_tensor_bytes

    @staticmethod
    def __require_non_negative_int(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("{} must be a non-negative integer, got {!r}".format(name, value))
        value = int(value)
        if value < 0:
            raise ValueError("{} must be a non-negative integer, got {!r}".format(name, value))
        return value

    @staticmethod
    def __require_positive_int(value, name):
        value = HugeCTRLoader.__require_non_negative_int(value, name)
        if value == 0:
            raise ValueError("{} must be a positive integer, got 0".format(name))
        return value

    def __checked_sum(self, name, *values):
        total = 0
        for value in values:
            total += self.__require_non_negative_int(value, name)
            if total > sys.maxsize:
                raise ValueError("{} is too large".format(name))
        return total

    def __checked_product(self, name, *values):
        product = 1
        for value in values:
            product *= self.__require_positive_int(value, name)
            if product > sys.maxsize:
                raise ValueError("{} is too large".format(name))
        return product

    def __validate_tensor_bytes(self, byte_count, name):
        byte_count = self.__require_positive_int(byte_count, name + " byte count")
        if byte_count > self.__max_tensor_bytes:
            raise ValueError(
                "{} requires {} bytes, exceeding {}={} bytes".format(
                    name, byte_count, MAX_TENSOR_BYTES_ENV, self.__max_tensor_bytes
                )
            )
        return byte_count

    @staticmethod
    def __get_file_size(path, name):
        try:
            return os.path.getsize(path)
        except OSError as error:
            raise ValueError("Unable to read {} size from {}: {}".format(name, path, error)) from error

    def __get_dense_model_size(self):
        if self.__dense_model_size is None:
            self.__dense_model_size = self.__get_file_size(self.__dense_model, "dense model")
        return self.__dense_model_size

    def __read_dense_model_bytes(self, layer_type, layer_bytes):
        layer_bytes = self.__validate_tensor_bytes(layer_bytes, layer_type + " weights")
        dense_model_size = self.__get_dense_model_size()
        if self.__offset > dense_model_size:
            raise ValueError(
                "{} layer starts at offset {} beyond dense model size {} for {}".format(
                    layer_type, self.__offset, dense_model_size, self.__dense_model
                )
            )
        remaining_bytes = dense_model_size - self.__offset
        if layer_bytes > remaining_bytes:
            raise ValueError(
                "{} layer requires {} bytes from dense model {} at offset {}, "
                "but only {} bytes remain".format(
                    layer_type,
                    layer_bytes,
                    self.__dense_model,
                    self.__offset,
                    remaining_bytes,
                )
            )
        with open(self.__dense_model, "rb") as file:
            file.seek(self.__offset, 0)
            buffer = file.read(layer_bytes)
        if len(buffer) != layer_bytes:
            raise ValueError(
                "{} layer expected {} bytes from dense model {}, got {}".format(
                    layer_type, layer_bytes, self.__dense_model, len(buffer)
                )
            )
        return buffer

    def __unpack_floats(self, buffer, offset, count, name):
        count = self.__require_positive_int(count, name + " count")
        byte_count = self.__checked_product(name + " byte size", count, FLOAT_BYTES)
        end = self.__checked_sum(name + " buffer range", offset, byte_count)
        if end > len(buffer):
            raise ValueError(
                "{} requires {} bytes at offset {}, but buffer has {} bytes".format(
                    name, byte_count, offset, len(buffer)
                )
            )
        return struct.unpack_from(str(count) + "f", buffer, offset)

    @property
    def key_to_indice_hash_all_tables(self):
        return self.__key_to_indice_hash_all_tables

    @property
    def dimensions(self):
        return self.__dimensions

    @property
    def layers(self):
        return self.__layers

    def load_layer(self):
        layer_params = LayerParams()
        layer_weights_dict = {}
        layer_config = self.__layers_config[self.__index]
        layer_params.layer_type = layer_config["type"]
        layer_params.bottom_names = get_tensor_names(layer_config.get("bottom"))
        layer_params.top_names = get_tensor_names(layer_config.get("top"))
        layer_type = layer_config["type"]
        if layer_type == "Data":
            layer_params.label_name = layer_config["label"]["top"]
            layer_params.label_dim = layer_config["label"]["label_dim"]
            layer_params.dense_name = layer_config["dense"]["top"]
            layer_params.dense_dim = layer_config["dense"]["dense_dim"]
            layer_params.sparse_names = []
            layer_params.sparse_dims = []
            for i in range(len(layer_config["sparse"])):
                sparse_i = layer_config["sparse"][i]
                layer_params.sparse_names.append(sparse_i["top"])
                max_nnz = max(sparse_i["nnz_per_slot"])
                layer_params.sparse_dims.append((sparse_i["slot_num"], max_nnz))
                self.__dimensions[sparse_i["top"]] = (sparse_i["slot_num"], max_nnz)

            if np.ndim(layer_params.label_dim) == 0:
                self.__dimensions[layer_params.label_name] = layer_params.label_dim
            else:
                for name, dim in zip(layer_params.label_name, layer_params.label_dim):
                    self.__dimensions[name] = dim

            self.__dimensions[layer_params.dense_name] = layer_params.dense_dim
            layer_weights_dict["key_to_indice_hash_all_tables"] = self.key_to_indice_hash_all_tables
        elif (
            layer_type == "DistributedSlotSparseEmbeddingHash"
            or layer_type == "LocalizedSlotSparseEmbeddingHash"
        ):
            embedding_vec_size = self.__require_positive_int(
                layer_config["sparse_embedding_hparam"]["embedding_vec_size"],
                "sparse_embedding_hparam.embedding_vec_size",
            )
            self.__dimensions[layer_config["top"]] = (
                self.__dimensions[layer_config["bottom"]][0],
                embedding_vec_size,
            )
            if self.__convert_embeddding:
                layer_params.combiner = (
                    0 if layer_config["sparse_embedding_hparam"]["combiner"] == "sum" else 1
                )
                max_vocab_size_global = layer_config["sparse_embedding_hparam"][
                    "max_vocabulary_size_global"
                ]
                max_vocab_size_global = self.__require_positive_int(
                    max_vocab_size_global,
                    "sparse_embedding_hparam.max_vocabulary_size_global",
                )
                key_path = os.path.join(
                    self.__sparse_models[self.__embedding_counter], "key"
                )
                vec_path = os.path.join(
                    self.__sparse_models[self.__embedding_counter], "emb_vector"
                )
                vector_bytes = self.__checked_product(
                    "sparse embedding vector byte size", embedding_vec_size, FLOAT_BYTES
                )
                self.__validate_tensor_bytes(vector_bytes, "sparse embedding vector")
                key_file_size = self.__get_file_size(key_path, "sparse embedding key file")
                vec_file_size = self.__get_file_size(vec_path, "sparse embedding vector file")
                if key_file_size % INT64_BYTES != 0:
                    raise ValueError(
                        "Sparse embedding key file {} size {} is not aligned to {} bytes".format(
                            key_path, key_file_size, INT64_BYTES
                        )
                    )
                if vec_file_size % vector_bytes != 0:
                    raise ValueError(
                        "Sparse embedding vector file {} size {} is not aligned to "
                        "embedding vector size {} bytes".format(
                            vec_path, vec_file_size, vector_bytes
                        )
                    )
                key_count = key_file_size // INT64_BYTES
                vector_count = vec_file_size // vector_bytes
                if key_count != vector_count:
                    raise ValueError(
                        "Sparse embedding key/vector row count mismatch for {} and {}: "
                        "{} keys vs {} vectors".format(
                            key_path, vec_path, key_count, vector_count
                        )
                    )
                if vector_count > max_vocab_size_global:
                    raise ValueError(
                        "Sparse embedding vector count {} exceeds max_vocabulary_size_global "
                        "{}".format(vector_count, max_vocab_size_global)
                    )
                embedding_rows = self.__checked_sum(
                    "sparse embedding table rows", vector_count, 1
                )
                embedding_table_bytes = self.__checked_product(
                    "sparse embedding table byte size",
                    embedding_rows,
                    embedding_vec_size,
                    FLOAT_BYTES,
                )
                self.__validate_tensor_bytes(embedding_table_bytes, "sparse embedding table")
                hash_table_size = self.__key_to_indice_hash_table_sizes[
                    self.__embedding_counter
                ]
                hash_table_bytes = self.__checked_product(
                    "sparse embedding hash table byte size",
                    hash_table_size,
                    np.dtype(np.int64).itemsize,
                )
                self.__validate_tensor_bytes(hash_table_bytes, "sparse embedding hash table")
                # indice 0 is reserved for default values of non-exisiting keys
                embedding_table = np.zeros(
                    shape=(embedding_rows, embedding_vec_size), dtype=np.float32
                )
                hash_table = np.zeros(shape=(hash_table_size,), dtype=np.int64)
                with open(key_path, "rb") as key_file, open(vec_path, "rb") as vec_file:
                    # indice 0 is reserved for default values of non-exisiting keys
                    for indice in range(1, vector_count + 1):
                        key_buffer = key_file.read(INT64_BYTES)
                        vec_buffer = vec_file.read(vector_bytes)
                        if len(key_buffer) != INT64_BYTES or len(vec_buffer) != vector_bytes:
                            raise ValueError(
                                "Sparse embedding files changed while reading {} and {}".format(
                                    key_path, vec_path
                                )
                            )
                        key = struct.unpack("q", key_buffer)[0]
                        if key < 0 or key >= hash_table_size:
                            raise ValueError(
                                "Sparse embedding key {} is outside hash table range [0, {})".format(
                                    key, hash_table_size
                                )
                            )
                        values = self.__unpack_floats(
                            vec_buffer, 0, embedding_vec_size, "sparse embedding vector"
                        )
                        hash_table[key] = indice
                        embedding_table[indice] = values
                self.__key_to_indice_hash_all_tables.append(hash_table)
                layer_weights_dict["embedding_table"] = embedding_table
                layer_weights_dict["hash_table"] = hash_table
                self.__embedding_counter += 1
            else:
                print("Skip sparse embedding layers in converted ONNX model")
        elif layer_type == "Add":
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"][0]]
        elif layer_type == "BatchNorm":
            layer_params.factor = layer_config["bn_param"]["factor"]
            layer_params.eps = layer_config["bn_param"]["eps"]
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"]]
            in_feature = self.__require_positive_int(
                self.__dimensions[layer_config["bottom"]], "BatchNorm input feature"
            )
            layer_bytes = self.__checked_product(
                "BatchNorm weights byte size", in_feature, FLOAT_BYTES, 2
            )
            buffer = self.__read_dense_model_bytes(layer_type, layer_bytes)
            gamma = self.__unpack_floats(buffer, 0, in_feature, "BatchNorm gamma")
            beta = self.__unpack_floats(
                buffer, in_feature * FLOAT_BYTES, in_feature, "BatchNorm beta"
            )
            gamma = np.reshape(np.float32(gamma), newshape=(in_feature,))
            beta = np.reshape(np.float32(beta), newshape=(in_feature,))
            self.__offset += layer_bytes
            ntp_config = self.__ntp_config[self.__ntp_counter]
            running_mean = np.array(ntp_config["mean"], dtype=np.float32)
            running_variance = np.array(ntp_config["var"], dtype=np.float32)
            self.__ntp_counter += 1
            layer_weights_dict[layer_config["top"] + "_gamma"] = gamma
            layer_weights_dict[layer_config["top"] + "_beta"] = beta
            layer_weights_dict[layer_config["top"] + "_running_mean"] = running_mean
            layer_weights_dict[layer_config["top"] + "_running_variance"] = running_variance
        elif layer_type == "LayerNorm":
            layer_params.eps = layer_config["ln_param"]["eps"]
            dim_in = self.__dimensions[layer_config["bottom"]]
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"]]
            in_feature = self.__require_positive_int(
                dim_in[len(dim_in) - 1], "LayerNorm input feature"
            )
            layer_bytes = self.__checked_product(
                "LayerNorm weights byte size", in_feature, FLOAT_BYTES, 2
            )
            buffer = self.__read_dense_model_bytes(layer_type, layer_bytes)
            gamma = self.__unpack_floats(buffer, 0, in_feature, "LayerNorm gamma")
            beta = self.__unpack_floats(
                buffer, in_feature * FLOAT_BYTES, in_feature, "LayerNorm beta"
            )
            gamma = np.reshape(np.float32(gamma), newshape=(in_feature,))
            beta = np.reshape(np.float32(beta), newshape=(in_feature,))
            self.__offset += layer_bytes
            # ntp_config = self.__ntp_config[self.__ntp_counter]
            # running_mean = np.array(ntp_config["mean"], dtype = np.float32)
            # running_variance = np.array(ntp_config["var"], dtype = np.float32)
            # self.__ntp_counter += 1
            layer_weights_dict[layer_config["top"] + "_gamma"] = gamma
            layer_weights_dict[layer_config["top"] + "_beta"] = beta
            # layer_weights_dict[layer_config["top"]+"_running_mean"] = running_mean
            # layer_weights_dict[layer_config["top"]+"_running_variance"] = running_variance
        elif layer_type == "Concat":
            layer_params.axis = layer_config["axis"]
            axis_without_batch = layer_config["axis"] - 1
            dim = 0
            for tensor in layer_config["bottom"]:
                if isinstance(self.__dimensions[tensor], tuple):
                    dims = self.__dimensions[tensor]
                    for i in range(len(dims)):
                        if i == axis_without_batch:
                            dim = dim + dims[i]
                else:
                    dim += self.__dimensions[tensor]
            if isinstance(self.__dimensions[layer_config["bottom"][0]], tuple):
                self.__dimensions[layer_config["top"]] = tuple(
                    [
                        dims[i] if i != axis_without_batch else dim
                        for i in range(len(self.__dimensions[layer_config["bottom"][0]]))
                    ]
                )
            else:
                self.__dimensions[layer_config["top"]] = dim
        elif layer_type == "Dropout":
            layer_params.dropout_rate = layer_config["rate"]
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"]]
        elif layer_type == "ElementwiseMultiply":
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"][0]]
        elif layer_type == "ELU":
            layer_params.elu_alpha = layer_config["elu_param"]["alpha"]
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"]]
        elif layer_type == "SequenceMask":
            layer_params.max_sequence_len_from = layer_config["max_sequence_len_from"]
            layer_params.max_sequence_len_to = layer_config["max_sequence_len_to"]
            self.__dimensions[layer_config["top"]] = (
                1,
                layer_params.max_sequence_len_from,
                layer_params.max_sequence_len_to,
            )
        elif layer_type == "FmOrder2":
            layer_params.out_dim = layer_config["out_dim"]
            self.__dimensions[layer_config["top"]] = layer_params.out_dim
        elif layer_type == "InnerProduct" or layer_type == "FusedInnerProduct":
            layer_params.num_output = self.__require_positive_int(
                layer_config["fc_param"]["num_output"], layer_type + " num_output"
            )
            dim = self.__dimensions[layer_config["bottom"]]
            if isinstance(dim, tuple):
                seq_len = dim[0]
                hidden_in = self.__require_positive_int(dim[1], layer_type + " input feature")
                self.__dimensions[layer_config["top"]] = (seq_len, layer_params.num_output)
                in_feature = hidden_in
            else:
                self.__dimensions[layer_config["top"]] = layer_params.num_output
                in_feature = self.__require_positive_int(
                    self.__dimensions[layer_config["bottom"]], layer_type + " input feature"
                )
            out_feature = layer_params.num_output
            weight_count = self.__checked_product(
                layer_type + " weight count", in_feature, out_feature
            )
            param_count = self.__checked_sum(layer_type + " parameter count", weight_count, out_feature)
            layer_bytes = self.__checked_product(
                layer_type + " weights byte size", param_count, FLOAT_BYTES
            )
            buffer = self.__read_dense_model_bytes(layer_type, layer_bytes)
            weight = self.__unpack_floats(buffer, 0, weight_count, layer_type + " weight")
            bias = self.__unpack_floats(
                buffer, weight_count * FLOAT_BYTES, out_feature, layer_type + " bias"
            )
            weight = np.reshape(np.float32(weight), newshape=(in_feature, out_feature))
            bias = np.reshape(np.float32(bias), newshape=(1, out_feature))
            self.__offset += layer_bytes
            layer_weights_dict[layer_config["top"] + "_weight"] = weight
            layer_weights_dict[layer_config["top"] + "_bias"] = bias
        elif layer_type == "MLP":
            if "num_outputs" in layer_config["mlp_param"]:
                layer_params.num_outputs = layer_config["mlp_param"]["num_outputs"]
            if "activation" in layer_config["mlp_param"]:
                layer_params.activation = layer_config["mlp_param"]["activation"]
            if "activations" in layer_config["mlp_param"]:
                layer_params.activations = layer_config["mlp_param"]["activations"]
            if "use_bias" in layer_config["mlp_param"]:
                layer_params.use_bias = layer_config["mlp_param"]["use_bias"]
            if "biases" in layer_config["mlp_param"]:
                layer_params.biases = layer_config["mlp_param"]["biases"]
            for i in range(len(layer_params.num_outputs)):
                in_feature = self.__require_positive_int(
                    self.__dimensions[layer_config["bottom"]],
                    "MLP layer {} input feature".format(i),
                )
                if i != 0:
                    in_feature = self.__require_positive_int(
                        layer_params.num_outputs[i - 1],
                        "MLP layer {} input feature".format(i),
                    )
                out_feature = self.__require_positive_int(
                    layer_params.num_outputs[i], "MLP layer {} output feature".format(i)
                )
                weight_count = self.__checked_product(
                    "MLP layer {} weight count".format(i), in_feature, out_feature
                )
                param_count = self.__checked_sum(
                    "MLP layer {} parameter count".format(i), weight_count, out_feature
                )
                layer_bytes = self.__checked_product(
                    "MLP layer {} weights byte size".format(i), param_count, FLOAT_BYTES
                )
                buffer = self.__read_dense_model_bytes(layer_type, layer_bytes)
                weight = self.__unpack_floats(buffer, 0, weight_count, "MLP layer {} weight".format(i))
                bias = self.__unpack_floats(
                    buffer,
                    weight_count * FLOAT_BYTES,
                    out_feature,
                    "MLP layer {} bias".format(i),
                )
                weight = np.reshape(np.float32(weight), newshape=(in_feature, out_feature))
                bias = np.reshape(np.float32(bias), newshape=(1, out_feature))
                self.__offset += layer_bytes
                layer_weights_dict[layer_config["top"] + str(i) + "_weight"] = weight
                layer_weights_dict[layer_config["top"] + str(i) + "_bias"] = bias
        elif layer_type == "FusedReshapeConcat":
            num_output = 0
            for tensor_name in layer_params.bottom_names:
                num_output += self.__dimensions[tensor_name][1]
            for tensor_name in layer_params.top_names:
                self.__dimensions[tensor_name] = num_output
        elif layer_type == "Interaction":
            slot_num = self.__dimensions[layer_params.bottom_names[1]][0]
            vec_size = self.__dimensions[layer_params.bottom_names[1]][1]
            self.__dimensions[layer_config["top"]] = (
                vec_size + (slot_num + 1) * (slot_num + 2) // 2 - (slot_num + 1) + 1
            )
        elif layer_type == "MultiHeadAttention":
            layer_params.num_attention_heads = layer_config["num_attention_heads"]
            dim1 = self.__dimensions[layer_params.bottom_names[0]]
            self.__dimensions[layer_config["top"]] = (dim1[0], dim1[1])
        elif layer_type == "MatrixMultiply":
            dim1 = self.__dimensions[layer_params.bottom_names[0]]
            dim2 = self.__dimensions[layer_params.bottom_names[1]]
            if len(dim1) == 3:
                self.__dimensions[layer_config["top"]] = (dim1[0], dim1[1], dim2[2])
            elif len(dim1) == 2:
                self.__dimensions[layer_config["top"]] = (dim1[0], dim2[1])
            else:
                self.__dimensions[layer_config["top"]] = dim2[1]
        elif layer_type == "MultiCross":
            layer_params.num_layers = self.__require_positive_int(
                layer_config["mc_param"]["num_layers"], "MultiCross num_layers"
            )
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"]]
            num_layers = layer_params.num_layers
            in_feature = self.__require_positive_int(
                self.__dimensions[layer_config["bottom"]], "MultiCross input feature"
            )
            param_count = self.__checked_product(
                "MultiCross parameter count", in_feature, 2, num_layers
            )
            layer_bytes = self.__checked_product(
                "MultiCross weights byte size", param_count, FLOAT_BYTES
            )
            buffer = self.__read_dense_model_bytes(layer_type, layer_bytes)
            weights = []
            biases = []
            each_layer_bytes = self.__checked_product(
                "MultiCross layer byte size", in_feature, 2, FLOAT_BYTES
            )
            for i in range(num_layers):
                layer_offset = i * each_layer_bytes
                weight = self.__unpack_floats(
                    buffer, layer_offset, in_feature, "MultiCross layer {} weight".format(i)
                )
                bias = self.__unpack_floats(
                    buffer,
                    layer_offset + in_feature * FLOAT_BYTES,
                    in_feature,
                    "MultiCross layer {} bias".format(i),
                )
                weights.append(np.reshape(np.float32(weight), newshape=(len(weight), 1)))
                biases.append(np.reshape(np.float32(bias), newshape=(1, len(bias))))
            self.__offset += layer_bytes
            layer_weights_dict[layer_config["top"] + "_weights"] = weights
            layer_weights_dict[layer_config["top"] + "_biases"] = biases
        elif layer_type == "PReLU_Dice":
            layer_params.prelu_alpha = layer_config["prelu_dice_param"]["alpha"]
            layer_params.prelu_eps = layer_config["prelu_dice_param"]["eps"]
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"]]
        elif layer_type == "ReduceMean":
            # keepdims = 1, 0 < axis < N
            layer_params.axis = layer_config["axis"]
            axis_without_batch = layer_config["axis"] - 1
            if isinstance(self.__dimensions[layer_params.bottom_names[0]], tuple):
                dims = self.__dimensions[layer_params.bottom_names[0]]
                self.__dimensions[layer_params.top_names[0]] = tuple(
                    [dims[i] if i != axis_without_batch else 1 for i in range(len(dims))]
                )
            else:
                dims = (self.__dimensions[layer_params.bottom_names[0]],)
                self.__dimensions[layer_params.top_names[0]] = 1
        elif layer_type == "ReduceSum":
            # keepdims = 1, 0 < axis < N
            layer_params.axis = layer_config["axis"]
            axis_without_batch = layer_config["axis"] - 1
            if isinstance(self.__dimensions[layer_params.bottom_names[0]], tuple):
                dims = self.__dimensions[layer_params.bottom_names[0]]
                self.__dimensions[layer_params.top_names[0]] = tuple(
                    [dims[i] if i != axis_without_batch else 1 for i in range(len(dims))]
                )
            else:
                dims = (self.__dimensions[layer_params.bottom_names[0]],)
                self.__dimensions[layer_params.top_names[0]] = 1
        elif layer_type == "ReLU":
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"]]
        elif layer_type == "Reshape":
            layer_params.selected_slots = layer_config.get("selected")
            layer_params.selected = layer_params.selected_slots is not None
            if not layer_params.selected:
                layer_params.leading_dim = layer_config["leading_dim"]
                layer_params.reshape_time_step = (
                    0 if layer_config.get("time_step") is None else layer_config["time_step"]
                )
            else:
                layer_params.leading_dim = (
                    len(layer_params.selected_slots) * self.__dimensions[layer_config["bottom"]][1]
                )
            if layer_params.reshape_time_step == 0:
                self.__dimensions[layer_config["top"]] = layer_params.leading_dim
            else:
                self.__dimensions[layer_config["top"]] = (
                    layer_params.reshape_time_step,
                    layer_params.leading_dim,
                )
        elif layer_type == "Scale":
            layer_params.scale_axis = layer_config["scale_param"]["axis"]
            layer_params.scale_factor = layer_config["scale_param"]["factor"]
            if layer_params.scale_axis == 0:
                self.__dimensions[layer_config["top"]] = self.__dimensions[
                    layer_config["bottom"]
                ] * int(layer_params.scale_factor)
            else:
                self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"]]
        elif layer_type == "Sigmoid":
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"]]
        elif layer_type == "Slice":
            layer_params.ranges = layer_config["ranges"]
            dim_in = self.__dimensions[layer_config["bottom"]]
            for tensor, dim in zip(layer_config["top"], layer_params.ranges):
                if isinstance(dim_in, tuple):
                    self.__dimensions[tensor] = tuple(
                        [
                            dim_in[i] if i != len(dim_in) - 1 else dim[1] - dim[0]
                            for i in range(len(dim_in))
                        ]
                    )
                else:
                    self.__dimensions[tensor] = dim[1] - dim[0]
        elif layer_type == "Softmax":
            layer_params.factor = layer_config["factor"]
            if isinstance(layer_config["bottom"], list):
                self.__dimensions[layer_config["top"]] = self.__dimensions[
                    layer_config["bottom"][0]
                ]
            else:
                self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"]]

        elif layer_type == "Sub":
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"][0]]
        elif layer_type == "WeightMultiply":
            layer_params.weight_dims = layer_config["weight_dims"]
            if len(layer_params.weight_dims) != 2:
                raise ValueError("WeightMultiply weight_dims must contain exactly two integers")
            slot_num = self.__require_positive_int(
                layer_params.weight_dims[0], "WeightMultiply slot_num"
            )
            vec_size = self.__require_positive_int(
                layer_params.weight_dims[1], "WeightMultiply vec_size"
            )
            self.__dimensions[layer_config["top"]] = slot_num * vec_size
            weight_count = self.__checked_product(
                "WeightMultiply weight count", slot_num, vec_size
            )
            layer_bytes = self.__checked_product(
                "WeightMultiply weights byte size", weight_count, FLOAT_BYTES
            )
            buffer = self.__read_dense_model_bytes(layer_type, layer_bytes)
            weight = self.__unpack_floats(buffer, 0, weight_count, "WeightMultiply weight")
            weight = np.reshape(np.float32(weight), newshape=(slot_num, vec_size))
            self.__offset += layer_bytes
            layer_weights_dict[layer_config["top"] + "_weight"] = weight
        elif layer_type == "BinaryCrossEntropyLoss":
            layer_params.layer_type = "Sigmoid"
            pred_name = layer_params.bottom_names[0]
            label_name = layer_params.bottom_names[1]
            layer_params.bottom_names = [pred_name]
            layer_params.top_names = [label_name]
        elif layer_type == "CrossEntropyLoss":
            layer_params.layer_type = "Softmax"
            pred_name = layer_params.bottom_names[0]
            layer_params.bottom_names = [pred_name]
            layer_params.top_names = []
        elif layer_type == "MultiCrossEntropyLoss":
            layer_params.layer_type = "Sigmoid"
            pred_name = layer_params.bottom_names[0]
            layer_params.bottom_names = [pred_name]
            layer_params.top_names = []
        else:
            raise ValueError(
                layer_type
                + " is not supported in HugeCTR to ONNX converter, please refer to "
                + "https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/onnx_converter#layer-support "
                + "to see the supported layers."
            )
        self.__index += 1
        return layer_params, layer_weights_dict, self.dimensions
