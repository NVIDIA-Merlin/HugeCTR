# 
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import struct
import numpy as np
import json

ONNX_LAYER_TYPES = {
    "Add",
    "BatchNorm",
    "Concat",
    "Dropout",
    "ElementwiseMultiply",
    "ELU",
    "FmOrder2",
    "InnerProduct",
    "FusedInnerProduct",
    "FusedReshapeConcat",
    "Interaction",
    "MatrixMultiply",
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
    "MultiCrossEntropyLoss"}

EXEMPTION_LAYER_TYPES = {
 "Cast",
 "FusedReshapeConcatGeneral",
 "GRU",
 "Gather",
 "ReLUHalf"
}

def get_tensor_names(clause):
    if isinstance(clause, list):
        return clause
    elif isinstance(clause, str):
        return [clause]
    else:
        return []

class LayerParams(object):
    def __init__(self):
        """Create LayerParams for HugeCTR
        """
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

class HugeCTRLoader(object):
    def __init__(self, graph_config, dense_model, convert_embedding = False, sparse_models = None, ntp_file = None):
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
        self.__layers_config = json.load(open(graph_config, "rb"))["layers"]
        self.__layers = len(self.__layers_config)
        self.__index = 0
        self.__embedding_counter = 0
        if self.__ntp_file != None:
            self.__ntp_config = json.load(open(self.__ntp_file, "rb"))["layers"]
        else:
            self.__ntp_config = None
        self.__ntp_counter = 0
        self.__dimensions = {}
        self.__offset = 0
        self.__vocab_size_all_tables = 0
        self.__key_to_indice_hash_all_tables = None
        for i in range(self.layers):
            layer_config = self.__layers_config[i]
            layer_type = layer_config["type"]
            if layer_type == "DistributedSlotSparseEmbeddingHash" or layer_type == "LocalizedSlotSparseEmbeddingHash":
                max_vocab_size_global = layer_config["sparse_embedding_hparam"]["max_vocabulary_size_global"]
                self.__vocab_size_all_tables += max_vocab_size_global
        self.__key_to_indice_hash_all_tables = np.zeros(shape=(self.__vocab_size_all_tables,), dtype=np.int64)

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
            self.__dimensions[layer_params.label_name] = layer_params.label_dim
            self.__dimensions[layer_params.dense_name] = layer_params.dense_dim
            layer_weights_dict["key_to_indice_hash_all_tables"] = self.key_to_indice_hash_all_tables
        elif layer_type == "DistributedSlotSparseEmbeddingHash" or layer_type == "LocalizedSlotSparseEmbeddingHash":
            embedding_vec_size = layer_config["sparse_embedding_hparam"]["embedding_vec_size"]
            self.__dimensions[layer_config["top"]] = (self.__dimensions[layer_config["bottom"]][0], embedding_vec_size)
            if self.__convert_embeddding:
                layer_params.combiner = 0 if layer_config["sparse_embedding_hparam"]["combiner"] == "sum" else 1
                max_vocab_size_global = layer_config["sparse_embedding_hparam"]["max_vocabulary_size_global"]
                # indice 0 is reserved for default values of non-exisiting keys
                embedding_table = np.zeros(shape=(max_vocab_size_global + 1, embedding_vec_size), dtype=np.float32)
                with open(self.__sparse_models[self.__embedding_counter]+ "/key", 'rb') as key_file, \
                    open(self.__sparse_models[self.__embedding_counter]+ "/emb_vector", 'rb') as vec_file:
                    try:
                        # indice 0 is reserved for default values of non-exisiting keys
                        indice = 1
                        while True:
                            key_buffer = key_file.read(8)
                            vec_buffer = vec_file.read(4 * embedding_vec_size)
                            if len(key_buffer) == 0 or len(vec_buffer) == 0:
                                break
                            key = struct.unpack('q', key_buffer)[0]
                            values = struct.unpack(str(embedding_vec_size) + "f", vec_buffer)
                            self.key_to_indice_hash_all_tables[key] = indice
                            embedding_table[indice] = values
                            indice += 1
                    except BaseException as error:
                        print(error)
                layer_weights_dict["embedding_table"] = embedding_table
                self.__embedding_counter += 1
            else:
                print("Skip sparse embedding layers in converted ONNX model")
        elif layer_type == "Add":
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"][0]]
        elif layer_type == "BatchNorm":
            layer_params.factor = layer_config["bn_param"]["factor"]
            layer_params.eps = layer_config["bn_param"]["eps"]
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"]]
            in_feature =  self.__dimensions[layer_config["bottom"]]
            layer_bytes = in_feature * 4 * 2
            with open(self.__dense_model, 'rb') as file:
                file.seek(self.__offset, 0)
                buffer = file.read(layer_bytes)
                gamma = struct.unpack(str(in_feature) + "f", buffer[ : in_feature * 4])
                beta = struct.unpack(str(in_feature) + "f", buffer[in_feature * 4 : ])
                gamma = np.reshape(np.float32(gamma), newshape=(in_feature, ))
                beta = np.reshape(np.float32(beta), newshape=(in_feature, ))
            self.__offset += layer_bytes
            ntp_config = self.__ntp_config[self.__ntp_counter]
            running_mean = np.array(ntp_config["mean"], dtype = np.float32)
            running_variance = np.array(ntp_config["var"], dtype = np.float32)
            self.__ntp_counter += 1
            layer_weights_dict[layer_config["top"]+"_gamma"] = gamma
            layer_weights_dict[layer_config["top"]+"_beta"] = beta
            layer_weights_dict[layer_config["top"]+"_running_mean"] = running_mean
            layer_weights_dict[layer_config["top"]+"_running_variance"] = running_variance
        elif layer_type == "Concat":
            dim = 0
            for tensor in layer_config["bottom"]:
                dim += self.__dimensions[tensor]
            self.__dimensions[layer_config["top"]] = dim            
        elif layer_type == "Dropout":
            layer_params.dropout_rate = layer_config["rate"]
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"]]
        elif layer_type == "ElementwiseMultiply":
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"][0]]           
        elif layer_type == "ELU":
            layer_params.elu_alpha = layer_config["elu_param"]["alpha"]
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"]]
        elif layer_type == "FmOrder2":
            layer_params.out_dim = layer_config["out_dim"]
            self.__dimensions[layer_config["top"]] = layer_params.out_dim        
        elif layer_type == "InnerProduct" or layer_type == "FusedInnerProduct":
            layer_params.num_output = layer_config["fc_param"]["num_output"]
            self.__dimensions[layer_config["top"]] = layer_params.num_output
            in_feature = self.__dimensions[layer_config["bottom"]]
            out_feature = layer_params.num_output
            layer_bytes = (in_feature * out_feature + 1 * out_feature) * 4
            with open(self.__dense_model, 'rb') as file:
                file.seek(self.__offset, 0)
                buffer = file.read(layer_bytes)
                weight = struct.unpack(str(in_feature * out_feature) + "f", buffer[ : in_feature * out_feature * 4])
                bias = struct.unpack(str(out_feature) + "f", buffer[in_feature * out_feature * 4 : ])
                weight = np.reshape(np.float32(weight), newshape=(in_feature, out_feature))
                bias = np.reshape(np.float32(bias), newshape=(1, out_feature))
            self.__offset += layer_bytes
            layer_weights_dict[layer_config["top"]+"_weight"] = weight
            layer_weights_dict[layer_config["top"]+"_bias"] = bias
        elif layer_type == "FusedReshapeConcat":
            num_output = 0
            for tensor_name in layer_params.bottom_names:
                num_output += self.__dimensions[tensor_name][1]
            for tensor_name in layer_params.top_names:
                self.__dimensions[tensor_name] = num_output        
        elif layer_type == "Interaction":
            slot_num = self.__dimensions[layer_params.bottom_names[1]][0]
            vec_size = self.__dimensions[layer_params.bottom_names[1]][1]
            self.__dimensions[layer_config["top"]] = vec_size + (slot_num + 1) * (slot_num + 2 ) // 2 - (slot_num + 1) + 1        
        elif layer_type == "MatrixMultiply":
            dim1 = self.__dimensions[layer_params.bottom_names[0]]
            dim2 = self.__dimensions[layer_params.bottom_names[1]]
            if len(dim1) == 2:
                self.__dimensions[layer_config["top"]] = (dim1[0], dim2[1])
            else:
                self.__dimensions[layer_config["top"]] = dim2[1]
        elif layer_type == "MultiCross":
            layer_params.num_layers = layer_config["mc_param"]["num_layers"]
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"]]
            num_layers = layer_params.num_layers
            in_feature = self.__dimensions[layer_config["bottom"]]
            layer_bytes = in_feature * 2 * num_layers * 4
            with open(self.__dense_model, "rb") as file:
                file.seek(self.__offset, 0)
                buffer = file.read(layer_bytes)
                weights = []
                biases = []
                each_layer_bytes = layer_bytes // num_layers
                for i in range(num_layers):
                    weight = struct.unpack(str(in_feature) + "f", buffer[i*each_layer_bytes : i*each_layer_bytes + in_feature * 4])
                    bias = struct.unpack(str(in_feature) + "f", buffer[i*each_layer_bytes + in_feature * 4 : (i+1)*each_layer_bytes])
                    weights.append(np.reshape(np.float32(weight), newshape=(len(weight), 1)))
                    biases.append(np.reshape(np.float32(bias), newshape=(1, len(bias))))
            self.__offset += layer_bytes
            layer_weights_dict[layer_config["top"]+"_weights"] = weights
            layer_weights_dict[layer_config["top"]+"_biases"] = biases
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
                self.__dimensions[layer_params.top_names[0]] = tuple([dims[i] if i != axis_without_batch else 1 for i in range(len(dims))])
            else:
                dims = (self.__dimensions[layer_params.bottom_names[0]], )
                self.__dimensions[layer_params.top_names[0]] = 1        
        elif layer_type == "ReduceSum":
            # keepdims = 1, 0 < axis < N
            layer_params.axis = layer_config["axis"]
            axis_without_batch = layer_config["axis"] - 1
            if isinstance(self.__dimensions[layer_params.bottom_names[0]], tuple):
                dims = self.__dimensions[layer_params.bottom_names[0]]
                self.__dimensions[layer_params.top_names[0]] = tuple([dims[i] if i != axis_without_batch else 1 for i in range(len(dims))])
            else:
                dims = (self.__dimensions[layer_params.bottom_names[0]], )
                self.__dimensions[layer_params.top_names[0]] = 1        
        elif layer_type == "ReLU":
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"]]        
        elif layer_type == "Reshape":
            layer_params.selected_slots = layer_config.get("selected")
            layer_params.selected = layer_params.selected_slots is not None
            if not layer_params.selected:
                layer_params.leading_dim = layer_config["leading_dim"]
                layer_params.reshape_time_step =  0 if layer_config.get("time_step") is None else layer_config["time_step"]
            else:
                layer_params.leading_dim = len(layer_params.selected_slots) * self.__dimensions[layer_config["bottom"]][1]
            if layer_params.reshape_time_step == 0:
                self.__dimensions[layer_config["top"]] = layer_params.leading_dim
            else:
                self.__dimensions[layer_config["top"]] = (layer_params.reshape_time_step, layer_params.leading_dim)
        elif layer_type == "Scale":
            layer_params.scale_axis = layer_config["scale_param"]["axis"]
            layer_params.scale_factor = layer_config["scale_param"]["factor"]
            if layer_params.scale_axis == 0:
                self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"]] * int(layer_params.scale_factor)
            else: 
                self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"]]        
        elif layer_type == "Sigmoid":
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"]]        
        elif layer_type == "Slice":
            layer_params.ranges = layer_config["ranges"]
            for tensor, dim in zip(layer_config["top"], layer_params.ranges):
                self.__dimensions[tensor] = dim[1]-dim[0]
        elif layer_type == "Softmax":
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"]]        
        elif layer_type == "Sub":
            self.__dimensions[layer_config["top"]] = self.__dimensions[layer_config["bottom"][0]]        
        elif layer_type == "WeightMultiply":
            layer_params.weight_dims = layer_config["weight_dims"]
            self.__dimensions[layer_config["top"]] = layer_params.weight_dims[0] * layer_params.weight_dims[1]
            slot_num = layer_params.weight_dims[0]
            vec_size = layer_params.weight_dims[1]
            layer_bytes = slot_num * vec_size * 4
            with open(self.__dense_model, "rb") as file:
                file.seek(self.__offset, 0)
                buffer = file.read(layer_bytes)
                weight = struct.unpack(str(slot_num * vec_size) + "f", buffer[ : slot_num * vec_size * 4])
                weight = np.reshape(np.float32(weight), newshape=(slot_num, vec_size))
            self.__offset += layer_bytes
            layer_weights_dict[layer_config["top"]+"_weight"] = weight
        elif layer_type == "BinaryCrossEntropyLoss":
            layer_params.layer_type = "Sigmoid"
            pred_name = layer_params.bottom_names[0]
            layer_params.bottom_names = [pred_name]
            layer_params.top_names = []
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
            raise ValueError(layer_type + " is not supported in HugeCTR to ONNX converter, please refer to "
                            + "https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/onnx_converter#layer-support "
                            + "to see the supported layers.")
        self.__index += 1
        return layer_params, layer_weights_dict, self.dimensions 


      
