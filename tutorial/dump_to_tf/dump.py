# 
# Copyright (c) 2020, NVIDIA CORPORATION.
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


import struct
import os
import numpy as np
import json


class DumpToTF(object):

    def __init__(self, sparse_model_names, dense_model_name, 
                model_json, non_training_params_json = None):

        self.sparse_model_names = sparse_model_names # list of strings
        self.dense_model_name = dense_model_name # string
        self.model_json = model_json # json file of the whole model
        self.non_training_params_json = non_training_params_json # non training params

        self.model_content = None
        self.embedding_layers = None
        self.dense_layers = None
        self.gpus = [0]
        self.key_type = 'I32' # {'I64', 'I32'}, default is 'I32'
        self.key_type_map = {"I32": ["I", 4], "I64": ["q", 8]}

        self.parse_json()

        self.offset = 0

    def parse_json(self):
        """
        parse the model json file to get the layers of the whole model.
        save in a list.
        then parse the non training params json file. to get the non-training-parameters.
        #returns:
            [embedding_layer, dense_layer0, dense_layer1, ...], 
            [non-training-params]
        """
        print("[INFO] begin to parse model json file: %s" %self.model_json)

        try:
            with open(self.model_json, 'r') as model_json:
                self.model_content = json.load(model_json)

                self.gpus = self.model_content["solver"]["gpu"]
                key_type = self.model_content["solver"].get("input_key_type")
                if key_type is not None:
                    self.key_type = key_type                
                
                layers = self.model_content["layers"]
                # embedding_layers
                self.embedding_layers = []
                for index in range(1, len(layers)):
                    if layers[index]["type"] not in ["DistributedSlotSparseEmbeddingHash",
                                                    "LocalizedSlotSparseEmbeddingHash"]:
                        break
                    else:
                        self.embedding_layers.append(layers[index])

                #dense layers
                self.dense_layers = layers[1 + len(self.embedding_layers): ]

        except BaseException as error:
            print(error)


    def parse_embedding(self):
        """
        get one embedding table at a time.
        """
        if self.model_content is None:
            self.parse_json()

        for index, layer in enumerate(self.embedding_layers):
            print("[INFO] begin to parse embedding weights: %s" %layer["name"])

            each_key_size = 0
            layer_type = layer["type"]
            embedding_vec_size = layer["sparse_embedding_hparam"]["embedding_vec_size"]
            max_vocab_size_per_gpu = layer["sparse_embedding_hparam"]["max_vocabulary_size_per_gpu"]

            vocabulary_size = max_vocab_size_per_gpu * len(self.gpus)

            if layer_type == "DistributedSlotSparseEmbeddingHash":
                # sizeof(TypeHashKey) + sizeof(float) * embedding_vec_size
                each_key_size = self.key_type_map[self.key_type][1] + 4 * embedding_vec_size

            elif layer_type == "LocalizedSlotSparseEmbeddingHash":
                # sizeof(TypeHashKey) + sizeof(TypeHashValueIndex) + sizeof(float) * embedding_vec_size
                each_key_size = self.key_type_map[self.key_type][1] + self.key_type_map[self.key_type][1] + 4 * embedding_vec_size

            embedding_table = np.zeros(shape=(vocabulary_size, embedding_vec_size), dtype=np.float32)

            with open(self.sparse_model_names[index], 'rb') as file:
                try:
                    while True:
                        buffer = file.read(each_key_size)
                        if len(buffer) == 0:
                            break
                        
                        if layer_type == "DistributedSlotSparseEmbeddingHash":
                            key = struct.unpack(self.key_type_map[self.key_type][0], buffer[0: self.key_type_map[self.key_type][1]])
                            values = struct.unpack(str(embedding_vec_size) + "f", buffer[self.key_type_map[self.key_type][1]: ])

                        elif layer_type == "LocalizedSlotSparseEmbeddingHash":
                            key, slot_id = struct.unpack("2" + self.key_type_map[self.key_type][0], 
                                                         buffer[0: 2*self.key_type_map[self.key_type][1]])
                            values = struct.unpack(str(embedding_vec_size) + "f", buffer[self.key_type_map[self.key_type][1]: ])

                        embedding_table[key] = values

                except BaseException as error:
                    print(error)

            yield layer["name"], embedding_table


    def parse_dense(self, layer_bytes, layer_type, **kwargs):
        """
        get one layer weights at a time.
        """
        if self.model_content is None:
            self.parse_json()
            self.offset = 0

        with open(self.dense_model_name, 'rb') as file:
            print("[INFO] begin to parse dense weights: %s" %layer_type)

            file.seek(self.offset, 0)

            buffer = file.read(layer_bytes)

            if layer_type == "BatchNorm":
                # TODO
                pass
            elif layer_type == "InnerProduct":
                in_feature = kwargs["in_feature"]
                out_feature = kwargs["out_feature"]

                weight = struct.unpack(str(in_feature * out_feature) + "f", buffer[ : in_feature * out_feature * 4])
                bias = struct.unpack(str(out_feature) + "f", buffer[in_feature * out_feature * 4 : ])

                weight = np.reshape(np.float32(weight), newshape=(in_feature, out_feature))
                bias = np.reshape(np.float32(bias), newshape=(1, out_feature))

                self.offset += layer_bytes
                return weight, bias


            elif layer_type == "MultiCross":
                vec_length = kwargs["vec_length"]
                num_layers = kwargs["num_layers"]

                weights = []
                biases = []

                each_layer_bytes = layer_bytes // num_layers

                for i in range(num_layers):
                    weight = struct.unpack(str(vec_length) + "f", buffer[i*each_layer_bytes : i*each_layer_bytes + vec_length * 4])
                    bias = struct.unpack(str(vec_length) + "f", buffer[i*each_layer_bytes + vec_length * 4 : (i+1)*each_layer_bytes])

                    weights.append(np.reshape(np.float32(weight), newshape=(1, len(weight))))
                    biases.append(np.reshape(np.float32(bias), newshape=(1, len(bias))))

                self.offset += layer_bytes

                return weights, biases

            elif layer_type == "Multiply":
                # TODO
                pass

    def read_dense_complete(self):
        if self.offset == os.path.getsize(self.dense_model_name):
            print("[INFO] all dense weights has been parsed.")
        else:
            print("[INFO] not all dense weights has been parsed.")

    def build_graph(self):
        """
        build computing-graph with tf according to model json file.
        """
        pass

    def save_to_checkpoint(self):
        """
        save the computing-graph with the loading weights into a tf checkpoint.
        """
        pass

    def get_key_type(self):
        return self.key_type


if __name__ == "__main__":
    samples_dir = r'/workspace/hugectr/samples/'
    model_json = os.path.join(samples_dir, r'dcn/dcn.json')

    model_path = r'./hugectr_model_file/'
    sparse_model_names = [os.path.join(model_path, r'dcn_model0_sparse_2000.model')]
    dense_model_name = os.path.join(model_path, r'dcn_model_dense_2000.model')

    test_dump = DumpToTF(sparse_model_names = sparse_model_names,
                         dense_model_name = dense_model_name,
                         model_json = model_json,
                         non_training_params_json = None)

    embeddings = test_dump.parse_embedding().__next__()
    print(embeddings)
    # name, weights = embeddings.__next__()
    # print(name)
    # print(weights)

    # for name, weights in test_dump.parse_embedding():
    #     print(name)
    #     for row in range(weights.shape[0]):
    #         print(row, " : ", weights[row])
