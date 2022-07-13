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
from onnx import AttributeProto, TensorProto, GraphProto, helper, numpy_helper
from hugectr2onnx.hugectr_loader import HugeCTRLoader, LayerParams
import numpy as np
import onnx

class GraphBuilder(object):
    def __init__(self, convert_embedding):
        """Create GraphBuilder
        Args:
            convert_embedding: boolean, whether converting sparse embedding models to ONNX
        """
        self.__convert_embeddding = convert_embedding
        self.__nodes = []
        self.__initializers = []
        self.__inputs = []
        self.__outputs = []
        self.__counter = 0

    def add_layer(self, layer_params, weights_dict, dimensions):
        """Add layer to ONNX graph, one layer may consist of multiple ONNX nodes
        Args:
            layer_params: HugeCTR layer parameters
            weights_dict: weights dictionary for HugeCTR trainable layer
            dimensions: dimension information of previous tensors
        """
        layer_type = layer_params.layer_type
        if layer_type == "Data":
            # Create input (ValueInfoProto)
            self.__inputs.append(helper.make_tensor_value_info(layer_params.dense_name, TensorProto.FLOAT, [None, layer_params.dense_dim]))
            if self.__convert_embeddding:
                for sparse_name, sparse_dim in zip(layer_params.sparse_names, layer_params.sparse_dims):
                    self.__inputs.append(helper.make_tensor_value_info(sparse_name, TensorProto.INT64, [None, sparse_dim[0], sparse_dim[1]]))
            # Create output (ValueInfoProto)
            self.__outputs.append(helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, layer_params.label_dim]))
            self.__key_to_indice_hash_all_tables = weights_dict["key_to_indice_hash_all_tables"]
            key_to_indice_hash_all_tables = weights_dict["key_to_indice_hash_all_tables"]
            key_to_indice_hash_all_tables_name = 'key_to_indice_hash_all_tables'
            self.__initializers.append(helper.make_tensor(name=key_to_indice_hash_all_tables_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[key_to_indice_hash_all_tables.dtype],
                                                        dims=key_to_indice_hash_all_tables.shape,
                                                        vals=key_to_indice_hash_all_tables.flatten()))
        elif layer_type == "DistributedSlotSparseEmbeddingHash" or layer_type == "LocalizedSlotSparseEmbeddingHash":
            if self.__convert_embeddding:
                embedding_table = weights_dict["embedding_table"]
                embedding_table_name = layer_params.top_names[0] + '_embedding_table'
                indice_name = layer_params.top_names[0] + '_indice'
                embedding_feature_name = layer_params.top_names[0] + '_embedding_feature'
                self.__initializers.append(helper.make_tensor(name=embedding_table_name,
                                                            data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[embedding_table.dtype],
                                                            dims=embedding_table.shape,
                                                            vals=embedding_table.flatten()))
                self.__nodes.append(helper.make_node(op_type = 'Gather',
                                                    inputs=['key_to_indice_hash_all_tables', layer_params.bottom_names[0]],
                                                    outputs=[indice_name],
                                                    axis=0))                                                                                             
                self.__nodes.append(helper.make_node(op_type = 'Gather',
                                                    inputs=[embedding_table_name, indice_name],
                                                    outputs=[embedding_feature_name],
                                                    axis=0))
                reduce_type = 'ReduceSum' if layer_params.combiner == 0 else 'ReduceMean'
                self.__nodes.append(helper.make_node(op_type = reduce_type,
                                                    inputs=[embedding_feature_name],
                                                    outputs=layer_params.top_names,
                                                    keepdims = 0,
                                                    axes=[-2]))
            else:
                self.__inputs.append(helper.make_tensor_value_info(layer_params.top_names[0], TensorProto.FLOAT, 
                                                                [None, dimensions[layer_params.top_names[0]][0], dimensions[layer_params.top_names[0]][1]]))
        elif layer_type == "Add":
            for i in range(len(layer_params.bottom_names)-1):
                x_name = layer_params.bottom_names[0] if i == 0 else layer_params.top_names[0] + str(i)
                y_name = layer_params.bottom_names[i+1]
                z_name = layer_params.top_names[0] if i == len(layer_params.bottom_names)-2 else layer_params.top_names[0] + str(i+1)
                self.__nodes.append(helper.make_node(op_type = 'Add',
                                                    inputs=[x_name, y_name],
                                                    outputs=[z_name]))
        elif layer_type == "BatchNorm":
            gamma_name = layer_params.top_names[0] + "_gamma"
            beta_name = layer_params.top_names[0] + "_beta"
            running_mean_name = layer_params.top_names[0] + "_running_mean"
            running_variance_name = layer_params.top_names[0] + "_running_variance"
            gamma = weights_dict[gamma_name]
            beta = weights_dict[beta_name]
            running_mean = weights_dict[running_mean_name]
            running_variance = weights_dict[running_variance_name]
            self.__initializers.append(helper.make_tensor(name=gamma_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[gamma.dtype],
                                                        dims=gamma.shape,
                                                        vals=gamma.flatten()))
            self.__initializers.append(helper.make_tensor(name=beta_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[beta.dtype],
                                                        dims=beta.shape,
                                                        vals=beta.flatten()))
            self.__initializers.append(helper.make_tensor(name=running_mean_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[running_mean.dtype],
                                                        dims=running_mean.shape,
                                                        vals=running_mean.flatten()))
            self.__initializers.append(helper.make_tensor(name=running_variance_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[running_variance.dtype],
                                                        dims=running_variance.shape,
                                                        vals=running_variance.flatten()))
            self.__nodes.append(helper.make_node(op_type = 'BatchNormalization',
                                                inputs=[layer_params.bottom_names[0], gamma_name, beta_name, running_mean_name, running_variance_name],
                                                outputs=layer_params.top_names,
                                                epsilon = layer_params.eps,
                                                momentum = layer_params.factor))
        elif layer_type == "LayerNorm":
            gamma_name = layer_params.top_names[0] + "_gamma"
            beta_name = layer_params.top_names[0] + "_beta"
            #running_mean_name = layer_params.top_names[0] + "_running_mean"
            #running_variance_name = layer_params.top_names[0] + "_running_variance"
            gamma = weights_dict[gamma_name]
            beta = weights_dict[beta_name]
            #running_mean = weights_dict[running_mean_name]
            #running_variance = weights_dict[running_variance_name]
            self.__initializers.append(helper.make_tensor(name=gamma_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[gamma.dtype],
                                                        dims=gamma.shape,
                                                        vals=gamma.flatten()))
            self.__initializers.append(helper.make_tensor(name=beta_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[beta.dtype],
                                                        dims=beta.shape,
                                                        vals=beta.flatten()))
            self.__nodes.append(helper.make_node(op_type = 'LayerNormalization',
                                                inputs=[layer_params.bottom_names[0], gamma_name, beta_name],
                                                outputs=layer_params.top_names,
                                                epsilon = layer_params.eps))

        elif layer_type == "Concat":
            self.__nodes.append(helper.make_node(op_type = 'Concat',
                                                inputs=layer_params.bottom_names,
                                                outputs=layer_params.top_names,
                                                axis = layer_params.axis))
        elif layer_type == "ElementwiseMultiply":
            for i in range(len(layer_params.bottom_names)-1):
                x_name = layer_params.bottom_names[0] if i == 0 else layer_params.top_names[0] + str(i)
                y_name = layer_params.bottom_names[i+1]
                z_name = layer_params.top_names[0] if i == len(layer_params.bottom_names)-2 else layer_params.top_names[0] + str(i+1)
                self.__nodes.append(helper.make_node(op_type = 'Mul',
                                                    inputs=[x_name, y_name],
                                                    outputs=[z_name]))
        elif layer_type == "Dropout":
            self.__nodes.append(helper.make_node(op_type = 'Dropout',
                                                inputs=layer_params.bottom_names,
                                                outputs=layer_params.top_names))
        elif layer_type == "ELU":
            self.__nodes.append(helper.make_node(op_type = 'Elu',
                                                inputs=layer_params.bottom_names,
                                                outputs=layer_params.top_names,
                                                alpha = layer_params.elu_alpha))
        elif layer_type == "FmOrder2":
            vec_size = layer_params.out_dim
            slot_num = dimensions[layer_params.bottom_names[0]] // vec_size
            shape_name = layer_params.top_names[0] + "_shape"
            shape = np.array([-1, slot_num, vec_size], dtype=np.int64)
            reshape_name = layer_params.top_names[0] + "_reshape"
            reduce_sum_name = layer_params.top_names[0] + "_reduce_sum"
            sum_square_name = layer_params.top_names[0] + "_sum_square"
            square_name = layer_params.top_names[0] + "_square"
            square_sum_name = layer_params.top_names[0] + "_square_sum"
            sub_name = layer_params.top_names[0] + "_sub"
            scalar_name = layer_params.top_names[0] + "_scalar"
            scalar = np.array([0.5], dtype=np.float32)
            self.__initializers.append(helper.make_tensor(name=shape_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape.dtype],
                                                        dims=shape.shape,
                                                        vals=shape.flatten()))
            self.__initializers.append(helper.make_tensor(name=scalar_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[scalar.dtype],
                                                        dims=scalar.shape,
                                                        vals=scalar.flatten()))
            self.__nodes.append(helper.make_node(op_type = 'Reshape',
                                                inputs=[layer_params.bottom_names[0], shape_name],
                                                outputs=[reshape_name]))
            self.__nodes.append(helper.make_node(op_type = 'ReduceSum',
                                                inputs=[reshape_name],
                                                outputs=[reduce_sum_name],
                                                keepdims = 0,
                                                axes = [-2]))
            self.__nodes.append(helper.make_node(op_type = 'Mul',
                                                inputs=[reduce_sum_name, reduce_sum_name],
                                                outputs=[sum_square_name]))
            self.__nodes.append(helper.make_node(op_type = 'Mul',
                                                inputs=[reshape_name, reshape_name],
                                                outputs=[square_name]))
            self.__nodes.append(helper.make_node(op_type = 'ReduceSum',
                                                inputs=[square_name],
                                                outputs=[square_sum_name],
                                                keepdims = 0,
                                                axes = [-2]))
            self.__nodes.append(helper.make_node(op_type = 'Sub',
                                                inputs=[sum_square_name, square_sum_name],
                                                outputs=[sub_name]))
            self.__nodes.append(helper.make_node(op_type = 'Mul',
                                                inputs=[sub_name, scalar_name],
                                                outputs=layer_params.top_names))
        elif layer_type == "InnerProduct":
            weight_name = layer_params.top_names[0]+"_weight"
            bias_name =  layer_params.top_names[0]+"_bias"
            weight = weights_dict[weight_name]
            bias = weights_dict[bias_name]
            if isinstance(dimensions[layer_params.bottom_names[0]], tuple):
                dim = dimensions[layer_params.bottom_names[0]]
                shape_name1 = layer_params.top_names[0] + "_shape1"
                shape1 = np.array([-1, dim[0]*dim[1] ], dtype=np.int64)

                shape_name2 = layer_params.top_names[0] + "_shape2"
                shape2 = np.array([-1, dim[0], layer_params.num_output], dtype=np.int64)
            else:
                shape_name1 = layer_params.top_names[0] + "_shape1"
                shape1 = np.array([-1, dimensions[layer_params.bottom_names[0]]], dtype=np.int64)

                shape_name2 = layer_params.top_names[0] + "_shape2"
                shape2 = np.array([-1, layer_params.num_output], dtype=np.int64)

            reshape_name1 = layer_params.bottom_names[0] + "_reshape_fc1"
            reshape_name2 = layer_params.top_names[0] + "_reshape_fc2"

            self.__initializers.append(helper.make_tensor(name=shape_name1,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape1.dtype],
                                                        dims=shape1.shape,
                                                        vals=shape1.flatten()))
            self.__initializers.append(helper.make_tensor(name=shape_name2,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape2.dtype],
                                                        dims=shape2.shape,
                                                        vals=shape2.flatten()))

            self.__initializers.append(helper.make_tensor(name=weight_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[weight.dtype],
                                                        dims=weight.shape,
                                                        vals=weight.flatten()))
            self.__initializers.append(helper.make_tensor(name=bias_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[bias.dtype],
                                                        dims=bias.shape,
                                                        vals=bias.flatten()))
            self.__nodes.append(helper.make_node(op_type = 'Reshape',
                                                inputs=[layer_params.bottom_names[0], shape_name1],
                                                outputs=[reshape_name1]))
            self.__nodes.append(helper.make_node(op_type = 'Gemm',
                                                inputs=[reshape_name1, weight_name, bias_name],
                                                outputs=[reshape_name2]))
            self.__nodes.append(helper.make_node(op_type = 'Reshape',
                                                inputs=[reshape_name2, shape_name2],
                                                outputs=layer_params.top_names))
        elif layer_type == "FusedInnerProduct":
            weight_name = layer_params.top_names[0]+"_weight"
            bias_name =  layer_params.top_names[0]+"_bias"
            weight = weights_dict[weight_name]
            bias = weights_dict[bias_name]
            gemm_name = layer_params.top_names[0]+"_gemm"
            self.__initializers.append(helper.make_tensor(name=weight_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[weight.dtype],
                                                        dims=weight.shape,
                                                        vals=weight.flatten()))
            self.__initializers.append(helper.make_tensor(name=bias_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[bias.dtype],
                                                        dims=bias.shape,
                                                        vals=bias.flatten()))
            self.__nodes.append(helper.make_node(op_type = 'Gemm',
                                                inputs=[layer_params.bottom_names[0], weight_name, bias_name],
                                                outputs=[gemm_name]))
            self.__nodes.append(helper.make_node(op_type = 'Relu',
                                                inputs=[gemm_name],
                                                outputs=layer_params.top_names))
        elif layer_type == "FusedReshapeConcat":
            slot_num = dimensions[layer_params.bottom_names[0]][0]
            output_fea_num = 0
            item_starts_name = layer_params.top_names[0] + "_item_start"
            item_ends_name = layer_params.top_names[0] + "_item_end"
            ad_starts_name = layer_params.top_names[0] + "_ad_start"
            ad_ends_name = layer_params.top_names[0] + "_ad_end"
            axes_name = layer_params.top_names[0] + "_axes"
            item_starts = np.array([0], dtype=np.int64)
            item_ends = np.array([slot_num-1], dtype=np.int64)
            ad_starts = np.array([slot_num-1], dtype=np.int64)
            ad_ends = np.array([slot_num], dtype=np.int64)
            axes = np.array([1], dtype=np.int64)
            self.__initializers.append(helper.make_tensor(name=item_starts_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[item_starts.dtype],
                                                        dims=item_starts.shape,
                                                        vals=item_starts.flatten()))
            self.__initializers.append(helper.make_tensor(name=item_ends_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[item_ends.dtype],
                                                        dims=item_ends.shape,
                                                        vals=item_ends.flatten()))
            self.__initializers.append(helper.make_tensor(name=ad_starts_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[ad_starts.dtype],
                                                        dims=ad_starts.shape,
                                                        vals=ad_starts.flatten()))
            self.__initializers.append(helper.make_tensor(name=ad_ends_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[ad_ends.dtype],
                                                        dims=ad_ends.shape,
                                                        vals=ad_ends.flatten()))
            self.__initializers.append(helper.make_tensor(name=axes_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[axes.dtype],
                                                        dims=axes.shape,
                                                        vals=axes.flatten()))
            for i in range(len(layer_params.bottom_names)):
                input_tensor_name = layer_params.bottom_names[i]
                output_fea_num += dimensions[input_tensor_name][1]
                item_slice_name = input_tensor_name + "_slice_item" + str(i)
                ad_slice_name = input_tensor_name + "_slice_ad" + str(i)
                item_concat_name = input_tensor_name + "_concat_item" + str(i)
                ad_concat_name = input_tensor_name + "_concat_ad" + str(i)
                self.__nodes.append(helper.make_node(op_type = 'Slice',
                                                    inputs=[input_tensor_name, item_starts_name, item_ends_name, axes_name],
                                                    outputs=[item_slice_name]))
                self.__nodes.append(helper.make_node(op_type = 'Slice',
                                                    inputs=[input_tensor_name, ad_starts_name, ad_ends_name, axes_name],
                                                    outputs=[ad_slice_name]))
                if i > 0:
                    prev_item_concat_name = layer_params.bottom_names[i-1] + "_slice_item" + str(i-1) if i == 1 else layer_params.bottom_names[i-1] + "_concat_item" + str(i-1)
                    prev_ad_concat_name = layer_params.bottom_names[i-1] + "_slice_ad" + str(i-1) if i == 1 else layer_params.bottom_names[i-1] + "_concat_ad" + str(i-1)
                    self.__nodes.append(helper.make_node(op_type = 'Concat',
                                                        inputs=[prev_item_concat_name, item_slice_name],
                                                        outputs=[item_concat_name],
                                                        axis = 2))
                    self.__nodes.append(helper.make_node(op_type = 'Concat',
                                                        inputs=[prev_ad_concat_name, ad_slice_name],
                                                        outputs=[ad_concat_name],
                                                        axis = 2))
            shape_name = layer_params.top_names[0] + "_item_shape"
            shape = np.array([-1, output_fea_num], dtype=np.int64)
            num_input_tensors = len(layer_params.bottom_names)
            last_bottom_tensor_name = layer_params.bottom_names[num_input_tensors-1]
            last_item_tensor_name =  last_bottom_tensor_name + "_slice_item" + str(num_input_tensors-1) if num_input_tensors == 1 else last_bottom_tensor_name + "_concat_item" + str(num_input_tensors-1)
            last_ad_tensor_name = last_bottom_tensor_name + "_slice_ad" + str(num_input_tensors-1) if num_input_tensors == 1 else last_bottom_tensor_name + "_concat_ad" + str(num_input_tensors-1)
            self.__initializers.append(helper.make_tensor(name=shape_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape.dtype],
                                                        dims=shape.shape,
                                                        vals=shape.flatten()))
            self.__nodes.append(helper.make_node(op_type = 'Reshape',
                                                inputs=[last_item_tensor_name, shape_name],
                                                outputs=[layer_params.top_names[0]]))
            self.__nodes.append(helper.make_node(op_type = 'Reshape',
                                                inputs=[last_ad_tensor_name, shape_name],
                                                outputs=[layer_params.top_names[1]]))
        elif layer_type == "MultiHeadAttention":
            query_name = layer_params.bottom_names[0]
            key_name = layer_params.bottom_names[1]
            transpose_name = key_name + "_transpose"
            self.__nodes.append(helper.make_node(op_type = 'Transpose',
                                                inputs=[query_name],
                                                outputs=[transpose_name],
                                                perm=[0,1,3,2]))
            self.__nodes.append(helper.make_node(op_type = 'MatMul',
                                                inputs=[query_name, transpose_name],
                                                outputs=layer_params.top_names))
        elif layer_type == "Interaction":
            slot_num = dimensions[layer_params.bottom_names[1]][0]
            vec_size = dimensions[layer_params.bottom_names[1]][1]
            out_dim = dimensions[layer_params.top_names[0]]
            mlp_name = layer_params.bottom_names[0]
            emb_name = layer_params.bottom_names[1]
            shape_name1 = layer_params.top_names[0] + "_shape1"
            shape1 = np.array([-1, 1, vec_size], dtype=np.int64)
            reshape_name1 = layer_params.top_names[0] + "_reshape1"
            concat_name = layer_params.top_names[0] + "_concat"
            transpose_name = layer_params.top_names[0] + "_transpose"
            matmul_name = layer_params.top_names[0] + "_matmul"
            shape_name2 = layer_params.top_names[0] + "_shape2"
            shape2 = np.array([-1, (slot_num+1)*(slot_num+1)], dtype=np.int64)
            reshape_name2 = layer_params.top_names[0] + "_reshape2"
            indices_name = layer_params.top_names[0] + "_indices"
            indices = [i*(slot_num+1)+j for j in range(1, 1+slot_num) for i in range(j)]
            indices = np.array(indices, dtype=np.int64)
            gather_name = layer_params.top_names[0] + "_gather"
            interaction_name = layer_params.top_names[0] + "_interaction"
            self.__initializers.append(helper.make_tensor(name=shape_name1,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape1.dtype],
                                                        dims=shape1.shape,
                                                        vals=shape1.flatten()))
            self.__initializers.append(helper.make_tensor(name=shape_name2,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape2.dtype],
                                                        dims=shape2.shape,
                                                        vals=shape2.flatten()))
            self.__initializers.append(helper.make_tensor(name=indices_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[indices.dtype],
                                                        dims=indices.shape,
                                                        vals=indices.flatten()))
            self.__nodes.append(helper.make_node(op_type = 'Reshape',
                                                inputs=[mlp_name, shape_name1],
                                                outputs=[reshape_name1]))
            self.__nodes.append(helper.make_node(op_type = 'Concat',
                                                inputs=[reshape_name1, emb_name],
                                                outputs=[concat_name],
                                                axis = -2))
            self.__nodes.append(helper.make_node(op_type = 'Transpose',
                                                inputs=[concat_name],
                                                outputs=[transpose_name],
                                                perm=[0,2,1]))
            self.__nodes.append(helper.make_node(op_type = 'MatMul',
                                                inputs=[concat_name, transpose_name],
                                                outputs=[matmul_name]))
            self.__nodes.append(helper.make_node(op_type = 'Reshape',
                                                inputs=[matmul_name, shape_name2],
                                                outputs=[reshape_name2]))
            self.__nodes.append(helper.make_node(op_type = 'Gather',
                                                inputs=[reshape_name2, indices_name],
                                                outputs=[gather_name],
                                                axis=1))
            self.__nodes.append(helper.make_node(op_type = 'Concat',
                                                inputs=[mlp_name, gather_name],
                                                outputs=[interaction_name],
                                                axis = -1))
            self.__nodes.append(helper.make_node(op_type = 'Pad',
                                                inputs=[interaction_name],
                                                outputs=layer_params.top_names,
                                                pads = [0, 0, 0, 1]))
        elif layer_type == "MatrixMultiply":
            self.__nodes.append(helper.make_node(op_type = 'MatMul',
                                                inputs=layer_params.bottom_names,
                                                outputs=layer_params.top_names))
        elif layer_type == "MultiCross":
            weights_name = layer_params.top_names[0]+"_weights"
            biases_name = layer_params.top_names[0]+"_biases"
            weights = weights_dict[weights_name]
            biases = weights_dict[biases_name]
            for i in range(layer_params.num_layers):
                weight_name = weights_name + str(i)
                bias_name = biases_name + str(i)
                weight = weights[i]
                bias = biases[i]
                feed_name = layer_params.bottom_names[0] if i == 0 else layer_params.top_names[0] + '_multicross' + str(i)
                matmul_name = layer_params.top_names[0] + "_matmul" + str(i+1)
                multiply_name = layer_params.top_names[0] + "_multiply" + str(i+1)
                add_name = layer_params.top_names[0] + "_add" + str(i+1)
                multicross_name = layer_params.top_names[0] if i == layer_params.num_layers-1 else layer_params.top_names[0] + '_multicross' + str(i+1)
                self.__initializers.append(helper.make_tensor(name=weight_name,
                                                            data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[weight.dtype],
                                                            dims=weight.shape,
                                                            vals=weight.flatten()))
                self.__initializers.append(helper.make_tensor(name=bias_name,
                                                            data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[bias.dtype],
                                                            dims=bias.shape,
                                                            vals=bias.flatten()))
                self.__nodes.append(helper.make_node(op_type = 'MatMul',
                                                    inputs=[feed_name, weight_name],
                                                    outputs=[matmul_name]))
                self.__nodes.append(helper.make_node(op_type = 'Mul',
                                                    inputs=[layer_params.bottom_names[0], matmul_name],
                                                    outputs=[multiply_name]))
                self.__nodes.append(helper.make_node(op_type = 'Add',
                                                    inputs=[feed_name, bias_name],
                                                    outputs=[add_name]))
                self.__nodes.append(helper.make_node(op_type = 'Add',
                                                    inputs=[multiply_name, add_name],
                                                    outputs=[multicross_name]))
        elif layer_type == "PReLU_Dice":
            mean_name = layer_params.top_names[0] + "_mean"
            square_name = layer_params.top_names[0] + "_square"
            square_mean_name = layer_params.top_names[0] + "_square_mean"
            mean_square_name = layer_params.top_names[0] + "_mean_square"
            var_name = layer_params.top_names[0] + "_var"
            epsilon_name = layer_params.top_names[0] + "_eps"
            epsilon = np.array([layer_params.prelu_eps], dtype=np.float32)
            var_epsilon_name = layer_params.top_names[0] + "_var_eps"
            var_epsilon_sqrt_name = layer_params.top_names[0] + "_var_eps_sqrt"
            diff_name = layer_params.top_names[0] + "_diff"
            value_name = layer_params.top_names[0] + "_value"
            ps_name = layer_params.top_names[0] + "_ps"
            alpha_name = layer_params.top_names[0] + "_alpha"
            alpha = np.array([layer_params.prelu_alpha], dtype=np.float32)
            first_item_name = layer_params.top_names[0] + "_first_item"
            ps_mul_name = layer_params.top_names[0] + "_ps_mul"
            second_item_tmp_name = layer_params.top_names[0] + "_second_item_tmp"
            second_item_name = layer_params.top_names[0] + "_second_item"
            self.__initializers.append(helper.make_tensor(name=epsilon_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[epsilon.dtype],
                                                        dims=epsilon.shape,
                                                        vals=epsilon.flatten()))
            self.__initializers.append(helper.make_tensor(name=alpha_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[alpha.dtype],
                                                        dims=alpha.shape,
                                                        vals=alpha.flatten()))
            self.__nodes.append(helper.make_node(op_type = 'ReduceMean',
                                                inputs=layer_params.bottom_names,
                                                outputs=[mean_name],
                                                axes = [0]))
            self.__nodes.append(helper.make_node(op_type = 'Mul',
                                                inputs=[mean_name, mean_name],
                                                outputs=[mean_square_name]))
            self.__nodes.append(helper.make_node(op_type = 'Mul',
                                                inputs=[layer_params.bottom_names[0], layer_params.bottom_names[0]],
                                                outputs=[square_name]))
            self.__nodes.append(helper.make_node(op_type = 'ReduceMean',
                                            inputs=[square_name],
                                            outputs=[square_mean_name],
                                            axes = [0]))
            self.__nodes.append(helper.make_node(op_type = 'Sub',
                                                inputs=[square_mean_name, mean_square_name],
                                                outputs=[var_name]))
            self.__nodes.append(helper.make_node(op_type = 'Add',
                                                inputs=[var_name, epsilon_name],
                                                outputs=[var_epsilon_name]))
            self.__nodes.append(helper.make_node(op_type = 'Sqrt',
                                                inputs=[var_epsilon_name],
                                                outputs=[var_epsilon_sqrt_name]))
            self.__nodes.append(helper.make_node(op_type = 'Sub',
                                                inputs=[layer_params.bottom_names[0], mean_name],
                                                outputs=[diff_name]))
            self.__nodes.append(helper.make_node(op_type = 'Div',
                                                inputs=[diff_name, var_epsilon_sqrt_name],
                                                outputs=[value_name]))
            self.__nodes.append(helper.make_node(op_type = 'Sigmoid',
                                                inputs=[value_name],
                                                outputs=[ps_name]))
            self.__nodes.append(helper.make_node(op_type = 'Mul',
                                                inputs=[layer_params.bottom_names[0], ps_name],
                                                outputs=[first_item_name]))
            self.__nodes.append(helper.make_node(op_type = 'Mul',
                                    inputs=[layer_params.bottom_names[0], ps_name],
                                    outputs=[ps_mul_name]))
            self.__nodes.append(helper.make_node(op_type = 'Sub',
                                    inputs=[ps_mul_name, layer_params.bottom_names[0]],
                                    outputs=[second_item_tmp_name]))
            self.__nodes.append(helper.make_node(op_type = 'Mul',
                                    inputs=[second_item_tmp_name, alpha_name],
                                    outputs=[second_item_name]))
            self.__nodes.append(helper.make_node(op_type = 'Sub',
                                    inputs=[first_item_name, second_item_name],
                                    outputs=layer_params.top_names))
        elif layer_type == "ReduceMean":
            self.__nodes.append(helper.make_node(op_type = 'ReduceMean',
                                                inputs=layer_params.bottom_names,
                                                outputs=layer_params.top_names,
                                                keepdims = 1,
                                                axes = [layer_params.axis]))
        elif layer_type == "ReduceSum":
            self.__nodes.append(helper.make_node(op_type = 'ReduceSum',
                                                inputs=layer_params.bottom_names,
                                                outputs=layer_params.top_names,
                                                keepdims = 1,
                                                axes = [layer_params.axis]))
        elif layer_type == "ReLU":
            self.__nodes.append(helper.make_node(op_type = 'Relu',
                                                inputs=layer_params.bottom_names,
                                                outputs=layer_params.top_names))                                                                
        elif layer_type == "Reshape":
            shape_name = layer_params.top_names[0] + "_shape"
            if layer_params.reshape_time_step == 0:
                shape = np.array([-1, layer_params.leading_dim], dtype=np.int64)
            else:
                shape = np.array([-1, layer_params.reshape_time_step, layer_params.leading_dim], dtype=np.int64)
            self.__initializers.append(helper.make_tensor(name=shape_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape.dtype],
                                                        dims=shape.shape,
                                                        vals=shape.flatten()))
            if layer_params.selected:
                gather_name = layer_params.top_names[0] + "_gather"
                selected_slots_name = layer_params.top_names[0] + "_selected_slots"
                selected_slots = np.array(layer_params.selected_slots, dtype=np.int64)
                self.__initializers.append(helper.make_tensor(name=selected_slots_name,
                                                            data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[selected_slots.dtype],
                                                            dims=selected_slots.shape,
                                                            vals=selected_slots.flatten()))
                self.__nodes.append(helper.make_node(op_type = 'Gather',
                                                    inputs=[layer_params.bottom_names[0], selected_slots_name],
                                                    outputs=[gather_name],
                                                    axis=1))
                self.__nodes.append(helper.make_node(op_type = 'Reshape',
                                                    inputs=[gather_name, shape_name],
                                                    outputs=layer_params.top_names))
            else:
                self.__nodes.append(helper.make_node(op_type = 'Reshape',
                                                    inputs=[layer_params.bottom_names[0], shape_name],
                                                    outputs=layer_params.top_names))
        elif layer_type == "Scale":
            concat_axis = 1
            concat_times = int(layer_params.scale_factor)
            if layer_params.scale_axis == 0:
                pre_shape_name = layer_params.top_names[0] + "_pre_shape"
                pre_shape = np.array([-1, 1], dtype=np.int64)
                reshape_name = layer_params.top_names[0] + "_reshape"
                self.__initializers.append(helper.make_tensor(name=pre_shape_name,
                                                            data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[pre_shape.dtype],
                                                            dims=pre_shape.shape,
                                                            vals=pre_shape.flatten()))
                self.__nodes.append(helper.make_node(op_type = 'Reshape',
                                                    inputs=[layer_params.bottom_names[0], pre_shape_name],
                                                    outputs=[reshape_name]))
            bottom_name = reshape_name if layer_params.scale_axis == 0 else layer_params.bottom_names[0]
            for i in range(concat_times):
                if i > 0:
                    prev_concat_name = bottom_name if i == 1 else layer_params.top_names[0] + "_concat" + str(i-1)
                    cur_concat_name = layer_params.top_names[0] + "_concat" + str(i)
                    self.__nodes.append(helper.make_node(op_type = 'Concat',
                                                        inputs=[prev_concat_name, bottom_name],
                                                        outputs=[cur_concat_name],
                                                        axis = concat_axis))
            last_concat_name = layer_params.top_names[0] + "_concat" + str(concat_times-1)
            shape_name = layer_params.top_names[0] + "_shape"
            shape = np.array([-1, dimensions[layer_params.top_names[0]]], dtype=np.int64)
            self.__initializers.append(helper.make_tensor(name=shape_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape.dtype],
                                                        dims=shape.shape,
                                                        vals=shape.flatten()))                                
            self.__nodes.append(helper.make_node(op_type = 'Reshape',
                                                inputs=[last_concat_name, shape_name],
                                                outputs=[layer_params.top_names[0]]))
        elif layer_type == "Sigmoid":
            if len(layer_params.top_names) == 0:
                layer_params.top_names.append("output")
            self.__nodes.append(helper.make_node(op_type = 'Sigmoid',
                                                inputs=layer_params.bottom_names,
                                                outputs=layer_params.top_names))                                        
        elif layer_type == "Slice":
            for tensor_name, rng in zip(layer_params.top_names, layer_params.ranges):
                starts_name = tensor_name + "_start" 
                ends_name = tensor_name + "_end"
                axes_name = tensor_name + "_axes"
                starts = np.array([rng[0]], dtype=np.int64)
                ends = np.array([rng[1]], dtype=np.int64)
                axes = np.array([1], dtype=np.int64)
                self.__initializers.append(helper.make_tensor(name=starts_name,
                                                            data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[starts.dtype],
                                                            dims=starts.shape,
                                                            vals=starts.flatten()))
                self.__initializers.append(helper.make_tensor(name=ends_name,
                                                            data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[ends.dtype],
                                                            dims=ends.shape,
                                                            vals=ends.flatten()))
                self.__initializers.append(helper.make_tensor(name=axes_name,
                                                            data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[axes.dtype],
                                                            dims=axes.shape,
                                                            vals=axes.flatten()))
                self.__nodes.append(helper.make_node(op_type = 'Slice',
                                                    inputs=[layer_params.bottom_names[0], starts_name, ends_name, axes_name],
                                                    outputs=[tensor_name]))
        elif layer_type == "Softmax":
            if len(layer_params.top_names) == 0:
                layer_params.top_names.append("output")
            self.__nodes.append(helper.make_node(op_type = 'Softmax',
                                                inputs=layer_params.bottom_names,
                                                outputs=layer_params.top_names))        
        elif layer_type == "Sub":
            x_name = layer_params.bottom_names[0]
            y_name = layer_params.bottom_names[1]
            z_name = layer_params.top_names[0]
            self.__nodes.append(helper.make_node(op_type = 'Sub',
                                                inputs=[x_name, y_name],
                                                outputs=[z_name]))
        elif layer_type == "WeightMultiply":
            weight_name = layer_params.top_names[0] + "_weight"
            weight = weights_dict[weight_name]
            flatten_name = layer_params.top_names[0] + "_flatten"
            transpose_name = layer_params.top_names[0] + "_transpose"
            expand_name = layer_params.top_names[0] + "_expand"
            expand_shape_name = layer_params.top_names[0] + "_expand_shape"
            expand_shape = np.array([1, layer_params.weight_dims[1]], dtype=np.int64)
            shape_name1 = layer_params.top_names[0] + "_shape1"
            shape1 = np.array([-1, layer_params.weight_dims[0], layer_params.weight_dims[1]], dtype=np.int64)
            reshape_name = layer_params.top_names[0] + "_reshape"
            multiply_name = layer_params.top_names[0] + "_multiply"
            shape_name2 = layer_params.top_names[0] + "_shape2"
            shape2 = np.array([-1, layer_params.weight_dims[0]*layer_params.weight_dims[1]], dtype=np.int64)
            self.__initializers.append(helper.make_tensor(name=weight_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[weight.dtype],
                                                        dims=weight.shape,
                                                        vals=weight.flatten()))
            self.__initializers.append(helper.make_tensor(name=expand_shape_name,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[expand_shape.dtype],
                                                        dims=expand_shape.shape,
                                                        vals=expand_shape.flatten()))
            self.__initializers.append(helper.make_tensor(name=shape_name1,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape1.dtype],
                                                        dims=shape1.shape,
                                                        vals=shape1.flatten()))
            self.__initializers.append(helper.make_tensor(name=shape_name2,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape2.dtype],
                                                        dims=shape2.shape,
                                                        vals=shape2.flatten()))
            self.__nodes.append(helper.make_node(op_type = 'Flatten',
                                                inputs=layer_params.bottom_names,
                                                outputs=[flatten_name],
                                                axis = 0))
            self.__nodes.append(helper.make_node(op_type = 'Transpose',
                                                inputs=[flatten_name],
                                                outputs=[transpose_name]))
            self.__nodes.append(helper.make_node(op_type = 'Expand',
                                                inputs=[transpose_name, expand_shape_name],
                                                outputs=[expand_name]))
            self.__nodes.append(helper.make_node(op_type = 'Reshape',
                                                inputs=[expand_name, shape_name1],
                                                outputs=[reshape_name]))
            self.__nodes.append(helper.make_node(op_type = 'Mul',
                                                inputs=[reshape_name, weight_name],
                                                outputs=[multiply_name]))
            self.__nodes.append(helper.make_node(op_type = 'Reshape',
                                                inputs=[multiply_name, shape_name2],
                                                outputs=layer_params.top_names))
        else:
            raise ValueError(layer_type + " is not supported in HugeCTR to ONNX converter")                               
        self.__counter += 1

    def create_graph(self, name = "hugectr_graph"):
        # Finalize key to indice hash
        key_to_indice_tensor = numpy_helper.from_array(self.__key_to_indice_hash_all_tables, 'key_to_indice_hash_all_tables')
        self.__initializers[0].CopyFrom(key_to_indice_tensor)
        # Create the graph (GraphProto)
        self.__graph_def = helper.make_graph(self.__nodes,
                                            name,
                                            self.__inputs,
                                            self.__outputs,
                                            self.__initializers)

    def save_model(self, model_path, op_version = 10, ir_version = 7):
        # Create the model (ModelProto)
        model_def = helper.make_model(self.__graph_def)
        model_def.opset_import[0].version = op_version
        model_def.ir_version = ir_version
        onnx.checker.check_model(model_def)
        print("[HUGECTR2ONNX][INFO]: The model is checked!")
        onnx.save(model_def, model_path)
        print("[HUGECTR2ONNX][INFO]: The model is saved at {}".format(model_path))
