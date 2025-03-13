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
            self.__inputs.append(
                helper.make_tensor_value_info(
                    layer_params.dense_name, TensorProto.FLOAT, [None, layer_params.dense_dim]
                )
            )
            if self.__convert_embeddding:
                for sparse_name, sparse_dim in zip(
                    layer_params.sparse_names, layer_params.sparse_dims
                ):
                    self.__inputs.append(
                        helper.make_tensor_value_info(
                            sparse_name, TensorProto.INT64, [None, sparse_dim[0], sparse_dim[1]]
                        )
                    )
            # Create output (ValueInfoProto)
            if np.ndim(layer_params.label_dim) == 0:
                self.__outputs.append(
                    helper.make_tensor_value_info(
                        layer_params.label_name, TensorProto.FLOAT, [None, layer_params.label_dim]
                    )
                )
            else:
                self.__label_names = layer_params.label_name
                for label_name, label_dim in zip(layer_params.label_name, layer_params.label_dim):
                    self.__outputs.append(
                        helper.make_tensor_value_info(
                            label_name, TensorProto.FLOAT, [None, label_dim]
                        )
                    )
        elif (
            layer_type == "DistributedSlotSparseEmbeddingHash"
            or layer_type == "LocalizedSlotSparseEmbeddingHash"
        ):
            if self.__convert_embeddding:
                embedding_table = weights_dict["embedding_table"]
                hash_table = weights_dict["hash_table"]
                embedding_table_name = layer_params.top_names[0] + "_embedding_table"
                hash_table_name = layer_params.top_names[0] + "_hash_table"
                indice_name = layer_params.top_names[0] + "_indice"
                embedding_feature_name = layer_params.top_names[0] + "_embedding_feature"
                self.__initializers.append(
                    helper.make_tensor(
                        name=embedding_table_name,
                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[embedding_table.dtype],
                        dims=embedding_table.shape,
                        vals=embedding_table.flatten(),
                    )
                )
                self.__initializers.append(
                    helper.make_tensor(
                        name=hash_table_name,
                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[hash_table.dtype],
                        dims=hash_table.shape,
                        vals=hash_table.flatten(),
                    )
                )
                self.__nodes.append(
                    helper.make_node(
                        op_type="Gather",
                        inputs=[hash_table_name, layer_params.bottom_names[0]],
                        outputs=[indice_name],
                        axis=0,
                    )
                )
                self.__nodes.append(
                    helper.make_node(
                        op_type="Gather",
                        inputs=[embedding_table_name, indice_name],
                        outputs=[embedding_feature_name],
                        axis=0,
                    )
                )
                reduce_type = "ReduceSum" if layer_params.combiner == 0 else "ReduceMean"
                self.__nodes.append(
                    helper.make_node(
                        op_type=reduce_type,
                        inputs=[embedding_feature_name],
                        outputs=layer_params.top_names,
                        keepdims=0,
                        axes=[-2],
                    )
                )
            else:
                self.__inputs.append(
                    helper.make_tensor_value_info(
                        layer_params.top_names[0],
                        TensorProto.FLOAT,
                        [
                            None,
                            dimensions[layer_params.top_names[0]][0],
                            dimensions[layer_params.top_names[0]][1],
                        ],
                    )
                )
        elif layer_type == "Add":
            for i in range(len(layer_params.bottom_names) - 1):
                x_name = (
                    layer_params.bottom_names[0] if i == 0 else layer_params.top_names[0] + str(i)
                )
                y_name = layer_params.bottom_names[i + 1]
                z_name = (
                    layer_params.top_names[0]
                    if i == len(layer_params.bottom_names) - 2
                    else layer_params.top_names[0] + str(i + 1)
                )
                self.__nodes.append(
                    helper.make_node(op_type="Add", inputs=[x_name, y_name], outputs=[z_name])
                )
        elif layer_type == "BatchNorm":
            gamma_name = layer_params.top_names[0] + "_gamma"
            beta_name = layer_params.top_names[0] + "_beta"
            running_mean_name = layer_params.top_names[0] + "_running_mean"
            running_variance_name = layer_params.top_names[0] + "_running_variance"
            gamma = weights_dict[gamma_name]
            beta = weights_dict[beta_name]
            running_mean = weights_dict[running_mean_name]
            running_variance = weights_dict[running_variance_name]
            self.__initializers.append(
                helper.make_tensor(
                    name=gamma_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[gamma.dtype],
                    dims=gamma.shape,
                    vals=gamma.flatten(),
                )
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=beta_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[beta.dtype],
                    dims=beta.shape,
                    vals=beta.flatten(),
                )
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=running_mean_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[running_mean.dtype],
                    dims=running_mean.shape,
                    vals=running_mean.flatten(),
                )
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=running_variance_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[running_variance.dtype],
                    dims=running_variance.shape,
                    vals=running_variance.flatten(),
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="BatchNormalization",
                    inputs=[
                        layer_params.bottom_names[0],
                        gamma_name,
                        beta_name,
                        running_mean_name,
                        running_variance_name,
                    ],
                    outputs=layer_params.top_names,
                    epsilon=layer_params.eps,
                    momentum=layer_params.factor,
                )
            )
        elif layer_type == "LayerNorm":
            input_name = layer_params.bottom_names[0]
            gamma_name = layer_params.top_names[0] + "_gamma"
            beta_name = layer_params.top_names[0] + "_beta"
            epsilon_name = layer_params.top_names[0] + "_epsilon"
            gamma = weights_dict[gamma_name]
            beta = weights_dict[beta_name]
            epsilon = np.array([layer_params.eps], dtype=np.float32)
            self.__initializers.append(
                helper.make_tensor(
                    name=gamma_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[gamma.dtype],
                    dims=gamma.shape,
                    vals=gamma.flatten(),
                )
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=beta_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[beta.dtype],
                    dims=beta.shape,
                    vals=beta.flatten(),
                )
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=epsilon_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[epsilon.dtype],
                    dims=epsilon.shape,
                    vals=epsilon.flatten(),
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="ReduceMean",
                    inputs=[input_name],
                    outputs=[input_name + "_mean"],
                    axes=[-1],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Sub",
                    inputs=[input_name, input_name + "_mean"],
                    outputs=[input_name + "_subtracted"],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Mul",
                    inputs=[input_name + "_subtracted", input_name + "_subtracted"],
                    outputs=[input_name + "_squared_diff"],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="ReduceMean",
                    inputs=[input_name + "_squared_diff"],
                    outputs=[input_name + "_variance"],
                    axes=[-1],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Add",
                    inputs=[input_name + "_variance", epsilon_name],
                    outputs=[input_name + "_var_epsilon"],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Sqrt",
                    inputs=[input_name + "_var_epsilon"],
                    outputs=[input_name + "_std_dev"],
                )
            )

            self.__nodes.append(
                helper.make_node(
                    op_type="Reciprocal",
                    inputs=[input_name + "_std_dev"],
                    outputs=[input_name + "_inv_std_dev"],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Sub",
                    inputs=[input_name, input_name + "_mean"],
                    outputs=[input_name + "_normalized"],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Mul",
                    inputs=[input_name + "_normalized", input_name + "_inv_std_dev"],
                    outputs=[input_name + "_scaled"],
                )
            )

            self.__nodes.append(
                helper.make_node(
                    op_type="Mul",
                    inputs=[input_name + "_scaled", gamma_name],
                    outputs=[input_name + "_scaled_output"],
                )
            )

            self.__nodes.append(
                helper.make_node(
                    op_type="Add",
                    inputs=[input_name + "_scaled_output", beta_name],
                    outputs=layer_params.top_names,
                )
            )

        elif layer_type == "Concat":
            self.__nodes.append(
                helper.make_node(
                    op_type="Concat",
                    inputs=layer_params.bottom_names,
                    outputs=layer_params.top_names,
                    axis=layer_params.axis,
                )
            )
        elif layer_type == "ElementwiseMultiply":
            for i in range(len(layer_params.bottom_names) - 1):
                x_name = (
                    layer_params.bottom_names[0] if i == 0 else layer_params.top_names[0] + str(i)
                )
                y_name = layer_params.bottom_names[i + 1]
                z_name = (
                    layer_params.top_names[0]
                    if i == len(layer_params.bottom_names) - 2
                    else layer_params.top_names[0] + str(i + 1)
                )
                self.__nodes.append(
                    helper.make_node(op_type="Mul", inputs=[x_name, y_name], outputs=[z_name])
                )
        elif layer_type == "Dropout":
            self.__nodes.append(
                helper.make_node(
                    op_type="Dropout",
                    inputs=layer_params.bottom_names,
                    outputs=layer_params.top_names,
                )
            )
        elif layer_type == "ELU":
            self.__nodes.append(
                helper.make_node(
                    op_type="Elu",
                    inputs=layer_params.bottom_names,
                    outputs=layer_params.top_names,
                    alpha=layer_params.elu_alpha,
                )
            )
        elif layer_type == "SequenceMask":
            sequence_lens_from_name = layer_params.bottom_names[0]
            sequence_lens_to_name = layer_params.bottom_names[1]
            range_from_values = np.arange(layer_params.max_sequence_len_from).astype(np.float32)
            range_to_values = np.arange(layer_params.max_sequence_len_to).astype(np.float32)
            range_from_name = layer_params.top_names[0] + "_range_from"
            range_to_name = layer_params.top_names[0] + "_range_to"
            mask_from_name = layer_params.top_names[0] + "_mask_from"
            mask_to_name = layer_params.top_names[0] + "_mask_to"
            mask_from_unsqueezed_name = layer_params.top_names[0] + "_mask_from_unsqueezed"
            mask_to_unsqueezed_name = layer_params.top_names[0] + "_mask_to_unsqueezed"
            mask_from_unsqueezed_int_name = layer_params.top_names[0] + "_mask_from_unsqueezed_int"
            mask_to_unsqueezed_int_name = layer_params.top_names[0] + "_mask_to_unsqueezed_int"
            mask_int_name = layer_params.top_names[0] + "_mask_int"
            mask_int_unsqueezed_name = layer_params.top_names[0] + "_mask_int_unsqueezed"

            self.__nodes.append(
                helper.make_node(
                    op_type="Constant",
                    inputs=[],
                    outputs=[range_from_name],
                    value=numpy_helper.from_array(range_from_values),
                )
            )

            self.__nodes.append(
                helper.make_node(
                    op_type="Constant",
                    inputs=[],
                    outputs=[range_to_name],
                    value=numpy_helper.from_array(range_to_values),
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Less",
                    inputs=[range_from_name, sequence_lens_from_name],
                    outputs=[mask_from_name],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Less",
                    inputs=[range_to_name, sequence_lens_to_name],
                    outputs=[mask_to_name],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Unsqueeze",
                    inputs=[mask_from_name],
                    outputs=[mask_from_unsqueezed_name],
                    axes=[2],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Unsqueeze",
                    inputs=[mask_to_name],
                    outputs=[mask_to_unsqueezed_name],
                    axes=[1],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Cast",
                    inputs=[mask_from_unsqueezed_name],
                    outputs=[mask_from_unsqueezed_int_name],
                    to=TensorProto.INT32,
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Cast",
                    inputs=[mask_to_unsqueezed_name],
                    outputs=[mask_to_unsqueezed_int_name],
                    to=TensorProto.INT32,
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Mul",
                    inputs=[mask_from_unsqueezed_int_name, mask_to_unsqueezed_int_name],
                    outputs=[mask_int_name],
                )
            )

            self.__nodes.append(
                helper.make_node(
                    op_type="Unsqueeze",
                    inputs=[mask_int_name],
                    outputs=[mask_int_unsqueezed_name],
                    axes=[1],
                )
            )

            self.__nodes.append(
                helper.make_node(
                    op_type="Cast",
                    inputs=[mask_int_unsqueezed_name],
                    outputs=layer_params.top_names,
                    to=TensorProto.BOOL,
                )
            )
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
            self.__initializers.append(
                helper.make_tensor(
                    name=shape_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape.dtype],
                    dims=shape.shape,
                    vals=shape.flatten(),
                )
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=scalar_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[scalar.dtype],
                    dims=scalar.shape,
                    vals=scalar.flatten(),
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Reshape",
                    inputs=[layer_params.bottom_names[0], shape_name],
                    outputs=[reshape_name],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="ReduceSum",
                    inputs=[reshape_name],
                    outputs=[reduce_sum_name],
                    keepdims=0,
                    axes=[-2],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Mul",
                    inputs=[reduce_sum_name, reduce_sum_name],
                    outputs=[sum_square_name],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Mul", inputs=[reshape_name, reshape_name], outputs=[square_name]
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="ReduceSum",
                    inputs=[square_name],
                    outputs=[square_sum_name],
                    keepdims=0,
                    axes=[-2],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Sub", inputs=[sum_square_name, square_sum_name], outputs=[sub_name]
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Mul", inputs=[sub_name, scalar_name], outputs=layer_params.top_names
                )
            )
        elif layer_type == "InnerProduct":
            weight_name = layer_params.top_names[0] + "_weight"
            bias_name = layer_params.top_names[0] + "_bias"
            weight = weights_dict[weight_name]
            bias = weights_dict[bias_name]
            self.__initializers.append(
                helper.make_tensor(
                    name=weight_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[weight.dtype],
                    dims=weight.shape,
                    vals=weight.flatten(),
                )
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=bias_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[bias.dtype],
                    dims=bias.shape,
                    vals=bias.flatten(),
                )
            )
            if isinstance(dimensions[layer_params.bottom_names[0]], tuple):
                dim = dimensions[layer_params.bottom_names[0]]
                shape_name1 = layer_params.top_names[0] + "_shape1"
                shape1 = np.array([-1, dim[1]], dtype=np.int64)

                shape_name2 = layer_params.top_names[0] + "_shape2"
                shape2 = np.array([-1, dim[0], layer_params.num_output], dtype=np.int64)
                reshape_name1 = layer_params.bottom_names[0] + "_reshape_fc1"
                reshape_name2 = layer_params.top_names[0] + "_reshape_fc2"

                self.__initializers.append(
                    helper.make_tensor(
                        name=shape_name1,
                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape1.dtype],
                        dims=shape1.shape,
                        vals=shape1.flatten(),
                    )
                )
                self.__initializers.append(
                    helper.make_tensor(
                        name=shape_name2,
                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape2.dtype],
                        dims=shape2.shape,
                        vals=shape2.flatten(),
                    )
                )
                self.__nodes.append(
                    helper.make_node(
                        op_type="Reshape",
                        inputs=[layer_params.bottom_names[0], shape_name1],
                        outputs=[reshape_name1],
                    )
                )
                self.__nodes.append(
                    helper.make_node(
                        op_type="Gemm",
                        inputs=[reshape_name1, weight_name, bias_name],
                        outputs=[reshape_name2],
                    )
                )
                self.__nodes.append(
                    helper.make_node(
                        op_type="Reshape",
                        inputs=[reshape_name2, shape_name2],
                        outputs=layer_params.top_names,
                    )
                )
            else:
                self.__nodes.append(
                    helper.make_node(
                        op_type="Gemm",
                        inputs=[layer_params.bottom_names[0], weight_name, bias_name],
                        outputs=layer_params.top_names,
                    )
                )

        elif layer_type == "FusedInnerProduct":
            weight_name = layer_params.top_names[0] + "_weight"
            bias_name = layer_params.top_names[0] + "_bias"
            weight = weights_dict[weight_name]
            bias = weights_dict[bias_name]
            gemm_name = layer_params.top_names[0] + "_gemm"
            self.__initializers.append(
                helper.make_tensor(
                    name=weight_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[weight.dtype],
                    dims=weight.shape,
                    vals=weight.flatten(),
                )
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=bias_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[bias.dtype],
                    dims=bias.shape,
                    vals=bias.flatten(),
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Gemm",
                    inputs=[layer_params.bottom_names[0], weight_name, bias_name],
                    outputs=[gemm_name],
                )
            )
            self.__nodes.append(
                helper.make_node(op_type="Relu", inputs=[gemm_name], outputs=layer_params.top_names)
            )
        elif layer_type == "MLP":
            num_layers = len(layer_params.num_outputs)
            acts = [layer_params.activation] * num_layers
            if layer_params.activations:
                acts = layer_params.activations

            biases = [layer_params.use_bias] * num_layers
            if layer_params.biases:
                biases = layer_params.biases
            for i in range(num_layers):
                weight_name = layer_params.top_names[0] + str(i) + "_weight"
                bias_name = layer_params.top_names[0] + str(i) + "_bias"
                weight = weights_dict[weight_name]
                bias = weights_dict[bias_name]

                bottom_name = (
                    layer_params.top_names[0] + str(i - 1) + "_out"
                    if i != 0
                    else layer_params.bottom_names[0]
                )
                gemm_name = layer_params.top_names[0] + str(i) + "_gemm"
                top_name = (
                    layer_params.top_names[0] + str(i) + "_out"
                    if i != num_layers - 1
                    else layer_params.top_names[0]
                )

                self.__initializers.append(
                    helper.make_tensor(
                        name=weight_name,
                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[weight.dtype],
                        dims=weight.shape,
                        vals=weight.flatten(),
                    )
                )
                output_name = gemm_name if acts[i] == "Relu" else top_name
                if biases[i]:
                    self.__initializers.append(
                        helper.make_tensor(
                            name=bias_name,
                            data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[bias.dtype],
                            dims=bias.shape,
                            vals=bias.flatten(),
                        )
                    )
                    self.__nodes.append(
                        helper.make_node(
                            op_type="Gemm",
                            inputs=[bottom_name, weight_name, bias_name],
                            outputs=[output_name],
                        )
                    )
                else:
                    self.__nodes.append(
                        helper.make_node(
                            op_type="Gemm", inputs=[bottom_name, weight_name], outputs=[output_name]
                        )
                    )

                if acts[i] == "Relu":
                    self.__nodes.append(
                        helper.make_node(op_type="Relu", inputs=[gemm_name], outputs=[top_name])
                    )
        elif layer_type == "FusedReshapeConcat":
            slot_num = dimensions[layer_params.bottom_names[0]][0]
            output_fea_num = 0
            item_starts_name = layer_params.top_names[0] + "_item_start"
            item_ends_name = layer_params.top_names[0] + "_item_end"
            ad_starts_name = layer_params.top_names[0] + "_ad_start"
            ad_ends_name = layer_params.top_names[0] + "_ad_end"
            axes_name = layer_params.top_names[0] + "_axes"
            item_starts = np.array([0], dtype=np.int64)
            item_ends = np.array([slot_num - 1], dtype=np.int64)
            ad_starts = np.array([slot_num - 1], dtype=np.int64)
            ad_ends = np.array([slot_num], dtype=np.int64)
            axes = np.array([1], dtype=np.int64)
            self.__initializers.append(
                helper.make_tensor(
                    name=item_starts_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[item_starts.dtype],
                    dims=item_starts.shape,
                    vals=item_starts.flatten(),
                )
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=item_ends_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[item_ends.dtype],
                    dims=item_ends.shape,
                    vals=item_ends.flatten(),
                )
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=ad_starts_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[ad_starts.dtype],
                    dims=ad_starts.shape,
                    vals=ad_starts.flatten(),
                )
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=ad_ends_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[ad_ends.dtype],
                    dims=ad_ends.shape,
                    vals=ad_ends.flatten(),
                )
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=axes_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[axes.dtype],
                    dims=axes.shape,
                    vals=axes.flatten(),
                )
            )
            for i in range(len(layer_params.bottom_names)):
                input_tensor_name = layer_params.bottom_names[i]
                output_fea_num += dimensions[input_tensor_name][1]
                item_slice_name = input_tensor_name + "_slice_item" + str(i)
                ad_slice_name = input_tensor_name + "_slice_ad" + str(i)
                item_concat_name = input_tensor_name + "_concat_item" + str(i)
                ad_concat_name = input_tensor_name + "_concat_ad" + str(i)
                self.__nodes.append(
                    helper.make_node(
                        op_type="Slice",
                        inputs=[input_tensor_name, item_starts_name, item_ends_name, axes_name],
                        outputs=[item_slice_name],
                    )
                )
                self.__nodes.append(
                    helper.make_node(
                        op_type="Slice",
                        inputs=[input_tensor_name, ad_starts_name, ad_ends_name, axes_name],
                        outputs=[ad_slice_name],
                    )
                )
                if i > 0:
                    prev_item_concat_name = (
                        layer_params.bottom_names[i - 1] + "_slice_item" + str(i - 1)
                        if i == 1
                        else layer_params.bottom_names[i - 1] + "_concat_item" + str(i - 1)
                    )
                    prev_ad_concat_name = (
                        layer_params.bottom_names[i - 1] + "_slice_ad" + str(i - 1)
                        if i == 1
                        else layer_params.bottom_names[i - 1] + "_concat_ad" + str(i - 1)
                    )
                    self.__nodes.append(
                        helper.make_node(
                            op_type="Concat",
                            inputs=[prev_item_concat_name, item_slice_name],
                            outputs=[item_concat_name],
                            axis=2,
                        )
                    )
                    self.__nodes.append(
                        helper.make_node(
                            op_type="Concat",
                            inputs=[prev_ad_concat_name, ad_slice_name],
                            outputs=[ad_concat_name],
                            axis=2,
                        )
                    )
            shape_name = layer_params.top_names[0] + "_item_shape"
            shape = np.array([-1, output_fea_num], dtype=np.int64)
            num_input_tensors = len(layer_params.bottom_names)
            last_bottom_tensor_name = layer_params.bottom_names[num_input_tensors - 1]
            last_item_tensor_name = (
                last_bottom_tensor_name + "_slice_item" + str(num_input_tensors - 1)
                if num_input_tensors == 1
                else last_bottom_tensor_name + "_concat_item" + str(num_input_tensors - 1)
            )
            last_ad_tensor_name = (
                last_bottom_tensor_name + "_slice_ad" + str(num_input_tensors - 1)
                if num_input_tensors == 1
                else last_bottom_tensor_name + "_concat_ad" + str(num_input_tensors - 1)
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=shape_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape.dtype],
                    dims=shape.shape,
                    vals=shape.flatten(),
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Reshape",
                    inputs=[last_item_tensor_name, shape_name],
                    outputs=[layer_params.top_names[0]],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Reshape",
                    inputs=[last_ad_tensor_name, shape_name],
                    outputs=[layer_params.top_names[1]],
                )
            )
        elif layer_type == "MultiHeadAttention":
            query_name = layer_params.bottom_names[0]
            key_name = layer_params.bottom_names[1]
            value_name = layer_params.bottom_names[2]
            sequence_mask_name = layer_params.bottom_names[3]
            output_name = layer_params.top_names[0]

            dims = dimensions[query_name]
            head_num = layer_params.num_attention_heads
            seq_len = dims[0]
            key_transpose_name = key_name + "_transpose"
            query_transpose_name = query_name + "_transpose"
            value_transpose_name = value_name + "_transpose"
            raw_score_prediv_name = "raw_score_prediv"
            raw_score_name = output_name + "_raw_score"
            shape_name = output_name + "_qkv_4d_shape"
            shape = np.array([-1, dims[0], head_num, dims[1] / head_num], dtype=np.int64)
            sqrtdk_name = output_name + "_sqrtdk"
            sqrtdk = np.array([np.sqrt(dims[1] / head_num)], dtype=np.float32)
            query_reshape = layer_params.bottom_names[0] + "_4d"
            key_reshape = layer_params.bottom_names[1] + "_4d"
            value_reshape = layer_params.bottom_names[2] + "_4d"
            self.__initializers.append(
                helper.make_tensor(
                    name=shape_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape.dtype],
                    dims=shape.shape,
                    vals=shape.flatten(),
                )
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=sqrtdk_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[sqrtdk.dtype],
                    dims=sqrtdk.shape,
                    vals=sqrtdk.flatten(),
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Reshape", inputs=[query_name, shape_name], outputs=[query_reshape]
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Reshape", inputs=[key_name, shape_name], outputs=[key_reshape]
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Reshape",
                    inputs=[value_name, shape_name],
                    outputs=[value_reshape],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Transpose",
                    inputs=[query_reshape],
                    outputs=[query_transpose_name],
                    perm=[0, 2, 1, 3],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Transpose",
                    inputs=[key_reshape],
                    outputs=[key_transpose_name],
                    perm=[0, 2, 3, 1],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Transpose",
                    inputs=[value_reshape],
                    outputs=[value_transpose_name],
                    perm=[0, 2, 1, 3],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="MatMul",
                    inputs=[query_transpose_name, key_transpose_name],
                    outputs=[raw_score_prediv_name],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Div",
                    inputs=[raw_score_prediv_name, sqrtdk_name],
                    outputs=[raw_score_name],
                )
            )

            masked_not_name = sequence_mask_name + "_not"
            masked_not_float_name = sequence_mask_name + "_not_float"
            neg_infinity = np.array([-10000.0], dtype=np.float32)
            neg_infinity_name = sequence_mask_name + "_neg_infinity"
            masked_not_float_mul_neg_infinity_name = (
                sequence_mask_name + "_not_float_mul_neg_infinity"
            )
            modified_raw_score_name = raw_score_name + "_modified"
            modified_raw_score_row_max_name = raw_score_name + "_modified_rowmax"
            modified_raw_score_stable_name = raw_score_name + "_modified_stable"
            softmax_score_name = raw_score_name + "_softmax"

            self.__initializers.append(
                helper.make_tensor(
                    name=neg_infinity_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[neg_infinity.dtype],
                    dims=neg_infinity.shape,
                    vals=neg_infinity.flatten(),
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Not",
                    inputs=[sequence_mask_name],
                    outputs=[masked_not_name],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Cast",
                    inputs=[masked_not_name],
                    outputs=[masked_not_float_name],
                    to=onnx.TensorProto.FLOAT,
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Mul",
                    inputs=[masked_not_float_name, neg_infinity_name],
                    outputs=[masked_not_float_mul_neg_infinity_name],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Add",
                    inputs=[raw_score_name, masked_not_float_mul_neg_infinity_name],
                    outputs=[modified_raw_score_name],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="ReduceMax",
                    inputs=[modified_raw_score_name],
                    outputs=[modified_raw_score_row_max_name],
                    axes=[-1],
                    keepdims=1,
                )
            )

            self.__nodes.append(
                helper.make_node(
                    op_type="Sub",
                    inputs=[modified_raw_score_name, modified_raw_score_row_max_name],
                    outputs=[modified_raw_score_stable_name],
                )
            )

            self.__nodes.append(
                helper.make_node(
                    op_type="Softmax",
                    inputs=[modified_raw_score_stable_name],
                    outputs=[softmax_score_name],
                    axis=-1,
                )
            )

            atten_output_init_name = output_name + "_init"
            atten_output_transpose_name = output_name + "_transpose"
            atten_output_shape_name = output_name + "_3d_shape"
            atten_output_shape = np.array([-1, dims[0], dims[1]], dtype=np.int64)
            self.__initializers.append(
                helper.make_tensor(
                    name=atten_output_shape_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[atten_output_shape.dtype],
                    dims=atten_output_shape.shape,
                    vals=atten_output_shape.flatten(),
                )
            )

            self.__nodes.append(
                helper.make_node(
                    op_type="MatMul",
                    inputs=[softmax_score_name, value_transpose_name],
                    outputs=[atten_output_init_name],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Transpose",
                    inputs=[atten_output_init_name],
                    outputs=[atten_output_transpose_name],
                    perm=[0, 2, 1, 3],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Reshape",
                    inputs=[atten_output_transpose_name, atten_output_shape_name],
                    outputs=layer_params.top_names,
                )
            )
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
            shape2 = np.array([-1, (slot_num + 1) * (slot_num + 1)], dtype=np.int64)
            reshape_name2 = layer_params.top_names[0] + "_reshape2"
            indices_name = layer_params.top_names[0] + "_indices"
            indices = [i * (slot_num + 1) + j for j in range(1, 1 + slot_num) for i in range(j)]
            indices = np.array(indices, dtype=np.int64)
            gather_name = layer_params.top_names[0] + "_gather"
            interaction_name = layer_params.top_names[0] + "_interaction"
            self.__initializers.append(
                helper.make_tensor(
                    name=shape_name1,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape1.dtype],
                    dims=shape1.shape,
                    vals=shape1.flatten(),
                )
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=shape_name2,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape2.dtype],
                    dims=shape2.shape,
                    vals=shape2.flatten(),
                )
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=indices_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[indices.dtype],
                    dims=indices.shape,
                    vals=indices.flatten(),
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Reshape", inputs=[mlp_name, shape_name1], outputs=[reshape_name1]
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Concat",
                    inputs=[reshape_name1, emb_name],
                    outputs=[concat_name],
                    axis=-2,
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Transpose",
                    inputs=[concat_name],
                    outputs=[transpose_name],
                    perm=[0, 2, 1],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="MatMul", inputs=[concat_name, transpose_name], outputs=[matmul_name]
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Reshape", inputs=[matmul_name, shape_name2], outputs=[reshape_name2]
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Gather",
                    inputs=[reshape_name2, indices_name],
                    outputs=[gather_name],
                    axis=1,
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Concat",
                    inputs=[mlp_name, gather_name],
                    outputs=[interaction_name],
                    axis=-1,
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Pad",
                    inputs=[interaction_name],
                    outputs=layer_params.top_names,
                    pads=[0, 0, 0, 1],
                )
            )
        elif layer_type == "MatrixMultiply":
            self.__nodes.append(
                helper.make_node(
                    op_type="MatMul",
                    inputs=layer_params.bottom_names,
                    outputs=layer_params.top_names,
                )
            )
        elif layer_type == "MultiCross":
            weights_name = layer_params.top_names[0] + "_weights"
            biases_name = layer_params.top_names[0] + "_biases"
            weights = weights_dict[weights_name]
            biases = weights_dict[biases_name]
            for i in range(layer_params.num_layers):
                weight_name = weights_name + str(i)
                bias_name = biases_name + str(i)
                weight = weights[i]
                bias = biases[i]
                feed_name = (
                    layer_params.bottom_names[0]
                    if i == 0
                    else layer_params.top_names[0] + "_multicross" + str(i)
                )
                matmul_name = layer_params.top_names[0] + "_matmul" + str(i + 1)
                multiply_name = layer_params.top_names[0] + "_multiply" + str(i + 1)
                add_name = layer_params.top_names[0] + "_add" + str(i + 1)
                multicross_name = (
                    layer_params.top_names[0]
                    if i == layer_params.num_layers - 1
                    else layer_params.top_names[0] + "_multicross" + str(i + 1)
                )
                self.__initializers.append(
                    helper.make_tensor(
                        name=weight_name,
                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[weight.dtype],
                        dims=weight.shape,
                        vals=weight.flatten(),
                    )
                )
                self.__initializers.append(
                    helper.make_tensor(
                        name=bias_name,
                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[bias.dtype],
                        dims=bias.shape,
                        vals=bias.flatten(),
                    )
                )
                self.__nodes.append(
                    helper.make_node(
                        op_type="MatMul", inputs=[feed_name, weight_name], outputs=[matmul_name]
                    )
                )
                self.__nodes.append(
                    helper.make_node(
                        op_type="Mul",
                        inputs=[layer_params.bottom_names[0], matmul_name],
                        outputs=[multiply_name],
                    )
                )
                self.__nodes.append(
                    helper.make_node(
                        op_type="Add", inputs=[feed_name, bias_name], outputs=[add_name]
                    )
                )
                self.__nodes.append(
                    helper.make_node(
                        op_type="Add", inputs=[multiply_name, add_name], outputs=[multicross_name]
                    )
                )
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
            self.__initializers.append(
                helper.make_tensor(
                    name=epsilon_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[epsilon.dtype],
                    dims=epsilon.shape,
                    vals=epsilon.flatten(),
                )
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=alpha_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[alpha.dtype],
                    dims=alpha.shape,
                    vals=alpha.flatten(),
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="ReduceMean",
                    inputs=layer_params.bottom_names,
                    outputs=[mean_name],
                    axes=[0],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Mul", inputs=[mean_name, mean_name], outputs=[mean_square_name]
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Mul",
                    inputs=[layer_params.bottom_names[0], layer_params.bottom_names[0]],
                    outputs=[square_name],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="ReduceMean", inputs=[square_name], outputs=[square_mean_name], axes=[0]
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Sub", inputs=[square_mean_name, mean_square_name], outputs=[var_name]
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Add", inputs=[var_name, epsilon_name], outputs=[var_epsilon_name]
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Sqrt", inputs=[var_epsilon_name], outputs=[var_epsilon_sqrt_name]
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Sub",
                    inputs=[layer_params.bottom_names[0], mean_name],
                    outputs=[diff_name],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Div", inputs=[diff_name, var_epsilon_sqrt_name], outputs=[value_name]
                )
            )
            self.__nodes.append(
                helper.make_node(op_type="Sigmoid", inputs=[value_name], outputs=[ps_name])
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Mul",
                    inputs=[layer_params.bottom_names[0], ps_name],
                    outputs=[first_item_name],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Mul",
                    inputs=[layer_params.bottom_names[0], ps_name],
                    outputs=[ps_mul_name],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Sub",
                    inputs=[ps_mul_name, layer_params.bottom_names[0]],
                    outputs=[second_item_tmp_name],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Mul",
                    inputs=[second_item_tmp_name, alpha_name],
                    outputs=[second_item_name],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Sub",
                    inputs=[first_item_name, second_item_name],
                    outputs=layer_params.top_names,
                )
            )
        elif layer_type == "ReduceMean":
            self.__nodes.append(
                helper.make_node(
                    op_type="ReduceMean",
                    inputs=layer_params.bottom_names,
                    outputs=layer_params.top_names,
                    keepdims=1,
                    axes=[layer_params.axis],
                )
            )
        elif layer_type == "ReduceSum":
            self.__nodes.append(
                helper.make_node(
                    op_type="ReduceSum",
                    inputs=layer_params.bottom_names,
                    outputs=layer_params.top_names,
                    keepdims=1,
                    axes=[layer_params.axis],
                )
            )
        elif layer_type == "ReLU":
            self.__nodes.append(
                helper.make_node(
                    op_type="Relu", inputs=layer_params.bottom_names, outputs=layer_params.top_names
                )
            )
        elif layer_type == "Reshape":
            shape_name = layer_params.top_names[0] + "_shape"
            if layer_params.reshape_time_step == 0:
                shape = np.array([-1, layer_params.leading_dim], dtype=np.int64)
            else:
                shape = np.array(
                    [-1, layer_params.reshape_time_step, layer_params.leading_dim], dtype=np.int64
                )
            self.__initializers.append(
                helper.make_tensor(
                    name=shape_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape.dtype],
                    dims=shape.shape,
                    vals=shape.flatten(),
                )
            )
            if layer_params.selected:
                gather_name = layer_params.top_names[0] + "_gather"
                selected_slots_name = layer_params.top_names[0] + "_selected_slots"
                selected_slots = np.array(layer_params.selected_slots, dtype=np.int64)
                self.__initializers.append(
                    helper.make_tensor(
                        name=selected_slots_name,
                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[selected_slots.dtype],
                        dims=selected_slots.shape,
                        vals=selected_slots.flatten(),
                    )
                )
                self.__nodes.append(
                    helper.make_node(
                        op_type="Gather",
                        inputs=[layer_params.bottom_names[0], selected_slots_name],
                        outputs=[gather_name],
                        axis=1,
                    )
                )
                self.__nodes.append(
                    helper.make_node(
                        op_type="Reshape",
                        inputs=[gather_name, shape_name],
                        outputs=layer_params.top_names,
                    )
                )
            else:
                self.__nodes.append(
                    helper.make_node(
                        op_type="Reshape",
                        inputs=[layer_params.bottom_names[0], shape_name],
                        outputs=layer_params.top_names,
                    )
                )
        elif layer_type == "Scale":
            concat_axis = 1
            concat_times = int(layer_params.scale_factor)
            if layer_params.scale_axis == 0:
                pre_shape_name = layer_params.top_names[0] + "_pre_shape"
                pre_shape = np.array([-1, 1], dtype=np.int64)
                reshape_name = layer_params.top_names[0] + "_reshape"
                self.__initializers.append(
                    helper.make_tensor(
                        name=pre_shape_name,
                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[pre_shape.dtype],
                        dims=pre_shape.shape,
                        vals=pre_shape.flatten(),
                    )
                )
                self.__nodes.append(
                    helper.make_node(
                        op_type="Reshape",
                        inputs=[layer_params.bottom_names[0], pre_shape_name],
                        outputs=[reshape_name],
                    )
                )
            bottom_name = (
                reshape_name if layer_params.scale_axis == 0 else layer_params.bottom_names[0]
            )
            for i in range(concat_times):
                if i > 0:
                    prev_concat_name = (
                        bottom_name
                        if i == 1
                        else layer_params.top_names[0] + "_concat" + str(i - 1)
                    )
                    cur_concat_name = layer_params.top_names[0] + "_concat" + str(i)
                    self.__nodes.append(
                        helper.make_node(
                            op_type="Concat",
                            inputs=[prev_concat_name, bottom_name],
                            outputs=[cur_concat_name],
                            axis=concat_axis,
                        )
                    )
            last_concat_name = layer_params.top_names[0] + "_concat" + str(concat_times - 1)
            shape_name = layer_params.top_names[0] + "_shape"
            shape = np.array([-1, dimensions[layer_params.top_names[0]]], dtype=np.int64)
            self.__initializers.append(
                helper.make_tensor(
                    name=shape_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape.dtype],
                    dims=shape.shape,
                    vals=shape.flatten(),
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Reshape",
                    inputs=[last_concat_name, shape_name],
                    outputs=[layer_params.top_names[0]],
                )
            )
        elif layer_type == "Sigmoid":
            if len(layer_params.top_names) == 0:
                layer_params.top_names.append("output")
            self.__nodes.append(
                helper.make_node(
                    op_type="Sigmoid",
                    inputs=layer_params.bottom_names,
                    outputs=layer_params.top_names,
                )
            )
        elif layer_type == "Slice":
            for tensor_name, rng in zip(layer_params.top_names, layer_params.ranges):
                starts_name = tensor_name + "_start"
                ends_name = tensor_name + "_end"
                axes_name = tensor_name + "_axes"
                starts = np.array([rng[0]], dtype=np.int64)
                ends = np.array([rng[1]], dtype=np.int64)
                axes = np.array([-1], dtype=np.int64)
                self.__initializers.append(
                    helper.make_tensor(
                        name=starts_name,
                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[starts.dtype],
                        dims=starts.shape,
                        vals=starts.flatten(),
                    )
                )
                self.__initializers.append(
                    helper.make_tensor(
                        name=ends_name,
                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[ends.dtype],
                        dims=ends.shape,
                        vals=ends.flatten(),
                    )
                )
                self.__initializers.append(
                    helper.make_tensor(
                        name=axes_name,
                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[axes.dtype],
                        dims=axes.shape,
                        vals=axes.flatten(),
                    )
                )
                self.__nodes.append(
                    helper.make_node(
                        op_type="Slice",
                        inputs=[layer_params.bottom_names[0], starts_name, ends_name, axes_name],
                        outputs=[tensor_name],
                    )
                )
        elif layer_type == "Softmax":
            if len(layer_params.top_names) == 0:
                layer_params.top_names.append("output")
            if len(layer_params.bottom_names) == 1:
                self.__nodes.append(
                    helper.make_node(
                        op_type="Softmax",
                        inputs=layer_params.bottom_names,
                        outputs=layer_params.top_names,
                    )
                )
            else:
                input_name = layer_params.bottom_names[0]
                mask_name = layer_params.bottom_names[1]
                masked_not_name = layer_params.bottom_names[1] + "_not"
                float_masked_not_name = layer_params.bottom_names[1] + "_float_not"
                masked_offset_name = layer_params.bottom_names[1] + "_masked_offset"
                head_num = dimensions[layer_params.bottom_names[0]][0]
                seq_len = dimensions[layer_params.bottom_names[0]][1]
                padding = np.array([-10000.0], dtype=np.float32)
                padding_name = layer_params.bottom_names[0] + "_padding_val"
                masked_input_name = layer_params.bottom_names[0] + "_masked_val"
                self.__initializers.append(
                    helper.make_tensor(
                        name=padding_name,
                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[padding.dtype],
                        dims=padding.shape,
                        vals=padding.flatten(),
                    )
                )
                self.__nodes.append(
                    helper.make_node(
                        op_type="Not",
                        inputs=[mask_name],
                        outputs=[masked_not_name],
                    )
                )
                self.__nodes.append(
                    helper.make_node(
                        op_type="Cast",
                        inputs=[masked_not_name],
                        outputs=[float_masked_not_name],
                        to=onnx.TensorProto.FLOAT,
                    )
                )
                self.__nodes.append(
                    helper.make_node(
                        op_type="Mul",
                        inputs=[float_masked_not_name, padding_name],
                        outputs=[masked_offset_name],
                    )
                )
                self.__nodes.append(
                    helper.make_node(
                        op_type="Add",
                        inputs=[input_name, masked_offset_name],
                        outputs=[masked_input_name],
                    )
                )
                self.__nodes.append(
                    helper.make_node(
                        op_type="Softmax",
                        inputs=[masked_input_name],
                        outputs=layer_params.top_names,
                    )
                )

        elif layer_type == "Sub":
            x_name = layer_params.bottom_names[0]
            y_name = layer_params.bottom_names[1]
            z_name = layer_params.top_names[0]
            self.__nodes.append(
                helper.make_node(op_type="Sub", inputs=[x_name, y_name], outputs=[z_name])
            )
        elif layer_type == "WeightMultiply":
            weight_name = layer_params.top_names[0] + "_weight"
            weight = weights_dict[weight_name]
            flatten_name = layer_params.top_names[0] + "_flatten"
            transpose_name = layer_params.top_names[0] + "_transpose"
            expand_name = layer_params.top_names[0] + "_expand"
            expand_shape_name = layer_params.top_names[0] + "_expand_shape"
            expand_shape = np.array([1, layer_params.weight_dims[1]], dtype=np.int64)
            shape_name1 = layer_params.top_names[0] + "_shape1"
            shape1 = np.array(
                [-1, layer_params.weight_dims[0], layer_params.weight_dims[1]], dtype=np.int64
            )
            reshape_name = layer_params.top_names[0] + "_reshape"
            multiply_name = layer_params.top_names[0] + "_multiply"
            shape_name2 = layer_params.top_names[0] + "_shape2"
            shape2 = np.array(
                [-1, layer_params.weight_dims[0] * layer_params.weight_dims[1]], dtype=np.int64
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=weight_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[weight.dtype],
                    dims=weight.shape,
                    vals=weight.flatten(),
                )
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=expand_shape_name,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[expand_shape.dtype],
                    dims=expand_shape.shape,
                    vals=expand_shape.flatten(),
                )
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=shape_name1,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape1.dtype],
                    dims=shape1.shape,
                    vals=shape1.flatten(),
                )
            )
            self.__initializers.append(
                helper.make_tensor(
                    name=shape_name2,
                    data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[shape2.dtype],
                    dims=shape2.shape,
                    vals=shape2.flatten(),
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Flatten",
                    inputs=layer_params.bottom_names,
                    outputs=[flatten_name],
                    axis=0,
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Transpose", inputs=[flatten_name], outputs=[transpose_name]
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Expand",
                    inputs=[transpose_name, expand_shape_name],
                    outputs=[expand_name],
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Reshape", inputs=[expand_name, shape_name1], outputs=[reshape_name]
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Mul", inputs=[reshape_name, weight_name], outputs=[multiply_name]
                )
            )
            self.__nodes.append(
                helper.make_node(
                    op_type="Reshape",
                    inputs=[multiply_name, shape_name2],
                    outputs=layer_params.top_names,
                )
            )
        else:
            raise ValueError(layer_type + " is not supported in HugeCTR to ONNX converter")
        self.__counter += 1

    def create_graph(self, name="hugectr_graph"):
        # Create the graph (GraphProto)
        self.__graph_def = helper.make_graph(
            self.__nodes, name, self.__inputs, self.__outputs, self.__initializers
        )

    def save_model(self, model_path, op_version=10, ir_version=7):
        # Create the model (ModelProto)
        model_def = helper.make_model(self.__graph_def)
        model_def.opset_import[0].version = op_version
        model_def.ir_version = ir_version
        onnx.checker.check_model(model_def)
        print("[HUGECTR2ONNX][INFO]: The model is checked!")
        onnx.save(model_def, model_path)
        print("[HUGECTR2ONNX][INFO]: The model is saved at {}".format(model_path))
