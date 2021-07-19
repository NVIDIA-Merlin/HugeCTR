/*
* Copyright (c) 2021, NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;
using namespace tensorflow::shape_inference;

REGISTER_OP("HugectrInit")
    .Input("visiable_gpus: int32")
    .Attr("seed: int = 0")
    .Attr("key_type: {'uint32', 'int64'} = 'int64'")
    .Attr("value_type: {'float', 'half'} = 'float'")
    .Attr("batch_size: int = 1")
    .Attr("batch_size_eval: int = 1");

REGISTER_OP("HugectrReset")
    .Doc(R"doc(
    This op is used to explicitly release resources managed by hugectr_tf_ops.
    )doc");

REGISTER_OP("HugectrCreateEmbedding")
    .Input("init_value: T")
    .Output("embedding_name: string")
    .Attr("T: {float, bool}")
    .Attr("name_: string = 'hugectr_embedding'")
    .Attr("embedding_type: {'distributed', 'localized'} = 'localized'")
    .Attr("optimizer_type: {'Adam', 'MomentumSGD', 'Nesterov', 'SGD'} = 'Adam'")
    .Attr("max_vocabulary_size_per_gpu: int = 1")
    .Attr("slot_size_array: list({uint64}) = []")
    .Attr("opt_hparams: list(float) = [0.001]")
    .Attr("update_type: {'Local', 'Global', 'LazyGlobal'} = 'Local'")
    .Attr("atomic_update: bool = true")
    .Attr("scaler: float = 1.0")
    .Attr("slot_num: int = 1")
    .Attr("max_nnz: int = 1")
    .Attr("max_feature_num: int = 1000")
    .Attr("embedding_vec_size: int = 1")
    .Attr("combiner: {'mean', 'sum'} = 'sum'");

REGISTER_OP("HugectrEmbeddingFprop")
    .Input("sparse_indices: int64")
    .Input("values: value_type")
    .Input("dense_shape: int64")
    .Input("embedding_name: string")
    .Input("bp_trigger: float")
    .Output("forward_result: output_type")
    .Attr("value_type: {uint32, int64}")
    .Attr("output_type: {float, half}")
    .Attr("is_training: bool = true");

REGISTER_OP("HugectrEmbeddingBprop")
    .Input("embedding_name: string")
    .Input("top_gradients: grad_type")
    .Input("bp_trigger: float")
    .Output("bp_trigger_grad: float")
    .Attr("grad_type: {float, half}");

REGISTER_OP("HugectrEmbeddingDistributeKeys")
    .Input("indices: int64")
    .Input("values: int64")
    .Input("dense_shape: int64")
    .Output("row_offsets: gpu_count * int64")
    .Output("value_tensors: gpu_count * int64")
    .Output("nnz_array: int64")
    .Attr("gpu_count: int >= 1 = 1")
    .Attr("embedding_type: {'distributed', 'localized'} = 'localized'")
    .Attr("max_feature_num: int = 1000000")
    .Attr("max_nnz: int");

REGISTER_OP("HugectrEmbeddingFpropV2")
    .Input("embedding_name: string")
    .Input("row_offsets: gpu_count * int64")
    .Input("value_tensors: gpu_count * int64")
    .Input("nnz_array: int64")
    .Input("bp_trigger: float")
    .Output("forward_result: float")
    .Attr("is_training: bool = true")
    .Attr("gpu_count: int >= 1 = 1")
    .Attr("output_shape: list(int)")
    .SetShapeFn([](InferenceContext* ctx) {
        std::vector<int64> output_shape_attr;
        TF_RETURN_IF_ERROR(ctx->GetAttr("output_shape", &output_shape_attr));
        if (output_shape_attr.size() != 3)  return errors::Aborted(__FILE__, ":", __LINE__, " ",
                                            "output_shape should be [batchsize, slot_num, embedding_vec_size].");
        std::vector<DimensionHandle> dims;
        for (const auto shape : output_shape_attr) {
            DimensionHandle dim = ctx->MakeDim(shape);
            dims.emplace_back(dim);
        }

        ShapeHandle output_shape = ctx->MakeShape(dims);
        ctx->set_output(0, output_shape);
        return Status::OK();
    });
    


REGISTER_OP("HugectrEmbeddingFpropV3")
    .Input("embedding_name: string")
    .Input("row_offsets: int64")
    .Input("value_tensors: int64")
    .Input("nnz_array: int64")
    .Input("bp_trigger: float")
    .Output("forward_result: float")
    .Attr("is_training: bool = true")
    .Attr("output_shape: list(int)")
    .SetShapeFn([](InferenceContext* ctx) {
        std::vector<int64> output_shape_attr;
        TF_RETURN_IF_ERROR(ctx->GetAttr("output_shape", &output_shape_attr));
        if (output_shape_attr.size() != 3)  return errors::Aborted(__FILE__, ":", __LINE__, " ",
                                            "output_shape should be [batchsize, slot_num, embedding_vec_size].");
        std::vector<DimensionHandle> dims;
        for (const auto shape : output_shape_attr) {
            DimensionHandle dim = ctx->MakeDim(shape);
            dims.emplace_back(dim);
        }

        ShapeHandle output_shape = ctx->MakeShape(dims);
        ctx->set_output(0, output_shape);
        return Status::OK();
    });


REGISTER_OP("HugectrEmbeddingFpropV4")
    .Input("embedding_name: string")
    .Input("row_indices: int64")
    .Input("values: int64")
    .Input("bp_trigger: float")
    .Output("forward_result: float")
    .Attr("is_training: bool = true")
    .Attr("output_shape: list(int)")
    .SetShapeFn([](InferenceContext* ctx) {
        std::vector<int64> output_shape_attr;
        TF_RETURN_IF_ERROR(ctx->GetAttr("output_shape", &output_shape_attr));
        if (output_shape_attr.size() != 3)  return errors::Aborted(__FILE__, ":", __LINE__, " ",
                                            "output_shape should be [batchsize, slot_num, embedding_vec_size].");
        std::vector<DimensionHandle> dims;
        for (const auto shape : output_shape_attr) {
            DimensionHandle dim = ctx->MakeDim(shape);
            dims.emplace_back(dim);
        }

        ShapeHandle output_shape = ctx->MakeShape(dims);
        ctx->set_output(0, output_shape);
        return Status::OK();
    });

REGISTER_OP("HugectrEmbeddingDistributeKeysGpu")
    .Input("embedding_name: string")
    .Input("row_indices: int64")
    .Input("values: int64")
    .Output("row_offsets: int64")
    .Output("value_tensors: int64")
    .Output("nnz_array: int64")
    .Attr("embedding_type: {'distributed', 'localized'} = 'distributed'")
    .Attr("batch_size: int")
    .Attr("slot_num: int")
    .Attr("max_nnz: int")
    .Attr("gpu_count: int");

REGISTER_OP("HugectrEmbeddingSave")
    .Input("embedding_name: string")
    .Input("save_name: string");

REGISTER_OP("HugectrEmbeddingRestore")
    .Input("embedding_name: string")
    .Input("file_name: string");