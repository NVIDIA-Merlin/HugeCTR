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

REGISTER_OP("PluginDenseFprop")
    .Input("emb_handle: variant")
    .Input("emb_variable: T")
    .Input("values: value_dtype")
    .Input("global_replica_id: int32")
    .Output("emb_vector: T")
    .Attr("training: bool")
    .Attr("value_dtype: {int64}")
    .Attr("T: {float32}")
    .Attr("unique_op_name: string")
    .SetShapeFn([](InferenceContext* ctx) {
        ShapeHandle variable_shape;
        TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(1), 2, &variable_shape));

        ShapeHandle emb_vec_size_shape;
        TF_RETURN_IF_ERROR(ctx->Subshape(variable_shape, /*start=*/1, /*end=*/2, &emb_vec_size_shape));

        ShapeHandle output_shape;
        TF_RETURN_IF_ERROR(ctx->Concatenate(ctx->input(2), emb_vec_size_shape, &output_shape));
        ctx->set_output(0, output_shape);

        return Status::OK();
    })
    .Doc(R"doc(
        This op can be used for all kinds of embedding forward propagation,
        which requires the unique_op_name to identify different op instance.
        For example:
            vec0 = plugin_dense_fprop(emb_handle0, values, unique_op_name='1')
            vec1 = plugin_dense_fprop(emb_handle1, values, unique_op_name='2')

            where different unique_op_name are set for different embedding layer instance.
    )doc");