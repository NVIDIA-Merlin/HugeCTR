/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>

using namespace tensorflow;
using namespace tensorflow::shape_inference;

REGISTER_OP("Lookup")
    .Input("values: value_dtype")
    .Input("global_replica_id: int32")
    .Output("emb_vector: dtype")
    .Attr("value_dtype: {int32, int64}")
    .Attr("model_name: string")
    .Attr("table_id: int")
    .Attr("emb_vec_size: int")
    .Attr("dtype: {float32}")
    .Input("init_status: status_dtype")
    .Attr("status_dtype: {string}")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle input_shape_0 = ctx->input(0);

      ShapeHandle input_shape_1 = ctx->input(1);
      DimensionHandle input_num_elem_1 = ctx->NumElements(input_shape_1);
      if (1 != ctx->Value(input_num_elem_1)) {
        return errors::InvalidArgument("global_replica_id must be a scalar.");
      }

  // Values must be of dense key tensors, but are not necessarily 1-D.
  // ShapeHandle values_shape;
  // TF_RETURN_IF_ERROR(ctx->WithRankAtMost(ctx->input(0), 1, &values_shape));
#ifndef TF_GE_211
      return Status::OK();
#else
      return OkStatus();
#endif
    });
