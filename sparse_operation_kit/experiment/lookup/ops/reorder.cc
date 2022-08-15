/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("Reorder")
    .Input("embedding: dtype")
    .Input("order: int32")
    .Output("output: dtype")
    .Attr("dtype: {float32, float16} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      // Step 1: Check shape of embedding,
      //         rank(embedding) must be 2
      ShapeHandle embedding_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &embedding_shape));
      // Step 2: Check shape of order,
      //         rank(order) must be 1
      ShapeHandle order_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &order_shape));
      // Step 3: Assert(embedding_shape[0] == order_shape[0])
      auto num_keys = c->Value(c->Dim(embedding_shape, 0));
      auto num_orders = c->Value(c->Dim(order_shape, 0));
      if (num_keys != -1 && num_orders != -1 && num_keys != num_orders) {
        return errors::InvalidArgument("embedding.shape[0] != order.shape[0]");
      }
      // Step 4: Set shape of output
      c->set_output(0, embedding_shape);
      return Status::OK();
    });

REGISTER_OP("GatherEx")
    .Input("grads: dtype")
    .Input("indices: int32")
    .Output("output: dtype")
    .Attr("dtype: {float32, float16} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      // Step 1: Check shape of grads,
      //         rank(grads) must be 2
      ShapeHandle grads_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &grads_shape));
      // Step 2: Check shape of indices,
      //         rank(indices) must be 1
      ShapeHandle indices_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &indices_shape));
      // Step 3: Assert(grads_shape[0] == indices_shape[0])
      auto num_keys = c->Value(c->Dim(grads_shape, 0));
      auto num_indices = c->Value(c->Dim(indices_shape, 0));
      if (num_keys != -1 && num_indices != -1 && num_keys != num_indices) {
        return errors::InvalidArgument("grads.shape[0] != indices.shape[0]");
      }
      // Step 4: Set shape of output
      c->set_output(0, grads_shape);
      return Status::OK();
    });

}  // namespace tensorflow
