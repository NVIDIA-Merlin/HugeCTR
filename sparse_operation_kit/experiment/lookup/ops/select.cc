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

REGISTER_OP("DistSelect")
    .Input("indices: Tindices")
    .Output("output: Tindices")
    .Output("order: int32")
    .Output("splits: int32")
    .Attr("num_splits: int")
    .Attr("Tindices: {int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) {
      // Step 1: Check shape of indices
      //         rank(indices) must be 1
      ShapeHandle indices_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &indices_shape));
      // Step 2: Check size(number of GPUs)
      int num_splits;
      TF_RETURN_IF_ERROR(c->GetAttr("num_splits", &num_splits));
      if (num_splits <= 0) {
        return errors::InvalidArgument("num_splits must > 0");
      }
      // Step 3: Set shape of output
      c->set_output(0, indices_shape);
      c->set_output(1, indices_shape);
      c->set_output(2, c->Vector(num_splits));
      return Status::OK();
    });

}  // namespace tensorflow
