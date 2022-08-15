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

REGISTER_OP("GroupLookup")
    .Input("handles: N * resource")
    .Input("indices: N * Tindices")
    .Output("outputs: N * dtype")
    .Attr("N: int")
    .Attr("Tindices: {int32, int64} = DT_INT64")
    .Attr("dtype: {float32, float16} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      int N;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
      for (int i = 0; i < N; ++i) {
        // rank(handle) should be 2
        auto handle_shapes_and_types = c->input_handle_shapes_and_types(i);
        auto handle_shape = (*handle_shapes_and_types)[0].shape;
        ShapeHandle unused;
        TF_RETURN_IF_ERROR(c->WithRank(handle_shape, 2, &unused));

        // rank(indices) should be 1
        ShapeHandle indices_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(N + i), 1, &indices_shape));

        // output shape: (indices.shape[0], handle.shape[1])
        c->set_output(i, c->Matrix(c->Dim(indices_shape, 0), c->Dim(handle_shape, 1)));
      }
      return Status::OK();
    });

}  // namespace tensorflow
