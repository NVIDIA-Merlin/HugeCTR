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

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeAndType;
using shape_inference::ShapeHandle;

REGISTER_OP("DummyVarHandle")
    .Attr("container: string = 'DummyVarContainer'")
    .Attr("shared_name: string")
    .Attr("shape: shape")
    .Attr("key_type: {int32, int64} = DT_INT64")
    .Attr("dtype: {float32} = DT_FLOAT")
    .Output("resource: resource")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Scalar());

      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("dtype", &t));

      PartialTensorShape shape;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape));
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shape, &s));

      // Shape must be rank 2
      ShapeHandle s_known_rank;
      TF_RETURN_IF_ERROR(c->WithRank(s, 2, &s_known_rank));

      // Shape[1] must > 0
      DimensionHandle dim_1 = c->DimKnownRank(s_known_rank, 1);
      if (!c->ValueKnown(dim_1)) {
        return errors::InvalidArgument("shape[1] must known");
      }
      if (c->Value(dim_1) <= 0) {
        return errors::InvalidArgument("shape[1] must > 0");
      }

      std::vector<ShapeAndType> st;
      st.push_back({s_known_rank, t});
      c->set_output_handle_shapes_and_types(0, st);

      return Status::OK();
    });

REGISTER_OP("DummyVarInitialize")
    .Input("resource: resource")
    .Input("initializer: init_dtype")
    .Attr("var_type: string")
    .Attr("init_dtype: {float32, string}")
    .Attr("unique_name: string")
    .Attr("key_type: {int32, int64} = DT_INT64")
    .Attr("dtype: {float32} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); });

REGISTER_OP("DummyVarShape")
    .Input("input: resource")
    .Output("output: out_type")
    .Attr("out_type: {int32, int64} = DT_INT32")
    .Attr("key_type: {int32, int64} = DT_INT64")
    .Attr("dtype: {float32} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(2));
      return Status::OK();
    });

}  // namespace tensorflow
