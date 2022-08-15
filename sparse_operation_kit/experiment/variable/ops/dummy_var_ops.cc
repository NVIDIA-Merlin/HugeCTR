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

REGISTER_OP("DummyVarAssign")
    .Input("resource: resource")
    .Input("indices: key_type")
    .Input("values: dtype")
    .Attr("key_type: {int32, int64}")
    .Attr("dtype: {float32}")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); });

REGISTER_OP("DummyVarExport")
    .Input("resource: resource")
    .Output("indices: key_type")
    .Output("values: dtype")
    .Attr("key_type: {int32, int64} = DT_INT64")
    .Attr("dtype: {float32} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); });

REGISTER_OP("DummyVarSparseRead")
    .Input("resource: resource")
    .Input("indices: key_type")
    .Output("output: dtype")
    .Attr("key_type: {int32, int64}")
    .Attr("dtype: {float32, float16} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      // Get handle.shape[1]
      auto handle_shapes_and_types = c->input_handle_shapes_and_types(0);
      auto handle_shape = (*handle_shapes_and_types)[0].shape;
      ShapeHandle handle_shape_1;
      TF_RETURN_IF_ERROR(c->Subshape(handle_shape, 1, 2, &handle_shape_1));

      // rank(indices) should == 1
      ShapeHandle indices_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &indices_shape));

      // Set output shape = [indices.shape[0], handle.shape[1]]
      ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(c->Concatenate(c->input(1), handle_shape_1, &output_shape));
      c->set_output(0, output_shape);

      return Status::OK();
    });

namespace {
Status DummyVarScatterShapeFn(InferenceContext* c) {
  // Get handle.shape[1]
  auto handle_shapes_and_types = c->input_handle_shapes_and_types(0);
  auto handle_shape = (*handle_shapes_and_types)[0].shape;
  ShapeHandle handle_shape_1;
  TF_RETURN_IF_ERROR(c->Subshape(handle_shape, 1, 2, &handle_shape_1));

  // rank(indices) should == 1
  ShapeHandle indices_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &indices_shape));

  // rank(updates) should == 2
  ShapeHandle updates_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &updates_shape));

  // updates.shape should == [indices.shape[0], handle.shape[1]]
  ShapeHandle correct_updates_shape, unused;
  TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, handle_shape_1, &correct_updates_shape));
  TF_RETURN_IF_ERROR(c->Merge(updates_shape, correct_updates_shape, &unused));

  return Status::OK();
}
}  // namespace

REGISTER_OP("DummyVarScatterAdd")
    .Input("resource: resource")
    .Input("indices: key_type")
    .Input("updates: dtype")
    .Attr("key_type: {int32, int64}")
    .Attr("dtype: {float32}")
    .SetShapeFn(DummyVarScatterShapeFn);

REGISTER_OP("DummyVarScatterUpdate")
    .Input("resource: resource")
    .Input("indices: key_type")
    .Input("updates: dtype")
    .Attr("key_type: {int32, int64}")
    .Attr("dtype: {float32}")
    .SetShapeFn(DummyVarScatterShapeFn);

}  // namespace tensorflow
