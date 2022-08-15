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

REGISTER_OP("PreprocessingForward")
    .Input("keys: num_lookups * Tindices")
    .Input("row_lengths: num_lookups * Toffsets")
    .Output("key_send_buffer: Tindices")
    .Output("row_length_send_buffer: Toffsets")
    .Attr("num_lookups: int")
    .Attr("combiners: list(string)")
    .Attr("hotness: list(int)")
    .Attr("shard: list(int)")
    .Attr("dimensions: list(int)")
    .Attr("rank: int")
    .Attr("num_ranks: int")
    .Attr("id_in_local_rank: int")
    .Attr("num_gpus: int")
    .Attr("Tindices: {int32, int64} = DT_INT64")
    .Attr("Toffsets: {int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); });

// There may be duplicates in the `handles`
REGISTER_OP("LookupForward")
    .Input("handles: num_lookups * resource")
    .Input("key_recv_buffer: Tindices")
    .Input("row_length_recv_buffer: Toffsets")
    .Output("emb_vec_buffer: num_gpus * dtype")
    .Output("model_key: Tindices")
    .Output("model_offsets: uint32")
    .Attr("num_lookups: int")
    .Attr("combiners: list(string)")
    .Attr("hotness: list(int)")
    .Attr("shard: list(int)")
    .Attr("dimensions: list(int)")
    .Attr("rank: int")
    .Attr("num_ranks: int")
    .Attr("id_in_local_rank: int")
    .Attr("num_gpus: int")
    .Attr("Tindices: {int32, int64} = DT_INT64")
    .Attr("Toffsets: {int32, int64} = DT_INT64")
    .Attr("dtype: {float32, float16} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); });

// There may be duplicates in the `handles`
REGISTER_OP("LookupForwardDynamic")
    .Input("handles: num_lookups * resource")
    .Input("key_recv_buffer: Tindices")
    .Input("row_length_recv_buffer: Toffsets")
    .Output("emb_vec_buffer: num_gpus * dtype")
    .Output("model_key: Tindices")
    .Output("model_offsets: uint32")
    .Attr("num_lookups: int")
    .Attr("combiners: list(string)")
    .Attr("hotness: list(int)")
    .Attr("shard: list(int)")
    .Attr("dimensions: list(int)")
    .Attr("rank: int")
    .Attr("num_ranks: int")
    .Attr("id_in_local_rank: int")
    .Attr("num_gpus: int")
    .Attr("Tindices: {int32, int64} = DT_INT64")
    .Attr("Toffsets: {int32, int64} = DT_INT64")
    .Attr("dtype: {float32, float16} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); });

REGISTER_OP("LookupBackward")
    .Input("emb_vec_buffer_grad: num_gpus * dtype")
    .Input("model_key: Tindices")
    .Input("model_offsets: uint32")
    .Output("unique_key: num_lookups * Tindices")
    .Output("grad: num_lookups * dtype")
    .Attr("num_lookups: int")
    .Attr("combiners: list(string)")
    .Attr("hotness: list(int)")
    .Attr("shard: list(int)")
    .Attr("dimensions: list(int)")
    .Attr("rank: int")
    .Attr("num_ranks: int")
    .Attr("id_in_local_rank: int")
    .Attr("num_gpus: int")
    .Attr("Tindices: {int32, int64} = DT_INT64")
    .Attr("Toffsets: {int32, int64} = DT_INT64")
    .Attr("dtype: {float32, float16} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); });

REGISTER_OP("PostprocessingForward")
    .Input("emb_vec_buffer: num_gpus * dtype")
    .Input("row_lengths: num_lookups * Toffsets")
    .Output("emb_vec: num_lookups * dtype")
    .Output("emb_vec_buffer_shape: int64")
    .Attr("num_lookups: int")
    .Attr("combiners: list(string)")
    .Attr("hotness: list(int)")
    .Attr("shard: list(int)")
    .Attr("dimensions: list(int)")
    .Attr("rank: int")
    .Attr("num_ranks: int")
    .Attr("id_in_local_rank: int")
    .Attr("num_gpus: int")
    .Attr("Tindices: {int32, int64} = DT_INT64")
    .Attr("Toffsets: {int32, int64} = DT_INT64")
    .Attr("dtype: {float32, float16} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); });

REGISTER_OP("PostprocessingBackward")
    .Input("emb_vec_grad: num_lookups * dtype")
    .Input("emb_vec_buffer_shape: int64")
    .Input("row_lengths: num_lookups * Toffsets")
    .Output("emb_vec_buffer_grad: num_gpus * dtype")
    .Attr("num_lookups: int")
    .Attr("combiners: list(string)")
    .Attr("hotness: list(int)")
    .Attr("shard: list(int)")
    .Attr("dimensions: list(int)")
    .Attr("rank: int")
    .Attr("num_ranks: int")
    .Attr("id_in_local_rank: int")
    .Attr("num_gpus: int")
    .Attr("Tindices: {int32, int64} = DT_INT64")
    .Attr("Toffsets: {int32, int64} = DT_INT64")
    .Attr("dtype: {float32, float16} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); });

}  // namespace tensorflow
