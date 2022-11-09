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

REGISTER_OP("HotnessCalculate")
    .Input("row_length_buffer: Tindices")
    .Output("hotness: int32")
    .Attr("num_gpus: int")
    .Attr("num_lookups: int")
    .Attr("Tindices: {int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) { 
        ShapeHandle unknown_1d_shape = c->UnknownShapeOfRank(1);
        
        c->set_output(0, unknown_1d_shape);
        
        return Status::OK(); 
    });

REGISTER_OP("PreprocessingForward")
    .Input("keys: num_lookups * Tindices")
    .Input("row_lengths: num_lookups * Toffsets")
    .Output("key_send_buffer: Tindices")
    .Output("row_length_send_buffer: Toffsets")
    .Attr("num_lookups: int")
    .Attr("combiners: list(string)")
    .Attr("shard: list(int)")
    .Attr("dimensions: list(int)")
    .Attr("rank: int")
    .Attr("num_ranks: int")
    .Attr("id_in_local_rank: int")
    .Attr("num_gpus: int")
    .Attr("Tindices: {int32, int64} = DT_INT64")
    .Attr("Toffsets: {int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) { 
        ShapeHandle unknown_1d_shape = c->UnknownShapeOfRank(1);
        
        c->set_output(0, unknown_1d_shape);
        c->set_output(1, unknown_1d_shape);
        
        return Status::OK(); 
    });

// There may be duplicates in the `handles`
REGISTER_OP("LookupForward")
    .Input("handles: num_lookups * resource")
    .Input("key_recv_buffer: Tindices")
    .Input("row_length_recv_buffer: Toffsets")
    .Input("hotness: int32")
    .Output("emb_vec_buffer: num_gpus * dtype")
    .Output("model_key: Tindices")
    .Output("model_offsets: uint32")
    .Attr("num_lookups: int")
    .Attr("combiners: list(string)")
    .Attr("shard: list(int)")
    .Attr("dimensions: list(int)")
    .Attr("rank: int")
    .Attr("num_ranks: int")
    .Attr("id_in_local_rank: int")
    .Attr("num_gpus: int")
    .Attr("Tindices: {int32, int64} = DT_INT64")
    .Attr("Toffsets: {int32, int64} = DT_INT64")
    .Attr("dtype: {float32, float16} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) { 
        int num_lookups;
        TF_RETURN_IF_ERROR(c->GetAttr("num_lookups", &num_lookups));
        
        int num_gpus;
        TF_RETURN_IF_ERROR(c->GetAttr("num_gpus", &num_gpus));
        
        std::vector<int> dimensions;
        TF_RETURN_IF_ERROR(c->GetAttr("dimensions", &dimensions));
        int sum_dimensions = std::accumulate(dimensions.begin(), dimensions.end(), 0);

        ShapeHandle row_length_recv_buffer = c->input(num_lookups);
        shape_inference::DimensionHandle row_length_recv_buffer_dimension = c->NumElements(row_length_recv_buffer);
        shape_inference::DimensionHandle global_batch_size;
        TF_RETURN_IF_ERROR(c->Divide(row_length_recv_buffer_dimension, num_lookups, true, &global_batch_size));
        
        shape_inference::DimensionHandle batch_size;
        TF_RETURN_IF_ERROR(c->Divide(global_batch_size, num_gpus, true, &batch_size));

        shape_inference::DimensionHandle emb_vec_buffer_size;
        TF_RETURN_IF_ERROR(c->Multiply(batch_size, sum_dimensions, &emb_vec_buffer_size));
        
        ShapeHandle emb_vec_buffer_shape = c->MakeShape({emb_vec_buffer_size});
        for  (int i = 0; i < num_gpus; ++i) {
            c->set_output(i, emb_vec_buffer_shape);
        }

        return Status::OK();  
    });

// There may be duplicates in the `handles`
REGISTER_OP("LookupForwardDynamic")
    .Input("handles: num_lookups * resource")
    .Input("key_recv_buffer: Tindices")
    .Input("row_length_recv_buffer: Toffsets")
    .Input("hotness: int32")
    .Output("emb_vec_buffer: num_gpus * dtype")
    .Output("model_key: Tindices")
    .Output("model_offsets: uint32")
    .Attr("num_lookups: int")
    .Attr("combiners: list(string)")
    .Attr("shard: list(int)")
    .Attr("dimensions: list(int)")
    .Attr("rank: int")
    .Attr("num_ranks: int")
    .Attr("id_in_local_rank: int")
    .Attr("num_gpus: int")
    .Attr("Tindices: {int32, int64} = DT_INT64")
    .Attr("Toffsets: {int32, int64} = DT_INT64")
    .Attr("dtype: {float32, float16} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) { 
        int num_lookups;
        TF_RETURN_IF_ERROR(c->GetAttr("num_lookups", &num_lookups));
        
        int num_gpus;
        TF_RETURN_IF_ERROR(c->GetAttr("num_gpus", &num_gpus));
        
        std::vector<int> dimensions;
        TF_RETURN_IF_ERROR(c->GetAttr("dimensions", &dimensions));
        int sum_dimensions = std::accumulate(dimensions.begin(), dimensions.end(), 0);

        ShapeHandle row_length_recv_buffer = c->input(num_lookups);
        shape_inference::DimensionHandle row_length_recv_buffer_dimension = c->NumElements(row_length_recv_buffer);
        shape_inference::DimensionHandle global_batch_size;
        TF_RETURN_IF_ERROR(c->Divide(row_length_recv_buffer_dimension, num_lookups, true, &global_batch_size));
        
        shape_inference::DimensionHandle batch_size;
        TF_RETURN_IF_ERROR(c->Divide(global_batch_size, num_gpus, true, &batch_size));

        shape_inference::DimensionHandle emb_vec_buffer_size;
        TF_RETURN_IF_ERROR(c->Multiply(batch_size, sum_dimensions, &emb_vec_buffer_size));
        
        ShapeHandle emb_vec_buffer_shape = c->MakeShape({emb_vec_buffer_size});
        for  (int i = 0; i < num_gpus; ++i) {
            c->set_output(i, emb_vec_buffer_shape);
        }

        return Status::OK(); 
    });

#ifdef GOOGLE_CUDA
#ifdef TENSORFLOW_USE_GPU_EV
// There may be duplicates in the `handles`
REGISTER_OP("LookupForwardEmbeddingVarGPU")
    .Input("handles: num_lookups * resource")
    .Input("key_recv_buffer: Tindices")
    .Input("row_length_recv_buffer: Toffsets")
    .Input("hotness: int32")
    .Output("emb_vec_buffer: num_gpus * dtype")
    .Output("model_key: Tindices")
    .Output("model_offsets: uint32")
    .Attr("num_lookups: int")
    .Attr("combiners: list(string)")
    .Attr("shard: list(int)")
    .Attr("dimensions: list(int)")
    .Attr("rank: int")
    .Attr("num_ranks: int")
    .Attr("id_in_local_rank: int")
    .Attr("num_gpus: int")
    .Attr("Tindices: {int32, int64} = DT_INT64")
    .Attr("Toffsets: {int32, int64} = DT_INT64")
    .Attr("dtype: {float32, float16} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
        int num_lookups;
        TF_RETURN_IF_ERROR(c->GetAttr("num_lookups", &num_lookups));
        
        int num_gpus;
        TF_RETURN_IF_ERROR(c->GetAttr("num_gpus", &num_gpus));
        
        std::vector<int> dimensions;
        TF_RETURN_IF_ERROR(c->GetAttr("dimensions", &dimensions));
        int sum_dimensions = std::accumulate(dimensions.begin(), dimensions.end(), 0);

        ShapeHandle row_length_recv_buffer = c->input(num_lookups);
        shape_inference::DimensionHandle row_length_recv_buffer_dimension = c->NumElements(row_length_recv_buffer);
        shape_inference::DimensionHandle global_batch_size;
        TF_RETURN_IF_ERROR(c->Divide(row_length_recv_buffer_dimension, num_lookups, true, &global_batch_size));
        
        shape_inference::DimensionHandle batch_size;
        TF_RETURN_IF_ERROR(c->Divide(global_batch_size, num_gpus, true, &batch_size));

        shape_inference::DimensionHandle emb_vec_buffer_size;
        TF_RETURN_IF_ERROR(c->Multiply(batch_size, sum_dimensions, &emb_vec_buffer_size));
        
        ShapeHandle emb_vec_buffer_shape = c->MakeShape({emb_vec_buffer_size});
        for  (int i = 0; i < num_gpus; ++i) {
            c->set_output(i, emb_vec_buffer_shape);
        }

        return Status::OK(); 
    });
#endif
#endif

REGISTER_OP("LookupBackward")
    .Input("emb_vec_buffer_grad: num_gpus * dtype")
    .Input("model_key: Tindices")
    .Input("model_offsets: uint32")
    .Input("hotness: int32")
    .Output("unique_key: num_lookups * Tindices")
    .Output("grad: num_lookups * dtype")
    .Attr("num_lookups: int")
    .Attr("combiners: list(string)")
    .Attr("shard: list(int)")
    .Attr("dimensions: list(int)")
    .Attr("rank: int")
    .Attr("num_ranks: int")
    .Attr("id_in_local_rank: int")
    .Attr("num_gpus: int")
    .Attr("Tindices: {int32, int64} = DT_INT64")
    .Attr("Toffsets: {int32, int64} = DT_INT64")
    .Attr("dtype: {float32, float16} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
        int num_lookups;
        TF_RETURN_IF_ERROR(c->GetAttr("num_lookups", &num_lookups));

        std::vector<int> dimensions;
        TF_RETURN_IF_ERROR(c->GetAttr("dimensions", &dimensions));

        for  (int i = 0; i < num_lookups; ++i) {
            shape_inference::DimensionHandle num_unique_key = c->UnknownDim();
            
            ShapeHandle unique_key_shape = c->MakeShape({num_unique_key});
            c->set_output(i, unique_key_shape);
            ShapeHandle wgrad_shape = c->MakeShape({num_unique_key, dimensions[i]});
            c->set_output(num_lookups + i, wgrad_shape);
        }

        return Status::OK(); 
    });

REGISTER_OP("PostprocessingForward")
    .Input("emb_vec_buffer: num_gpus * dtype")
    .Input("row_lengths: num_lookups * Toffsets")
    .Input("hotness: int32")
    .Output("emb_vec: num_lookups * dtype")
    .Output("emb_vec_buffer_shape: int64")
    .Attr("num_lookups: int")
    .Attr("combiners: list(string)")
    .Attr("shard: list(int)")
    .Attr("dimensions: list(int)")
    .Attr("rank: int")
    .Attr("num_ranks: int")
    .Attr("id_in_local_rank: int")
    .Attr("num_gpus: int")
    .Attr("Tindices: {int32, int64} = DT_INT64")
    .Attr("Toffsets: {int32, int64} = DT_INT64")
    .Attr("dtype: {float32, float16} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
        int num_lookups;
        TF_RETURN_IF_ERROR(c->GetAttr("num_lookups", &num_lookups));
        
        int num_gpus;
        TF_RETURN_IF_ERROR(c->GetAttr("num_gpus", &num_gpus));
        
        std::vector<int> dimensions;
        TF_RETURN_IF_ERROR(c->GetAttr("dimensions", &dimensions));

        ShapeHandle row_lengths = c->input(num_gpus);
        shape_inference::DimensionHandle batch_size = c->NumElements(row_lengths);

        for  (int i = 0; i < num_lookups; ++i) {
            ShapeHandle emb_vec_shape = c->MakeShape({batch_size, dimensions[i]});
            c->set_output(i, emb_vec_shape);
        }
        c->set_output(num_lookups, c->MakeShape({num_gpus}));

        return Status::OK(); 
    });

REGISTER_OP("PostprocessingBackward")
    .Input("emb_vec_grad: num_lookups * dtype")
    .Input("emb_vec_buffer_shape: int64")
    .Input("row_lengths: num_lookups * Toffsets")
    .Input("hotness: int32")
    .Output("emb_vec_buffer_grad: num_gpus * dtype")
    .Attr("num_lookups: int")
    .Attr("combiners: list(string)")
    .Attr("shard: list(int)")
    .Attr("dimensions: list(int)")
    .Attr("rank: int")
    .Attr("num_ranks: int")
    .Attr("id_in_local_rank: int")
    .Attr("num_gpus: int")
    .Attr("Tindices: {int32, int64} = DT_INT64")
    .Attr("Toffsets: {int32, int64} = DT_INT64")
    .Attr("dtype: {float32, float16} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) { 
        int num_gpus;
        TF_RETURN_IF_ERROR(c->GetAttr("num_gpus", &num_gpus));
        
        ShapeHandle unknown_1d_shape = c->UnknownShapeOfRank(1);
        for  (int i = 0; i < num_gpus; ++i) {
            c->set_output(i, unknown_1d_shape);
        }
        return Status::OK(); 
    });

}  // namespace tensorflow
