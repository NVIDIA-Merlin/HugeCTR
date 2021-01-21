/*
* Copyright (c) 2020, NVIDIA CORPORATION.
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

REGISTER_OP("ThreadingTest")
    .Input("input: int32")
    .Output("output: int32");



REGISTER_OP("V2HugectrInit")
    .Input("visible_gpus: int32")
    .Attr("seed: int = 0")
    .Attr("key_type: {'uint32', 'int64'} = 'int64'")
    .Attr("value_type: {'float', 'half'} = 'float'")
    .Attr("batch_size: int = 1")
    .Attr("batch_size_eval: int = 1");

REGISTER_OP("V2HugectrReset")
    .Doc(R"doc(
    This op is used to explicitly release resources managed by hugectr_tf_ops.
    )doc");

REGISTER_OP("V2HugectrCreateEmbedding")
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

REGISTER_OP("V2HugectrEmbeddingFpropV1")
    .Input("embedding_name: string")
    .Input("replica_id: int32")
    .Input("row_offset: int64")
    .Input("values: int64")
    .Input("nnz: int64")
    .Input("bp_trigger: float")
    .Output("replica_forward_result: float")
    .Attr("is_training: bool = true")
    .Attr("input_buffer_reset: bool = false");

REGISTER_OP("V2HugectrEmbeddingBprop")
    .Input("embedding_name: string")
    .Input("replica_id: int32")
    .Input("replica_top_gradients: grad_type")
    .Input("bp_trigger: float")
    .Output("bp_trigger_grad: float")
    .Attr("grad_type: {float, half}");

REGISTER_OP("V2HugectrBroadcastThenConvertToCSR")
    .Input("embedding_name: string")
    .Input("row_indices: int64")
    .Input("values: int64")
    .Output("to_each_replica: T")
    .Attr("T: list({int32})")
    .Attr("is_training: bool = true")
    .Doc(R"doc(
        This op is used to broad cast input data to all devices.
        It must be call only once in each iteration.
        To reduce the amount of data transfering from CPU->GPUs, only one op should be created and called. 
    )doc");

REGISTER_OP("V2HugectrEmbeddingFpropV2")
    .Input("embedding_name: string")
    .Input("replica_id: int32")
    .Input("to_each_replica: int32")
    .Input("bp_trigger: float")
    .Output("replica_forward_result: float")
    .Attr("is_training: bool = true")
    .Doc(R"doc(
        This op is used to do forward propagation.
        It must be called inner the scope of Mirrored Strategy.
    )doc");


REGISTER_OP("V2HugectrEmbeddingSave")
    .Input("embedding_name: string")
    .Input("save_name: string");

REGISTER_OP("V2HugectrEmbeddingRestore")
    .Input("embedding_name: string")
    .Input("file_name: string");