"""
Copyright (c) 2020, NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from tensorflow.python.framework import load_library, ops, dtypes
import subprocess

lib_file = r"libembedding_plugin.so"
try:
    process = subprocess.run(["find", "/", "-name", lib_file], stdout=subprocess.PIPE, encoding='UTF-8')
    lib_file = process.stdout.strip()
except subprocess.CalledProcessError as error:
    print(error)

embedding_plugin_ops = load_library.load_op_library(lib_file)
# for item in dir(embedding_plugin_ops):
    # print(item)

init = embedding_plugin_ops.hugectr_init
create_embedding = embedding_plugin_ops.hugectr_create_embedding

fprop = embedding_plugin_ops.hugectr_embedding_fprop
bprop = embedding_plugin_ops.hugectr_embedding_bprop

distribute_keys = embedding_plugin_ops.hugectr_embedding_distribute_keys
distribute_keys_v2 = embedding_plugin_ops.hugectr_embedding_distribute_keys_v2
distribute_keys_v3 = embedding_plugin_ops.hugectr_embedding_distribute_keys_v3
distribute_keys_v4 = embedding_plugin_ops.hugectr_embedding_distribute_keys_v4

fprop_v2 = embedding_plugin_ops.hugectr_embedding_fprop_v2
fprop_v3 = embedding_plugin_ops.hugectr_embedding_fprop_v3

save = embedding_plugin_ops.hugectr_embedding_save
restore = embedding_plugin_ops.hugectr_embedding_restore


@ops.RegisterGradient("HugectrEmbeddingFprop")
def _HugectrEmbeddingGrad(op, top_grad):
    embedding_name = op.inputs[3]
    bp_trigger = op.inputs[-1]
    bp_trigger_grad = bprop(embedding_name, top_grad, bp_trigger)
    return tuple(None for _ in range(len(op.inputs)-1)) + (bp_trigger_grad,)

@ops.RegisterGradient("HugectrEmbeddingFpropV2")
def _HugectrEmbeddingGradV2(op, top_grad):
    embedding_name = op.inputs[0]
    bp_trigger = op.inpust[-1]
    bp_trigger_grad = bprop(embedding_name, top_grad, bp_trigger)
    return tuple(None for _ in range(len(op.inputs)-1)) + (bp_trigger_grad,)

@ops.RegisterGradient("HugectrEmbeddingFpropV3")
def _HugectrEmbeddingGradV3(op, top_grad):
    embedding_name = op.inputs[0]
    bp_trigger = op.inputs[-1]
    bp_trigger_grad = bprop(embedding_name, top_grad, bp_trigger)
    return tuple(None for _ in range(len(op.inputs)-1)) + (bp_trigger_grad,)