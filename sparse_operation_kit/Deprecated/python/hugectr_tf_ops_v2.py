"""
Copyright (c) 2021, NVIDIA CORPORATION.

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
#TODO: export PYTHONPATH=$PYTHONPATH:/workspace/hugectr/tools/embedding_plugin/python

from tensorflow.python.framework import load_library, ops, dtypes
# from mpi4py import MPI 

lib_name = r"libembedding_plugin_v2.so"

paths = [r'/usr/local/hugectr/lib/']

lib_file = None
for path in paths:
    try:
        file = open(path + lib_name)
        file.close()
        lib_file = path + lib_name
        break
    except FileNotFoundError:
        continue

if lib_file is None:
    raise FileNotFoundError("Could not find %s" %lib_name)

print("[INFO]: loadding from %s" %lib_file)

embedding_plugin_ops_v2 = load_library.load_op_library(lib_file)
# for item in dir(embedding_plugin_ops_v2):
    # print(item)

threading_test = embedding_plugin_ops_v2.threading_test

init = embedding_plugin_ops_v2.v2_hugectr_init
reset = embedding_plugin_ops_v2.v2_hugectr_reset

create_embedding = embedding_plugin_ops_v2.v2_hugectr_create_embedding

fprop_experimental = embedding_plugin_ops_v2.v2_hugectr_embedding_fprop_v1

broadcast_then_convert_to_csr = embedding_plugin_ops_v2.v2_hugectr_broadcast_then_convert_to_csr
fprop = embedding_plugin_ops_v2.v2_hugectr_embedding_fprop_v2

bprop = embedding_plugin_ops_v2.v2_hugectr_embedding_bprop

save = embedding_plugin_ops_v2.v2_hugectr_embedding_save
restore = embedding_plugin_ops_v2.v2_hugectr_embedding_restore

@ops.RegisterGradient("V2HugectrEmbeddingFpropV1")
def _HugectrEmbeddingGradV1(op, top_grad):
    embedding_name = op.inputs[0]
    replica_id = op.inputs[1]
    bp_trigger = op.inputs[-1]
    bp_trigger_grad = bprop(embedding_name, replica_id, top_grad, bp_trigger)
    return tuple(None for _ in range(len(op.inputs)-1)) + (bp_trigger_grad,)

@ops.RegisterGradient("V2HugectrEmbeddingFpropV2")
def _HugectrEmbeddingGradV2(op, top_grad):
    embedding_name = op.inputs[0]
    replica_id = op.inputs[1]
    bp_trigger = op.inputs[-1]
    bp_trigger_grad = bprop(embedding_name, replica_id, top_grad, bp_trigger)
    return tuple(None for _ in range(len(op.inputs)-1)) + (bp_trigger_grad,)