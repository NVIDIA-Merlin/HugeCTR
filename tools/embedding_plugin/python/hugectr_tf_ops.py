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
#TODO: export PYTHONPATH=$PYTHONPATH:/workspace/hugectr/tools/embedding_plugin/python

from tensorflow.python.framework import load_library, ops, dtypes
# from mpi4py import MPI 

lib_name = r"libembedding_plugin.so"

paths = [r'../../build/lib/',
         r'../../build/build_single/lib/',
         r'/usr/local/hugectr/lib/',
         r'/workspace/hugectr/build/lib/',
         r'/workspace/home/hugectr/build/lib/']

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
    raise FileNotFoundError("Could not find libembedding_plugin.so")

print("[INFO]: loadding from %s" %lib_file)

embedding_plugin_ops = load_library.load_op_library(lib_file)
# for item in dir(embedding_plugin_ops):
    # print(item)

init = embedding_plugin_ops.hugectr_init
reset = embedding_plugin_ops.hugectr_reset
create_embedding = embedding_plugin_ops.hugectr_create_embedding

fprop = embedding_plugin_ops.hugectr_embedding_fprop
bprop = embedding_plugin_ops.hugectr_embedding_bprop

distribute_keys = embedding_plugin_ops.hugectr_embedding_distribute_keys
distribute_keys_gpu = embedding_plugin_ops.hugectr_embedding_distribute_keys_gpu

fprop_v2 = embedding_plugin_ops.hugectr_embedding_fprop_v2
fprop_v3 = embedding_plugin_ops.hugectr_embedding_fprop_v3
fprop_v4 = embedding_plugin_ops.hugectr_embedding_fprop_v4

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

@ops.RegisterGradient("HugectrEmbeddingFpropV4")
def _HugectrEmbeddingGradV4(op, top_grad):
    embedding_name = op.inputs[0]
    bp_trigger = op.inputs[-1]
    bp_trigger_grad = bprop(embedding_name, top_grad, bp_trigger)
    return tuple(None for _ in range(len(op.inputs)-1)) + (bp_trigger_grad,)
