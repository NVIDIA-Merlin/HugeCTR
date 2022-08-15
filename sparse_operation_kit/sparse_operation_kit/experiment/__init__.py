"""
 Copyright (c) 2022, NVIDIA CORPORATION.
 
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

from tensorflow.python.framework import load_library

so_file = r"/usr/local/lib/libsok_experiment.so"
raw_ops = load_library.load_op_library(so_file)
print("[SOK INFO] Import %s" % so_file)


import sparse_operation_kit.experiment.communication
from sparse_operation_kit.experiment.communication import set_comm_tool


from sparse_operation_kit.experiment.distributed_variable import Variable
from sparse_operation_kit.experiment.distributed_variable import DistributedVariable
from sparse_operation_kit.experiment.distributed_variable import LocalizedVariable


from sparse_operation_kit.experiment.dynamic_variable import DynamicVariable
from sparse_operation_kit.experiment.dynamic_variable import assign, export


from sparse_operation_kit.experiment.optimizer import OptimizerWrapper
from sparse_operation_kit.experiment.optimizer import SGD


from sparse_operation_kit.experiment.lookup import lookup_sparse
from sparse_operation_kit.experiment.lookup import all2all_dense_embedding


def init(comm_tool="horovod"):
    set_comm_tool(comm_tool)
    print("[SOK INFO] Initialize finished, communication tool: " + comm_tool)


def filter_variables(vars):
    sok_vars, other_vars = [], []
    for v in vars:
        if isinstance(v, DynamicVariable):
            sok_vars.append(v)
        elif isinstance(v, DistributedVariable):
            sok_vars.append(v)
        elif isinstance(v, LocalizedVariable):
            sok_vars.append(v)
        else:
            other_vars.append(v)
    return sok_vars, other_vars
