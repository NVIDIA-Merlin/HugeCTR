#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
from tensorflow.python.framework import load_library

#   When installed with pip, the .so files should be in
# sparse_operation_kit/lib/
#   When installed manually via `make install`, the .so files
# should be in /usr/local/lib/
lib_path = os.path.join(os.path.dirname(__file__), "../lib")
lib_path = os.path.abspath(lib_path)
lib_path = [lib_path, "/usr/local/lib/"]
syspath = [spath + "/sparse_operation_kit/lib" for spath in sys.path]
lib_path.extend(syspath)

raw_ops = None
for path in lib_path:
    file = os.path.join(path, "libsok_experiment.so")
    if os.path.exists(file):
        # The order of loading core, embedding, sok_experiment cannot
        # be changed, because there is a dependency between them:
        # libsok_experiment.so -> libembedding.so -> libcore.so
        load_library.load_op_library(os.path.join(path, "libhugectr_core23.so"))
        load_library.load_op_library(os.path.join(path, "libembedding.so"))
        raw_ops = load_library.load_op_library(file)
        print("[SOK INFO] Import %s" % file)
if raw_ops is None:
    raise Exception("[SOK INFO] libsok_experiment.so is not found")

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

from sparse_operation_kit.experiment.dump_load import dump, load


# a specific code path for dl framework tf2.11.0
import tensorflow
from tensorflow.python.framework import ops


def init(comm_tool="horovod", use_legacy_optimizer=True):
    """
    Abbreviated as ``sok.experiment.init``.

    This function is used to do the initialization of SparseOperationKit (SOK).

    SOK will leverage all available GPUs for current CPU process. Please set
    `CUDA_VISIBLE_DEVICES` or `tf.config.set_visible_devices` to specify which
    GPU(s) are used in this process before launching tensorflow runtime
    and calling this function.

    Currently, these experiment API only support ``horovod`` as the communication
    tool, so ``horovod.init`` must be called before initializing SOK.

    Example code for doing initialization:

    .. code-block:: python

        import tensorflow as tf
        import horovod.tensorflow as hvd
        import sparse_operation_kit.experiment as sok

        hvd.init()
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

        sok.init()

    Parameters
    ----------
    comm_tool: string
            a string to specify which communication tool to use. Default value is "horovod".
    use_legacy_optimizer: bool
            From tensorflow 2.11.0 , keras default optimizer is optimizer experimental. SOK won't support it in future, so if you switch use_legacy_optimizer to True,
            SOK will redefine tensorflow.keras.optimizers to tensorflow.keras.optimizers.legacy(tf.keras.optimizers.optimizer_v2).
            Default value is True, if you want to use new optimizer in the other part in your code , and only use legacy optimizer in SOK, please set to False
    Returns
    -------
    None
    """
    if use_legacy_optimizer:
        try:
            if tensorflow.keras.optimizers.legacy.Optimizer.__name__ == "OptimizerV2":
                tensorflow.keras.optimizers = tensorflow.keras.optimizers.legacy
                tensorflow.optimizers = tensorflow.optimizers.legacy
        except:
            pass

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
