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
from hierarchical_parameter_server import hps_lib
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.distribute as tf_dist
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.ops import array_ops


CommToolSet = set(["Strategy", "MPI", "Horovod", "OneDevice"])


def get_global_replica_id(comm_tool=None, var=None):
    def _strategy():
        def _get_current_replica_id_in_group_sync():
            replica_ctx = tf_dist.get_replica_context()
            if replica_ctx:
                replica_id = replica_ctx.replica_id_in_sync_group
            else:
                replica_id = (
                    distribute_lib.get_update_replica_id()
                    if hps_lib.in_tensorflow2()
                    else distribute_lib.get_update_device()
                )
            if replica_id is None:
                replica_id = array_ops.constant(0, dtype=array_ops.dtypes.int32)
            return replica_id

        return _get_current_replica_id_in_group_sync()

    def _MPI():
        return int(os.getenv("OMPI_COMM_WORLD_RANK"))

    def _Horovod():
        import horovod.tensorflow as hvd

        return hvd.local_rank()

    def _OneDevice():
        return 0

    if comm_tool is None:
        raise RuntimeError(
            "HPS can only works with " "tf.distribute.Strategy, MPI, Horovod or single GPU."
        )

    if comm_tool not in CommToolSet:
        raise RuntimeError(
            "HPS only works with tf.distribute.Strategy, "
            "MPI, Horovod or single GPU. But got %s" % comm_tool
        )

    if "Strategy" == comm_tool:
        return _strategy()
    elif "MPI" == comm_tool:
        return _MPI()
    elif "Horovod" == comm_tool:
        return _Horovod()
    elif "OneDevice" == comm_tool:
        return _OneDevice()


def _get_comm_tool():
    if "horovod.tensorflow" in sys.modules:
        return "Horovod"
    elif tf_dist.has_strategy():
        return "Strategy"
    elif os.getenv("OMPI_COMM_WORLD_SIZE") is not None:
        return "MPI"
    else:
        return "OneDevice"


def lookup(values, model_name, table_id, emb_vec_size, emb_vec_dtype=tf.float32):
    """
    This function is a wrapper of HPS's lookup forward propagation.
    """
    global_replica_id = get_global_replica_id(_get_comm_tool())
    vector = hps_lib.lookup(
        values=values,
        global_replica_id=global_replica_id,
        model_name=model_name,
        table_id=table_id,
        emb_vec_size=emb_vec_size,
        dtype=emb_vec_dtype,
    )
    return vector
