"""
 Copyright (c) 2023, NVIDIA CORPORATION.
 
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
from hierarchical_parameter_server import Init
from hierarchical_parameter_server import hps_lib
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.distribute as tf_dist
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.ops import array_ops, math_ops, clip_ops
from tensorflow.python.framework import ops

CommToolSet = set(["Strategy", "MPI", "Horovod", "OneDevice"])


def clip(embeddings, ids, max_norm):
    """Helper function for lookup
    This function optionally clips embeddings to an l2-norm of max_norm.
    Args:
        embeddings: A `Tensor` of embeddings retrieved by `gather`.
        ids: The `ids` argument that was passed to `gather`.
        max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
            than this value.
    Returns:
        A `Tensor` with the same type as `embeddings`.
    """

    def _rank(x):
        """Helper function to retrieve the rank of a tensor.
        Args:
            x: Something convertible to `Tensor`.
        Returns:
            Either a pair `(rank, True)` where `rank` is an integer or a pair
            `(rank, False)` where `rank` is an integer `Tensor`. In either case,
            `rank` is the rank of `x`.
        """
        rank = ops.convert_to_tensor(x).get_shape().ndims
        if rank:
            return rank, True
        else:
            return array_ops.rank(x), False

    if max_norm is None:
        return embeddings
    ids_rank, ids_static = _rank(ids)
    embeddings_rank, embeddings_static = _rank(embeddings)
    return clip_ops.clip_by_norm(
        embeddings,
        max_norm,
        axes=(
            list(range(ids_rank, embeddings_rank))
            if ids_static and embeddings_static
            else math_ops.range(ids_rank, embeddings_rank)
        ),
    )


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


def lookup(
    ids,
    model_name,
    table_id,
    emb_vec_size,
    emb_vec_dtype,
    ps_config_file,
    global_batch_size,
    max_norm,
):
    """
    This function is a wrapper of HPS's lookup forward propagation.
    """
    # Lazy initialization of hps
    status = Init(ps_config_file=ps_config_file, global_batch_size=global_batch_size)
    global_replica_id = get_global_replica_id(_get_comm_tool())
    embeddings = hps_lib.lookup(
        values=ids,
        global_replica_id=global_replica_id,
        model_name=model_name,
        table_id=table_id,
        emb_vec_size=emb_vec_size,
        dtype=emb_vec_dtype,
        init_status=status,
    )
    ret = clip(embeddings, ids, max_norm)
    return array_ops.identity(ret)
