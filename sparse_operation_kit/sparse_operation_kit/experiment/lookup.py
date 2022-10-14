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

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.resource_variable_ops import variable_shape
from tensorflow.python.ops.resource_variable_ops import variable_accessed

from sparse_operation_kit.experiment import raw_ops

from sparse_operation_kit.experiment.communication import rank
from sparse_operation_kit.experiment.communication import num_ranks
from sparse_operation_kit.experiment.communication import id_in_rank
from sparse_operation_kit.experiment.communication import num_gpus
from sparse_operation_kit.experiment.communication import alltoall
from sparse_operation_kit.experiment.communication import allreduce
from sparse_operation_kit.experiment.communication import allgather

from sparse_operation_kit.experiment.distributed_variable import DistributedVariable
from sparse_operation_kit.experiment.distributed_variable import LocalizedVariable

from sparse_operation_kit.experiment.dynamic_variable import DynamicVariable


def group_lookup(params, indices, dtype=None, name=None):
    # Fused-version of tf.nn.embedding_lookup on single GPU
    if not (isinstance(params, list) or isinstance(params, tuple)):
        params = [params]
    if not (isinstance(indices, list) or isinstance(indices, tuple)):
        indices = [indices]
    with ops.name_scope("GroupLookup" if name is None else name) as name:
        for param in params:
            variable_accessed(param)
        handles = [param.handle for param in params]
        outputs = raw_ops.group_lookup(handles, indices, dtype=dtype)
        for i in range(len(outputs)):
            outputs[i] = array_ops.identity(outputs[i])
    return outputs


@tf.RegisterGradient("GroupLookup")
def _GroupLookupGrad(op, *top_grads):
    N = op.get_attr("N")
    grads = []
    for i in range(N):
        handle = op.inputs[i]
        indices = op.inputs[N + i]
        params_shape = variable_shape(handle)
        size = array_ops.expand_dims(array_ops.size(indices), 0)
        values_shape = array_ops.concat([size, params_shape[1:]], 0)
        values = array_ops.reshape(top_grads[i], values_shape)
        indices = array_ops.reshape(indices, size)
        grads.append(tf.IndexedSlices(values, indices, params_shape))
    grads += [None] * N
    return grads


@tf.RegisterGradient("Reorder")
def _ReorderGrad(op, grad):
    indices = op.inputs[1]
    return (raw_ops.gather_ex(grad, indices), None)


def all2all_dense_embedding(param, indices):
    # Filter key
    selected_indices, order, splits = raw_ops.dist_select(indices, num_splits=param.num_gpus)

    # All-to-all of indices
    ex_indices, rsplits = alltoall(selected_indices, splits)
    ex_indices = param.key_map(ex_indices)

    # Local lookup
    embeddings = tf.nn.embedding_lookup(param, ex_indices)

    # All-to-all of embedding vectors
    ex_embeddings, _ = alltoall(embeddings, rsplits)

    # Reorder of embedding vectors
    ex_embeddings = raw_ops.reorder(ex_embeddings, order)

    return ex_embeddings


def _preprocessing_forward(*args, **kwargs):
    """
    This function should not be used by user directly.
    """
    name = kwargs.pop("name") if "name" in kwargs else "PreprocessingForward"
    with ops.name_scope(name) as name:
        return raw_ops.preprocessing_forward(*args, **kwargs)


def _lookup_forward(params, *args, **kwargs):
    """
    This function should not be used by user directly.
    """
    name = kwargs.pop("name") if "name" in kwargs else "LookupForward"
    with ops.name_scope(name) as name:
        for param in params:
            # For tf.GradientTape
            variable_accessed(param)
        handles = [param.handle for param in params]
        if isinstance(params[0], DynamicVariable):
            return raw_ops.lookup_forward_dynamic(handles, *args, **kwargs)
        else:
            return raw_ops.lookup_forward(handles, *args, **kwargs)


@tf.RegisterGradient("LookupForward")
def _LookupBackward(op, *top_grads):
    attr_list = [
        "num_lookups",
        "combiners",
        "hotness",
        "shard",
        "dimensions",
        "rank",
        "num_ranks",
        "id_in_local_rank",
        "Toffsets",
    ]
    kwargs = {}
    for attr in attr_list:
        kwargs[attr] = op.get_attr(attr)

    num_gpus = op.get_attr("num_gpus")
    top_grads = top_grads[:num_gpus]
    other_data = op.outputs[num_gpus:]
    indices, values = raw_ops.lookup_backward(top_grads, *other_data, **kwargs)
    grads = []
    for i in range(len(indices)):
        handle = op.inputs[i]
        params_shape = variable_shape(handle)
        size = array_ops.expand_dims(array_ops.size(indices[i]), 0)
        values_shape = array_ops.concat([size, params_shape[1:]], 0)
        values[i] = tf.reshape(values[i], values_shape)
        if kwargs["shard"][i] < 0 and num_gpus > 1:
            indices[i] = indices[i] // num_gpus
        grads.append(tf.IndexedSlices(values[i], indices[i], params_shape))
    return grads + [None] * (len(op.inputs) - len(grads))


@tf.RegisterGradient("LookupForwardDynamic")
def _LookupDynamicBackward(op, *top_grads):
    attr_list = [
        "num_lookups",
        "combiners",
        "hotness",
        "shard",
        "dimensions",
        "rank",
        "num_ranks",
        "id_in_local_rank",
        "Toffsets",
    ]
    kwargs = {}
    for attr in attr_list:
        kwargs[attr] = op.get_attr(attr)

    num_gpus = op.get_attr("num_gpus")
    top_grads = top_grads[:num_gpus]
    other_data = op.outputs[num_gpus:]
    indices, values = raw_ops.lookup_backward(top_grads, *other_data, **kwargs)
    grads = []
    for i in range(len(indices)):
        handle = op.inputs[i]
        params_shape = raw_ops.dummy_var_shape(handle)
        size = array_ops.expand_dims(array_ops.size(indices[i]), 0)
        values_shape = array_ops.concat([size, params_shape[1:]], 0)
        values[i] = tf.reshape(values[i], values_shape)
        # if kwargs["shard"][i] < 0 and num_gpus > 1:
        #     indices[i] = indices[i] // num_gpus
        grads.append(tf.IndexedSlices(values[i], indices[i], params_shape))
    return grads + [None] * (len(op.inputs) - len(grads))


def _postprocessing_forward(*args, **kwargs):
    """
    This function should not be used by user directly.
    """
    name = kwargs.pop("name") if "name" in kwargs else "PostprocessingForward"
    with ops.name_scope(name) as name:
        return raw_ops.postprocessing_forward(*args, **kwargs)


@tf.RegisterGradient("PostprocessingForward")
def _PostprocessingBackward(op, *top_grads):
    attr_list = [
        "combiners",
        "hotness",
        "shard",
        "dimensions",
        "rank",
        "num_ranks",
        "id_in_local_rank",
        "num_gpus",
        "Tindices",
        # "Toffsets",
    ]
    kwargs = {}
    for attr in attr_list:
        kwargs[attr] = op.get_attr(attr)

    num_lookups = op.get_attr("num_lookups")
    num_gpus = op.get_attr("num_gpus")
    top_grads = top_grads[:num_lookups]
    row_lengths = op.inputs[num_gpus:]
    other_data = op.outputs[num_lookups:]
    grads = raw_ops.postprocessing_backward(top_grads, other_data, row_lengths, **kwargs)
    return grads + [None] * (len(op.inputs) - len(grads))


def to_list(any_obj):
    if not (isinstance(any_obj, list) or isinstance(any_obj, tuple)):
        return [any_obj]
    else:
        return any_obj


def lookup_sparse(params, sp_ids, hotness, combiners):
    """
    Abbreviated as ``sok.experiment.lookup_sparse``.

    Peform fused sparse lookup on the given embedding ``params``. This function
    is similar to the ``tf.nn.embedding_lookup_sparse``, but with two differences:

        - It can do distributed lookup.
        - It can accept multiple params and multiple sp_ids to do fused lookup at once,
          which brings performance benifits.

    Parameters
    ----------
    params: list, tuple
            a list or tuple of trainable *sok.Variable*.
    sp_ids: list, tuple
            a list or tuple of tf.SparseTensor or tf.RaggedTensor.
    hotness: list, tuple
            a list or tuple of int to specify the max hotness of each lookup.
    combiners: list, tuple
            a list or tuple of string to specify the combiner of each lookup.

    Returns
    -------
    emb_vec: list
            a list of tf.Tensor(the results of lookup).

    Example
    -------
    .. code-block:: python

        import numpy as np
        import tensorflow as tf
        import horovod.tensorflow as hvd
        from sparse_operation_kit import experiment as sok

        hvd.init()
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

        sok.init()

        v1 = sok.Variable(np.arange(17 * 3).reshape(17, 3), dtype=tf.float32)
        v2 = sok.Variable(np.arange(7 * 5).reshape(7, 5), dtype=tf.float32)

        indices1 = tf.SparseTensor(
            indices=[[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]],
            values=[1, 1, 3, 4, 5],
            dense_shape=[2, 3])
        )
        indices2 = tf.SparseTensor(
            indices=[[0, 0], [1, 0], [1, 1]],
            values=[1, 2, 3],
            dense_shape=[2, 2]
        )

        embeddings = sok.lookup_sparse(
            [v1, v2], [indices1, indices2], hotness=[3, 2], combiners=["sum", "sum"]
        )
        print(embeddings[0])
        print(embeddings[1])
    """
    # `is_list` determines whether to return a list or a tensor in the end
    is_list = isinstance(sp_ids, list) or isinstance(sp_ids, tuple)

    params = to_list(params)
    sp_ids = to_list(sp_ids)
    hotness = to_list(hotness)
    combiners = to_list(combiners)

    shard, dimensions = [], []
    for param in params:
        shard.append(param.target_gpu)
        dimensions.append(param.shape[1])

    for i in range(1, len(params)):
        if type(params[i]) != type(params[0]):
            raise RuntimeError(
                "Distributed/Localized/Dynamic Variable cannot be used in the same lookup currently"
            )

    keys = []
    row_lengths = []
    for sp_id in sp_ids:
        if isinstance(sp_id, tf.SparseTensor):
            sp_id = tf.RaggedTensor.from_sparse(sp_id)
        keys.append(sp_id.values)
        row_lengths.append(sp_id.row_lengths())

    kwargs = {
        "combiners": combiners,
        "hotness": hotness,
        "shard": shard,
        "dimensions": dimensions,
        "rank": rank(),
        "num_ranks": num_ranks(),
        "id_in_local_rank": id_in_rank(),
    }

    # Step1
    key_send_buffer, row_length_send_buffer = _preprocessing_forward(
        keys, row_lengths, num_gpus=num_gpus(), **kwargs
    )

    # Step2
    if num_gpus() > 1:
        key_recv_buffer = allgather(key_send_buffer)
        row_length_recv_buffer = allgather(row_length_send_buffer)
    else:
        key_recv_buffer = key_send_buffer
        row_length_recv_buffer = row_length_send_buffer

    # Step3
    if isinstance(params[0], DynamicVariable) and key_recv_buffer.dtype != params[0].key_type:
        key_recv_buffer = tf.cast(key_recv_buffer, params[0].key_type)
    emb_vec_buffer, _, _ = _lookup_forward(
        params, key_recv_buffer, row_length_recv_buffer, num_gpus=num_gpus(), **kwargs
    )

    # Step4
    if num_gpus() > 1:
        splits = []
        for i, emb_vec in enumerate(emb_vec_buffer):
            size = tf.expand_dims(tf.size(emb_vec), 0)
            splits.append(size)
            emb_vec_buffer[i] = tf.reshape(emb_vec, [tf.size(emb_vec)])
        splits = tf.concat(splits, 0)
        emb_vec_buffer = tf.concat(emb_vec_buffer, 0)
        emb_vec_buffer, rsplits = alltoall(emb_vec_buffer, splits)
        emb_vec_buffer = tf.split(emb_vec_buffer, rsplits)

    # Step5
    emb_vec, _ = _postprocessing_forward(
        emb_vec_buffer, row_lengths, Tindices=keys[0].dtype, **kwargs
    )

    if not is_list:
        emb_vec = emb_vec[0]
    return emb_vec
