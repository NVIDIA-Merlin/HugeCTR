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

import numpy as np
from itertools import chain
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.resource_variable_ops import variable_shape
from tensorflow.python.ops.resource_variable_ops import variable_accessed

from sparse_operation_kit import raw_ops
from sparse_operation_kit import tf_version

from sparse_operation_kit.communication import rank
from sparse_operation_kit.communication import num_ranks
from sparse_operation_kit.communication import id_in_rank
from sparse_operation_kit.communication import num_gpus
from sparse_operation_kit.communication import alltoall
from sparse_operation_kit.communication import allreduce
from sparse_operation_kit.communication import allgather

from sparse_operation_kit.distributed_variable import DistributedVariable
from sparse_operation_kit.distributed_variable import LocalizedVariable

from sparse_operation_kit.dynamic_variable import DynamicVariable
from sparse_operation_kit.utils import SOK_IndexedSlices
import sys

if sys.version_info < (3, 12):
    from importlib import find_loader
else:
    from importlib.util import find_spec as find_loader

try:
    from tensorflow.python.ops import kv_variable_ops
except:
    pass

try:
    from tensorflow.python.ops import resource_variable_ops
except:
    pass


@tf.RegisterGradient("DummyVarSparseReadEvict")
def _DummyVarSparseReadEvictGrad(op, *top_grads):
    handle = op.inputs[0]
    indices = op.inputs[1]
    key_type = op.get_attr("key_type")
    dtype = op.get_attr("dtype")
    variable_shape = raw_ops.dummy_var_shape(handle, key_type=key_type, dtype=dtype)
    size = array_ops.expand_dims(array_ops.size(indices), 0)
    values_shape = array_ops.concat([size, variable_shape[1:]], 0)
    grad = array_ops.reshape(top_grads[0], values_shape)
    indices = array_ops.reshape(indices, size)

    grads = [SOK_IndexedSlices()(grad, indices, values_shape)]
    return grads + [None]


def sparse_read_and_evict(var, indices, name=None):
    # only used on hybrid backend
    if var.backend_type != "hybrid":
        raise TypeError("sparse_read_and_evict only use on hybrid backend")
    variable_accessed(var)
    return raw_ops.dummy_var_sparse_read_evict(var._dummy_handle, indices, dtype=var.handle_dtype)


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


def _preprocessing_forward(keys, row_lengths, sp_weight_value, *args, **kwargs):
    """
    This function should not be used by user directly.
    """
    if len(sp_weight_value) == 0:
        name = kwargs.pop("name") if "name" in kwargs else "PreprocessingForward"
        with ops.name_scope(name) as name:
            return raw_ops.preprocessing_forward(keys, row_lengths, *args, **kwargs)
    else:
        name = kwargs.pop("name") if "name" in kwargs else "PreprocessingForwardWithWeight"
        with ops.name_scope(name) as name:
            return raw_ops.preprocessing_forward_with_weight(
                keys, row_lengths, sp_weight_value, *args, **kwargs
            )


def _hotness_calculate(*args, **kwargs):
    """
    This function should not be used by user directly.
    """
    name = kwargs.pop("name") if "name" in kwargs else "HotnessCalculate"
    with ops.name_scope(name) as name:
        return raw_ops.hotness_calculate(*args, **kwargs)


def isDynamicVariable(variable):
    return isinstance(variable, DynamicVariable)


def isEmbeddingVariable(variable):
    return find_loader("tensorflow.python.ops.kv_variable_ops") and isinstance(
        variable, kv_variable_ops.EmbeddingVariable
    )


def isResourceVariable(variable):
    return (
        find_loader("tensorflow.python.ops.resource_variable_ops")
        and isinstance(variable, resource_variable_ops.ResourceVariable)
        and not isEmbeddingVariable(variable)
        and not isDynamicVariable(variable)
    )


def isVariable(variable):
    return (
        isinstance(variable, tf.Variable)
        and not isResourceVariable(variable)
        and not isEmbeddingVariable(variable)
        and not isDynamicVariable(variable)
    )


def allSupportedVariableCheckFunc():
    return [
        isDynamicVariable,
        isEmbeddingVariable,
        isResourceVariable,
        isVariable,
    ]


def _lookup_forward(params, *args, **kwargs):
    """
    This function should not be used by user directly.
    """
    name = kwargs.pop("name") if "name" in kwargs else "LookupForward"
    with ops.name_scope(name) as name:
        for param in params:
            # For tf.GradientTape
            variable_accessed(param)
        if isDynamicVariable(params[0]):
            handles = [param.handle for param in params]
            return raw_ops.lookup_forward_dynamic(handles, *args, **kwargs)
        elif isEmbeddingVariable(params[0]):
            handles = [param.handle for param in params]
            return raw_ops.lookup_forward_embedding_var_gpu(handles, *args, **kwargs)
        elif isResourceVariable(params[0]):
            handles = [param.handle for param in params]
            return raw_ops.lookup_forward(handles, *args, **kwargs)
        elif isVariable(params[0]):
            return raw_ops.lookup_forward_variable(params, *args, **kwargs)
        else:
            raise NotImplementedError(str(type(params[0])) + " is not supported in lookup")


@tf.RegisterGradient("LookupForward")
def _LookupBackward(op, *top_grads):
    attr_list = [
        "num_lookups",
        "combiners",
        "shard",
        "dimensions",
        "rank",
        "num_ranks",
        "id_in_local_rank",
        # "Toffsets",
        "use_sp_weight",
        "use_filter",
    ]
    kwargs = {}
    for attr in attr_list:
        kwargs[attr] = op.get_attr(attr)

    num_gpus = op.get_attr("num_gpus")
    num_lookups = op.get_attr("num_lookups")
    top_grads = top_grads[:num_gpus]
    other_data = op.outputs[num_gpus : num_gpus + 3]
    hotness = op.inputs[num_lookups + 2]
    indices, values = raw_ops.lookup_backward(top_grads, *other_data, hotness, **kwargs)
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


@tf.RegisterGradient("LookupForwardVariable")
def _LookupBackward(op, *top_grads):
    attr_list = [
        "num_lookups",
        "combiners",
        "shard",
        "dimensions",
        "rank",
        "num_ranks",
        "id_in_local_rank",
        # "Toffsets",
        "use_sp_weight",
        "use_filter",
    ]
    kwargs = {}
    for attr in attr_list:
        kwargs[attr] = op.get_attr(attr)

    num_gpus = op.get_attr("num_gpus")
    num_lookups = op.get_attr("num_lookups")
    top_grads = top_grads[:num_gpus]
    other_data = op.outputs[num_gpus : num_gpus + 3]
    hotness = op.inputs[num_lookups + 2]
    indices, values = raw_ops.lookup_backward(top_grads, *other_data, hotness, **kwargs)
    grads = []
    for i in range(len(indices)):
        handle = op.inputs[i]
        params_shape = handle.shape
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
        "shard",
        "dimensions",
        "rank",
        "num_ranks",
        "id_in_local_rank",
        # "Toffsets",
        "use_sp_weight",
        "use_filter",
    ]
    kwargs = {}
    for attr in attr_list:
        kwargs[attr] = op.get_attr(attr)

    num_gpus = op.get_attr("num_gpus")
    num_lookups = op.get_attr("num_lookups")
    top_grads = top_grads[:num_gpus]
    other_data = op.outputs[num_gpus : num_gpus + 3]
    hotness = op.inputs[num_lookups + 2]
    indices, values = raw_ops.lookup_backward(top_grads, *other_data, hotness, **kwargs)
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


@tf.RegisterGradient("LookupForwardEmbeddingVarGPU")
def _LookupBackwardEmbeddingVarGPU(op, *top_grads):
    from tensorflow.python.framework import tensor_shape

    attr_list = [
        "num_lookups",
        "combiners",
        "shard",
        "dimensions",
        "rank",
        "num_ranks",
        "id_in_local_rank",
        # "Toffsets",
        "use_sp_weight",
        "use_filter",
    ]
    kwargs = {}
    for attr in attr_list:
        kwargs[attr] = op.get_attr(attr)

    num_gpus = op.get_attr("num_gpus")
    num_lookups = op.get_attr("num_lookups")
    top_grads = top_grads[:num_gpus]
    other_data = op.outputs[num_gpus : num_gpus + 3]
    hotness = op.inputs[num_lookups + 2]
    indices, values = raw_ops.lookup_backward(top_grads, *other_data, hotness, **kwargs)
    grads = []
    for i in range(len(indices)):
        handle = op.inputs[i]
        params_shape = ops.convert_to_tensor(tensor_shape.TensorShape(handle.op.get_attr("shape")))
        size = array_ops.expand_dims(array_ops.size(indices[i]), 0)
        values_shape = array_ops.concat([size, params_shape[0:]], 0)
        values[i] = array_ops.reshape(values[i], values_shape)
        indices[i] = array_ops.reshape(indices[i], size)
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
        "shard",
        "dimensions",
        "rank",
        "num_ranks",
        "id_in_local_rank",
        "num_gpus",
        "Tindices",
        "use_sp_weight",
        "use_filter",
        # "Toffsets",
    ]
    kwargs = {}
    for attr in attr_list:
        kwargs[attr] = op.get_attr(attr)

    num_lookups = op.get_attr("num_lookups")
    num_gpus = op.get_attr("num_gpus")
    top_grads = top_grads[:num_lookups]
    row_lengths = op.inputs[num_gpus : num_gpus + num_lookups]
    hotness = op.inputs[num_gpus + num_lookups]
    sp_sum = op.inputs[num_gpus + num_lookups + 1]
    other_data = op.outputs[num_lookups:]
    grads = raw_ops.postprocessing_backward(
        top_grads, other_data, row_lengths, hotness, sp_sum, **kwargs
    )
    return grads + [None] * (len(op.inputs) - len(grads))


def to_list(any_obj):
    if not (isinstance(any_obj, list) or isinstance(any_obj, tuple)):
        return [any_obj]
    else:
        return any_obj


def lookup_sparse_impl(params, sp_ids, sp_weights=None, combiners=None, use_filter=False):
    shard, dimensions = [], []
    for param in params:
        shard.append(param.target_gpu)
        if find_loader("tensorflow.python.ops.kv_variable_ops") and isinstance(
            param, kv_variable_ops.EmbeddingVariable
        ):
            dimensions.append(param.shape[0])
        else:
            dimensions.append(param.shape[1])

    for i in range(1, len(params)):
        if type(params[i]) != type(params[0]):
            raise RuntimeError(
                "Distributed/Localized/Dynamic Variable cannot be used in the same lookup currently"
            )

    keys = []
    row_lengths = []
    sp_weight_value = []
    use_sp_weight = False if len(sp_weights) == 0 else True
    # first collect keys from Ragged tensors
    # keys means lookup key
    # every element in row_lengths meas number of lookup keys in a table for a sample
    for index in range(len(sp_ids)):
        sp_id = sp_ids[index]
        if isinstance(sp_id, tf.SparseTensor):
            sp_id = tf.RaggedTensor.from_sparse(sp_id)
        keys.append(sp_id.values)
        id_row_length = sp_id.row_lengths()
        row_lengths.append(id_row_length)
        if len(sp_weights) > 0:
            sp_weight = sp_weights[index]

            if isinstance(sp_weight, tf.SparseTensor):
                sp_weight = tf.RaggedTensor.from_sparse(sp_weight)
            weight_row_length = sp_weight.row_lengths()
            weight_row_length.shape.assert_is_compatible_with(id_row_length.shape)
            # is_same = weight_row_length.shape == id_row_length.shape
            # is_same = is_same and tf.math.reduce_all(id_row_length == weight_row_length).numpy()
            # if not is_same:
            #    raise RuntimeError("sp_id and sp_weight should be have same shape.")
            sp_weight_value.append(sp_weight.values)

    hotness_kwargs = {
        "num_lookups": len(params),
    }
    kwargs = {
        "combiners": combiners,
        "shard": shard,
        "dimensions": dimensions,
        "rank": rank(),
        "num_ranks": num_ranks(),
        "id_in_local_rank": id_in_rank(),
        "use_sp_weight": use_sp_weight,
        "use_filter": use_filter,
    }

    # Step1
    # copy all the key and row length in one buffer
    key_send_buffer, row_length_send_buffer, sp_weight_send_buffer = _preprocessing_forward(
        keys, row_lengths, sp_weight_value, num_gpus=num_gpus(), **kwargs
    )

    # Step2
    if num_gpus() > 1:
        key_recv_buffer = allgather(key_send_buffer)
        row_length_recv_buffer = allgather(row_length_send_buffer)
        if use_sp_weight:
            sp_weight_recv_buffer = allgather(sp_weight_send_buffer)
        else:
            sp_weight_recv_buffer = sp_weight_send_buffer
    else:
        key_recv_buffer = key_send_buffer
        row_length_recv_buffer = row_length_send_buffer
        sp_weight_recv_buffer = sp_weight_send_buffer

    hotness = _hotness_calculate(row_length_recv_buffer, num_gpus=num_gpus(), **hotness_kwargs)
    # Step3
    if isinstance(params[0], DynamicVariable) and key_recv_buffer.dtype != params[0].key_type:
        key_recv_buffer = tf.cast(key_recv_buffer, params[0].key_type)
    emb_vec_buffer, _, _, _ = _lookup_forward(
        params,
        key_recv_buffer,
        row_length_recv_buffer,
        hotness,
        sp_weight_recv_buffer,
        num_gpus=num_gpus(),
        **kwargs
    )

    sp_sum = []

    for i in range(len(sp_weight_value)):
        sp_sum.append(tf.reduce_sum(sp_weights[i], 1))
    if use_sp_weight:
        sp_sum = tf.concat(sp_sum, 0)

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
        emb_vec_buffer, row_lengths, hotness, sp_sum, Tindices=keys[0].dtype, **kwargs
    )

    return emb_vec


def lookup_sparse(params, sp_ids, sp_weights=None, combiners=None, use_low_frequency_filter=False):
    """
    Abbreviated as ``sok.lookup_sparse``.

    Perform fused sparse lookup on the given embedding ``params``. This function
    is similar to the ``tf.nn.embedding_lookup_sparse``, but with two differences:

        - It can do distributed lookup.
        - It can accept multiple params and multiple sp_ids to do fused lookup at once,
          which brings performance benefits.

    Parameters
    ----------
    params: list, tuple
            a list or tuple of trainable *sok.Variable*.
    sp_ids: list, tuple
            a list or tuple of tf.SparseTensor or tf.RaggedTensor.
    sp_weights: list tuple,optional
            a list or tuple of tf.SparseTensor or tf.RaggedTensor(float / double weights).
            if don't specify , indicate all weights should be taken to be 1.
    combiners: list, tuple,optional
            a list or tuple of string to specify the combiner of each lookup,for now only suupport "mean" "sum".
            if don't specify , indicate all elements(numbero of elements is same with number of sok.Variables) in combiners will should be set to be mean.
    use_low_frequency_filter: bool,optional
            For new indices that are not in the embedding table, should low-frequency filtering be performed to enter the embedding table

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
        import sparse_operation_kit as sok

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
            [v1, v2], [indices1, indices2], combiners=["sum", "sum"]
        )
        print(embeddings[0])
        print(embeddings[1])
    """
    # `is_list` determines whether to return a list or a tensor in the end
    #

    # check every gpu have lookup

    is_list = isinstance(sp_ids, list) or isinstance(sp_ids, tuple) or isinstance(sp_weights, tuple)

    params = to_list(params)
    sp_ids = to_list(sp_ids)
    num_tables = len(params)

    # check combiners
    if combiners == None:
        combiners_numpy = np.chararray(num_tables, itemsize=4)
        combiners_numpy[:] = "mean"
        combiners = combiners_numpy.tolist()
    else:
        combiners = to_list(combiners)
    if sp_weights == None:
        sp_weights = [] * len(params)
    else:
        if len(sp_ids) != len(sp_weights):
            raise RuntimeError("sp_ids length is not equal sp_weights")
        sp_weights = to_list(sp_weights)

    gpu_flag = [0] * num_gpus()
    all_gpu_allocate = False
    for param in params:
        tmp_target_gpu = param.target_gpu
        if tmp_target_gpu == -1:
            all_gpu_allocate = True
            break
        if isinstance(tmp_target_gpu, int):
            tmp_target_gpu = [tmp_target_gpu]
        for tmp_gpu_id in tmp_target_gpu:
            gpu_flag[tmp_gpu_id] += 1

    if all_gpu_allocate == False:
        for tmp_gpu_flag in gpu_flag:
            if tmp_gpu_flag == 0:
                raise Exception("every gpu must have table!")

    # group same type of variable
    assert len(params) == len(sp_ids)

    emb_vec = [None for _ in range(len(params))]
    variable_type_check_func = allSupportedVariableCheckFunc()
    for check_func in variable_type_check_func:
        selected_idx = [i for i in range(len(params)) if check_func(params[i])]

        if len(selected_idx) > 0:
            selected_params = [params[i] for i in selected_idx]
            selected_sp_ids = [sp_ids[i] for i in selected_idx]
            selected_sp_weights = (
                [] if len(sp_weights) == 0 else [sp_weights[i] for i in selected_idx]
            )
            selected_combiners = [combiners[i] for i in selected_idx]

            selected_emb_vec = lookup_sparse_impl(
                selected_params,
                selected_sp_ids,
                selected_sp_weights,
                selected_combiners,
                use_low_frequency_filter,
            )
            for ii, i in enumerate(selected_idx):
                emb_vec[i] = selected_emb_vec[ii]
    assert None not in emb_vec
    if not is_list:
        emb_vec = emb_vec[0]
    return emb_vec
