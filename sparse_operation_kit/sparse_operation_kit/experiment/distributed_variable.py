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
import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

from sparse_operation_kit.experiment.communication import global_gpu_id, num_gpus


def Variable(*args, **kwargs):
    """
    Abbreviated as ``sok.experiment.Variable``.

    This is a helper function to generate model-parallel variable. There
    are two use cases:

    Distributed Variable:

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
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")  # nopep8

        sok.init()

        # If there are 2 GPUs in total, the shape on GPU0 will be [2, 3] and the shape
        # on GPU1 will be [2, 3]
        v = sok.Variable(np.arange(4 * 3).reshape(4, 3), dtype=tf.float32)

        # GPU0 output: [[0, 1, 2]
        #               [6, 7, 8]]
        # GPU1 output: [[3, 4,  5]
        #                9, 10, 11]
        print(v)

    Localized Variable:

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
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")  # nopep8

        sok.init()

        # If there are 2 GPUs in total, the shape on GPU0 will be [5, 3] and the shape
        # on GPU1 will be [0, 3]
        v = sok.Variable(
            np.arange(5 * 3).reshape(5, 3), dtype=tf.float32, mode="localized:0"
        )
        print(v.shape)

    As shown in the two examples above, when you need to store different parts of a variable on different
    GPUs (that is, allocating a model-parallel variable), this function can help you allocate the required
    memory on each GPU.

    Parameters
    ----------
    args:
        compatible with tf.Variable.

    kwargs:
        compatible with tf.Variable.

    mode: string
        a string to specify which model-parallel mode to use. Default value is "distributed", which stands
        for the Distributed Variable that mentioned above. Another option is "localized:#", which stands
        for Localized Variable, where # indicates which GPU you want to put this variable on. See the
        explanation above for specific examples.

    Returns
    -------
    variable: tf.Variable
            a tf.Variable that represents a part of the model-parallel variable.
    """
    mode = kwargs.pop("mode") if "mode" in kwargs else None

    if mode is None or mode == "distributed":
        kwargs["global_gpu_id"] = global_gpu_id()
        kwargs["num_gpus"] = num_gpus()
        return DistributedVariable(*args, **kwargs)

    elif mode[: len("localized")] == "localized":
        target_gpu = int(mode.split(":")[1])
        kwargs["target_gpu"] = target_gpu
        kwargs["global_gpu_id"] = global_gpu_id()
        return LocalizedVariable(*args, **kwargs)

    else:
        raise RuntimeError("Not supported mode: %s" % mode)


class DistributedVariable(ResourceVariable):
    def __init__(
        self,
        initial_value=None,
        trainable=None,
        collections=None,
        validate_shape=True,
        caching_device=None,
        name=None,
        dtype=None,
        variable_def=None,
        import_scope=None,
        constraint=None,
        distribute_strategy=None,
        synchronization=None,
        aggregation=None,
        shape=None,
        initializer=None,
        global_gpu_id=None,
        num_gpus=None,
    ):
        self._global_gpu_id = global_gpu_id
        self._num_gpus = num_gpus

        if initial_value is not None:
            if isinstance(initial_value, list):
                initial_value = np.array(initial_value)
            initial_value_shape_length = len(initial_value.shape)

            if initial_value_shape_length == 1 and initial_value.shape[0] > 0:
                self._global_shape = (1, initial_value.shapae[0])
            elif initial_value_shape_length == 2:
                self._global_shape = initial_value.shape
            else:
                raise RuntimeError(
                    "initial_value shape is {} please input a one-dimension(dimension shape must > 0)".format(
                        initial_value.shape
                    )
                )

            local_size = int(self._global_shape[0] // num_gpus)
            if global_gpu_id < self._global_shape[0] % num_gpus:
                local_size += 1

            if isinstance(initial_value, np.ndarray):
                device = "CPU"
            else:
                device = initial_value.device

            with tf.device(device):
                indices = tf.convert_to_tensor(np.arange(local_size), dtype=tf.int64)
                indices = indices * num_gpus + global_gpu_id
                initial_value = tf.nn.embedding_lookup(initial_value, indices)
                if dtype is not None:
                    initial_value = tf.cast(initial_value, dtype)
                else:
                    initial_value = tf.cast(initial_value, initial_value.dtype)
        else:
            self._global_shape = shape
            shape = None

            local_size = self._global_shape[0] // num_gpus
            if global_gpu_id < self._global_shape[0] % num_gpus:
                local_size += 1

            initial_value = initializer(shape=(local_size, self._global_shape[1]), dtype=dtype)

        super(DistributedVariable, self).__init__(
            initial_value=initial_value,
            trainable=trainable,
            collections=collections,
            validate_shape=validate_shape,
            caching_device=caching_device,
            name=name,
            dtype=dtype,
            variable_def=variable_def,
            import_scope=import_scope,
            constraint=constraint,
            distribute_strategy=distribute_strategy,
            synchronization=synchronization,
            aggregation=aggregation,
            shape=shape,
        )

    @property
    def global_shape(self):
        return self._global_shape

    @property
    def global_gpu_id(self):
        return self._global_gpu_id

    @property
    def target_gpu(self):
        return -1

    @property
    def num_gpus(self):
        return self._num_gpus

    def key_map(self, indices):
        return indices // self._num_gpus


class LocalizedVariable(ResourceVariable):
    def __init__(
        self,
        initial_value=None,
        trainable=None,
        collections=None,
        validate_shape=True,
        caching_device=None,
        name=None,
        dtype=None,
        variable_def=None,
        import_scope=None,
        constraint=None,
        distribute_strategy=None,
        synchronization=None,
        aggregation=None,
        shape=None,
        initializer=None,
        global_gpu_id=None,
        target_gpu=None,
    ):
        self._global_gpu_id = global_gpu_id
        self._target_gpu = target_gpu
        self._num_gpus = num_gpus()
        if target_gpu >= self._num_gpus:
            raise RuntimeError(
                "There are only %d GPU(s), cannot put embedding table on %dth(zero-indexed) GPU."
                % (self._num_gpus, target_gpu)
            )

        if initial_value is not None:
            if isinstance(initial_value, list):
                initial_value = np.array(initial_value)

            initial_value_shape_length = len(initial_value.shape)
            if initial_value_shape_length == 1 and initial_value.shape[0] > 0:
                self._global_shape = (1, initial_value.shapae[0])
            elif initial_value_shape_length == 2:
                self._global_shape = initial_value.shape
            else:
                raise RuntimeError(
                    "initial_value shape is {} please input a one-dimension(dimension shape must > 0)".format(
                        initial_value.shape
                    )
                )
        else:
            self._global_shape = shape
            shape = None

        if target_gpu != global_gpu_id:
            empty_value = np.ndarray(shape=[0, self._global_shape[1]], dtype=np.float32)
            if dtype is not None:
                initial_value = tf.cast(empty_value, dtype=dtype)
            else:
                initial_value = tf.convert_to_tensor(empty_value)
        elif initial_value is None:
            initial_value = initializer(shape=self._global_shape, dtype=dtype)

        super(LocalizedVariable, self).__init__(
            initial_value=initial_value,
            trainable=trainable,
            collections=collections,
            validate_shape=validate_shape,
            caching_device=caching_device,
            name=name,
            dtype=dtype,
            variable_def=variable_def,
            import_scope=import_scope,
            constraint=constraint,
            distribute_strategy=distribute_strategy,
            synchronization=synchronization,
            aggregation=aggregation,
            shape=shape,
        )

    @property
    def global_shape(self):
        return self._global_shape

    @property
    def global_gpu_id(self):
        return self._global_gpu_id

    @property
    def target_gpu(self):
        return self._target_gpu

    @property
    def num_gpus(self):
        return self._num_gpus

    def key_map(self, indices):
        return indices
