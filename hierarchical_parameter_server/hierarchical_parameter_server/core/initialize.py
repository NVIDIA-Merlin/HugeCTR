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
import tensorflow.distribute as tf_dist
from tensorflow import print as tf_print
from tensorflow import function
from tensorflow.python.framework import config
from tensorflow.dtypes import int32, int64
from tensorflow.python.ops import array_ops

MirroredStrategy = tf_dist.MirroredStrategy
try:
    MultiWorkerMirroredStrategy = tf_dist.MultiWorkerMirroredStrategy
except AttributeError:
    MultiWorkerMirroredStrategy = tf_dist.experimental.MultiWorkerMirroredStrategy
import sys


def Init(**kwargs):
    """
    Abbreviated as ``hps.Init(**kwargs)``.

    This function is used to do the initialization of HierarchicalParameterServer (HPS).

    HPS will leverage all available GPUs for current CPU process. Please set
    `CUDA_VISIBLE_DEVICES` or `tf.config.set_visible_devices` to specify which
    GPU(s) are used in this process before launching tensorflow runtime
    and calling this function. Besides, please ensure that deployed_device_list
    in the HPS configuration json file matches the visible devices.

    In **TensorFlow 2.x**, HPS can be used with **tf.distribute.Strategy** or **Horovod**.
    When it's used with tf.distribute.Strategy, it must be called under `strategy.scope()`.
    For example,

    .. code-block:: python

        with strategy.scope():
            hps.Init(**kwargs)

    When it's used with Horovod, it must be called at each process. For example,

    .. code-block:: python

        import horovod.tensorflow as hvd

        hvd.init()

        hps.Init(**kwargs)

    In **TensorFlow 1.15**, HPS can only work with **Horovod**. The retured status
    must be evaluated with `sess.run`, and it must be the first step before evaluate
    any other HPS APIs.

    .. code-block:: python

        hps_init = hps.Init(**kwargs)
        with tf.Session() as sess:
            sess.run(hps_init)
            ...

    Parameters
    ----------
    kwargs: dictionary
            keyword arguments for this function.
            Currently, it must contains `global_batch_size` and 'ps_config_file'.

    Returns
    -------
    status: string
            a string will be returned if this function executed successfully.
            And its contents will be 'OK'.
    """

    def _get_visible_devices():
        gpus = config.get_visible_devices("GPU")
        assert len(gpus) > 0
        visible_devices = []
        for i in range(len(gpus)):
            visible_devices.append(int(gpus[i].name.split(":")[-1]))
        return array_ops.constant(visible_devices, dtype=int32)

    def _single_worker_init(**kwargs):
        replica_ctx = tf_dist.get_replica_context()
        replica_ctx.merge_call(
            lambda strategy: tf_print("You are using the plugin with MirroredStrategy.")
        )
        global_id = replica_ctx.replica_id_in_sync_group
        visible_devices = _get_visible_devices()
        status = hps_lib.init(
            global_id,
            replica_ctx.num_replicas_in_sync,
            visible_devices,
            global_batch_size=kwargs["global_batch_size"],
            ps_config_file=kwargs["ps_config_file"],
        )
        return status

    def _multi_worker_init(**kwargs):
        replica_ctx = tf_dist.get_replica_context()
        global_id = replica_ctx.replica_id_in_sync_group
        visible_devices = _get_visible_devices()
        status = hps_lib.init(
            global_id,
            replica_ctx.num_replicas_in_sync,
            visible_devices,
            global_batch_size=kwargs["global_batch_size"],
            ps_config_file=kwargs["ps_config_file"],
        )
        return status

    def _horovod_init(**kwargs):
        local_rank = hvd.local_rank()
        visible_devices = _get_visible_devices()
        status = hps_lib.init(
            local_rank,
            hvd.size(),
            visible_devices,
            global_batch_size=kwargs["global_batch_size"],
            ps_config_file=kwargs["ps_config_file"],
        )
        return status

    def _one_device_init(**kwargs):
        local_rank = 0
        visible_devices = _get_visible_devices()
        status = hps_lib.init(
            local_rank,
            1,
            visible_devices,
            global_batch_size=kwargs["global_batch_size"],
            ps_config_file=kwargs["ps_config_file"],
        )
        return status

    if tf_dist.has_strategy():
        strategy = tf_dist.get_strategy()

        @function
        def _init_wrapper(run_fn, init_fn, **kwargs):
            return run_fn(init_fn, kwargs=kwargs)

        if isinstance(strategy, MirroredStrategy):
            _init_fn = _single_worker_init
        elif isinstance(strategy, MultiWorkerMirroredStrategy):
            _init_fn = _multi_worker_init
        else:
            raise RuntimeError("This strategy type is not supported yet.")

        if not hps_lib.in_tensorflow2():
            _run_fn = strategy.experimental_run_v2
        else:
            _run_fn = strategy.run

        _init_results = _init_wrapper(_run_fn, _init_fn, **kwargs)
        if hasattr(_init_results, "values"):
            _init_results = _init_results.values
        return _init_results

    elif "horovod.tensorflow" in sys.modules:
        import horovod.tensorflow as hvd

        if not hps_lib.in_tensorflow2():

            @function
            def _init_wrapper(**kwargs):
                return _horovod_init(**kwargs)

            return _init_wrapper(**kwargs)
        else:
            return _horovod_init(**kwargs)
    else:
        return _one_device_init(**kwargs)
