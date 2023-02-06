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

from hierarchical_parameter_server import hps_lib
import tensorflow.distribute as tf_dist
from tensorflow import print as tf_print
from tensorflow import function
from tensorflow.python.framework import config
from tensorflow.dtypes import int32, int64
from tensorflow.python.ops import array_ops
import tensorflow as tf

MirroredStrategy = tf_dist.MirroredStrategy
try:
    MultiWorkerMirroredStrategy = tf_dist.MultiWorkerMirroredStrategy
except AttributeError:
    MultiWorkerMirroredStrategy = tf_dist.experimental.MultiWorkerMirroredStrategy
import sys


def Init(**kwargs):
    """
    Abbreviated as ``hps.Init(**kwargs)``.

    This function initializes the HPS for all the deployed models. It can be used
    explicitly or implicitly. When used explicitly, you must call the function
    only once and you must call it before any other HPS APIs.
    When used implicitly, ``ps_config_file`` and ``global_batch_size``
    should be specified in the constructor of ``hps.SparseLookupLayer``
    and ``hps.LookupLayer``. When the layer is executed for the first time, it triggers
    the internal HPS initialization implicitly in a thread-safe call-once manner. The
    implicit initialization is especially useful for deploying the SavedModels that
    leverage the HPS layers for online inference.

    HPS leverages all available GPUs for the current CPU process. Set
    ``CUDA_VISIBLE_DEVICES`` or ``tf.config.set_visible_devices`` to specify which
    GPUs to use in this process before you launch the TensorFlow runtime
    and calling this function. Additionally, ensure that the ``deployed_device_list``
    parameter in the HPS configuration JSON file matches the visible devices.

    In **TensorFlow 2.x**, HPS can be used with ``tf.distribute.Strategy`` or Horovod.
    When it is used with ``tf.distribute.Strategy``, you must call it under ``strategy.scope()``
    as shown in the following code block.

    .. code-block:: python

        import hierarchical_parameter_server as hps

        with strategy.scope():
            hps.Init(**kwargs)

    To use the function with Horovod, call it one for each time you initialize a
    Horovod process such as the following code block shows.

    .. code-block:: python

        import hierarchical_parameter_server as hps
        import horovod.tensorflow as hvd

        hvd.init()

        hps.Init(**kwargs)

    In **TensorFlow 1.15**, HPS can only work with Horovod. The returned status
    must be evaluated with ``sess.run`` and it must be the first step before evaluating
    any other HPS APIs.

    .. code-block:: python

        import hierarchical_parameter_server as hps

        hps_init = hps.Init(**kwargs)
        with tf.Session() as sess:
            sess.run(hps_init)
            ...

    Parameters
    ----------
    kwargs: dict
            Keyword arguments for this function.
            The dictionary must contain ``global_batch_size`` and ``ps_config_file``.

            * `global_batch_size`: int, the global batch size for HPS that is deployed on multiple GPUs.

            * `ps_config_file`: str, the JSON configuration file for HPS initialization.

            An example ``ps_config_file`` is as follows and ``global_batch_size`` can be
            configured as 16384 correspondingly:

            .. code-block:: python

                ps_config_file = {
                    "supportlonglong" : True,
                    "models" :
                    [{
                        "model": "demo_model",
                        "sparse_files": ["demo_model_sparse.model"],
                        "num_of_worker_buffer_in_pool": 3,
                        "embedding_table_names":["sparse_embedding0"],
                        "embedding_vecsize_per_table": [16],
                        "maxnum_catfeature_query_per_table_per_sample": [10],
                        "default_value_for_each_table": [1.0],
                        "deployed_device_list": [0],
                        "max_batch_size": 16384,
                        "cache_refresh_percentage_per_iteration": 0.2,
                        "hit_rate_threshold": 1.0,
                        "gpucacheper": 1.0,
                        "gpucache": True
                    },
                    {
                        "model": "demo_model2",
                        "sparse_files": ["demo_model2_sparse_0.model", "demo_model2_sparse_1.model"],
                        "num_of_worker_buffer_in_pool": 3,
                        "embedding_table_names":["sparse_embedding0", "sparse_embedding1"],
                        "embedding_vecsize_per_table": [64, 32],
                        "maxnum_catfeature_query_per_table_per_sample": [3, 5],
                        "default_value_for_each_table": [1.0, 1.0],
                        "deployed_device_list": [0],
                        "max_batch_size": 16384,
                        "cache_refresh_percentage_per_iteration": 0.2,
                        "hit_rate_threshold": 1.0,
                        "gpucacheper": 1.0,
                        "gpucache": True},
                    ]
                }


    Returns
    -------
    status: str
            On success, the function returns string with the value ``OK``.
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

        if tf.distribute.get_replica_context() is None:
            _init_results = _init_wrapper(_run_fn, _init_fn, **kwargs)
        else:
            _init_results = _init_fn(**kwargs)
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
