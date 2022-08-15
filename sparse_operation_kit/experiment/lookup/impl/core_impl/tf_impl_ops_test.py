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
# ==============================================================================
"""Unit tests of HugeCTR TensorFlow backend ops
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

_tf_impl_ops = tf.load_op_library('./libtf_impl_ops_test.so')

default_config = config_pb2.ConfigProto(
    allow_soft_placement=False,
    log_device_placement=True,
    gpu_options=config_pb2.GPUOptions(allow_growth=True),
)


@test_util.deprecated_graph_mode_only
class TFImplOpsTest(test.TestCase):

  def test_storage_impl_on_gpu(self):
    gpu_list = tf.config.list_physical_devices('GPU')
    with self.session(use_gpu=gpu_list, config=default_config) as sess:
      with self.captureWritesToStream(sys.stderr) as printed:
        initial_size = 1024 * 2048
        extend_size = 2048 * 2048
        on_gpu = True
        gpu_id = 1 if len(gpu_list) > 1 else 0
        self.evaluate(
            _tf_impl_ops.storage_impl_test(initial_size=initial_size,
                                           extend_size=extend_size,
                                           gpu_id=gpu_id))

      print("[printed contents] : ", printed.contents())
      self.assertTrue("allocated pointer=0 " not in printed.contents())
      self.assertTrue("total size={}".format(initial_size +
                                             extend_size) in printed.contents())
      self.assertTrue("gpu_id={}".format(gpu_id) in printed.contents())
      self.assertTrue(
          "allocator=GPU_{}_bfc".format(gpu_id) in printed.contents())

  def test_gpu_resource_impl(self):
    gpu_list = tf.config.list_physical_devices('GPU')
    with self.session(use_gpu=gpu_list, config=default_config) as sess:
      with self.captureWritesToStream(sys.stderr) as printed:
        self.evaluate(_tf_impl_ops.gpu_resource_impl_test())

      print("[printed contents] : ", printed.contents())
      self.assertTrue("Get default CUDA stream fail!" not in printed.contents())
      self.assertTrue(
          "The default stream is got successfully!" in printed.contents())

  
  def test_tf_backend_impl(self):
    gpu_list = tf.config.list_physical_devices('GPU')
    with self.session(use_gpu=gpu_list, config=default_config) as sess:
      with self.captureWritesToStream(sys.stderr) as printed:
        c = tf.constant([[1, 2]])
        self.evaluate(_tf_impl_ops.tf_backend_test_test_op(c))

      print("[printed contents] : ", printed.contents())
      self.assertTrue("TfBackendTestOpTest Fail!" not in printed.contents())


if __name__ == "__main__":
    tf.get_logger().setLevel("WARNING")
    test.main()
