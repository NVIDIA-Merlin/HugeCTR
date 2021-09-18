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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sparse_operation_kit import kit_lib
from tensorflow.python.training.tracking import base as trackable

class EmbeddingLayerHandle(trackable.Trackable):
    """
    This is the base class used to track embedding layer handle.
    """
    def __init__(self):
        pass


class DenseEmbeddingLayerHandle(EmbeddingLayerHandle):
    """
    This is the handle for dense embedding layer,
    which means no reduction conducted intra slots.
    """
    def __init__(self):
        pass

class SparseEmbeddingLayerHandle(EmbeddingLayerHandle):
    """
    This is the handle for sparse embedding layer.
    which means reduction will be conducted intra slots.
    """
    def __init__(self):
        pass


if __name__ == "__main__":
    pass