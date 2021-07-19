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

import tensorflow as tf
from sparse_operation_kit.kit_lib import dump_to_file, restore_from_file, load_embedding_values


# TODO: make it inherit from trackable???
class Saver(object):
    def __init__(self):
        # TODO: how to get all emb_var from Model???
        pass

    def __call__(self):
        pass

    def dump_to_file(self, embedding_variable, filepath):
        """
        This function is used to save one emb_var to file.
        """
        return dump_to_file(embedding_variable.values[0].emb_handle, filepath)

    def restore_from_file(self, embedding_variable, filepath):
        """
        This function is used to restore one emb_var from file.
        """
        return restore_from_file(embedding_variable.values[0].emb_handle, filepath)

    def load_embedding_values(self, embedding_variable, tensors):
        """
        This function is used to load tensors to variable GPU memory.
        The input tensors is a list of tensor, where all tensor
        make up to a whole tensor. For example:
        tensors = ([vocabulary_size_0, embedding_vec_size], [vocabulary_size_1, embedding_vec_size])
        It is equal to a big tensor whose shape is [vocabulary_size_0 + vocabulary_size_1, embedding_vec_size].
        """
        return load_embedding_values(embedding_variable.values[0].emb_handle, tensors)

    