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

import os
from os.path import dirname, realpath

import pytest
from testbook import testbook

try:
    import hugectr
except ImportError:
    hugectr = None

try:
    import hugectr2onnx
except ImportError:
    hugectr2onnx = None

TEST_PATH = dirname(dirname(realpath(__file__)))


@pytest.mark.skipif(hugectr is None, reason="hugectr not installed")
def test_embedding_training_cache_example():
    notebook = os.path.join(dirname(TEST_PATH), "notebooks/embedding_training_cache_example.ipynb")
    with testbook(
        notebook,
        execute=False,
        timeout=3600,
    ) as nb:
        nb.execute_cell(list(range(0, len(nb.cells))))


@pytest.mark.skipif(hugectr is None, reason="hugectr not installed")
def test_multi_gpu_offline_inference():
    notebook = os.path.join(dirname(TEST_PATH), "notebooks/multi_gpu_offline_inference.ipynb")
    with testbook(
        notebook,
        execute=False,
        timeout=3600,
    ) as nb:
        nb.execute_cell(list(range(0, len(nb.cells))))


def test_prototype_indices():
    notebook = os.path.join(dirname(TEST_PATH), "notebooks/prototype_indices.ipynb")
    with testbook(
        notebook,
        execute=False,
        timeout=3600,
    ) as nb:
        nb.execute_cell(list(range(0, len(nb.cells))))


@pytest.mark.skipif(hugectr is None, reason="hugectr not installed")
def test_hugectr_wdl_prediction():
    notebook = os.path.join(dirname(TEST_PATH), "notebooks/hugectr_wdl_prediction.ipynb")
    with testbook(
        notebook,
        execute=False,
        timeout=3600,
    ) as nb:
        # Create data folder
        nb.execute_cell(list(range(0, 5)))
        nb.execute_cell(list(range(9, len(nb.cells))))


@pytest.mark.skipif(hugectr is None, reason="hugectr not installed")
@pytest.mark.skipif(hugectr2onnx is None, reason="hugectr2onnx not installed")
def test_hugectr2onnx_demo():
    notebook = os.path.join(dirname(TEST_PATH), "notebooks/hugectr2onnx_demo.ipynb")
    with testbook(
        notebook,
        execute=False,
        timeout=3600,
    ) as nb:
        nb.execute_cell(list(range(0, len(nb.cells))))


@pytest.mark.skipif(hugectr is None, reason="hugectr not installed")
def test_continuous_training():
    notebook = os.path.join(dirname(TEST_PATH), "notebooks/continuous_training.ipynb")
    with testbook(
        notebook,
        execute=False,
        timeout=3600,
    ) as nb:
        nb.execute_cell(list(range(0, len(nb.cells))))


@pytest.mark.skipif(hugectr is None, reason="hugectr not installed")
def test_movie_lens_example():
    notebook = os.path.join(dirname(TEST_PATH), "notebooks/movie-lens-example.ipynb")
    with testbook(
        notebook,
        execute=False,
        timeout=3600,
    ) as nb:
        nb.execute_cell(list(range(0, len(nb.cells))))
