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

TEST_PATH = dirname(dirname(realpath(__file__)))


@pytest.mark.skipif(hugectr is None, reason="hugectr not installed")
def test_hps_demo():
    notebook = os.path.join(dirname(TEST_PATH), "notebooks/hps_demo.ipynb")
    with testbook(
        notebook,
        execute=False,
        timeout=3600,
    ) as nb:
        nb.execute_cell(list(range(0, 31)))
        nb.execute_cell(list(range(32, 47)))
        nb.execute_cell(list(range(48, len(nb.cells))))
