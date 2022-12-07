#
# Copyright (c) 2021, NVIDIA CORPORATION.
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
def test_criteo_hugectr():
    notebook = os.path.join(dirname(TEST_PATH), "notebooks/hugectr_e2e_demo.ipynb")
    input_path = "/dir/to/criteo/day_0"  # hard code here may need to change
    with testbook(
        notebook,
        execute=False,
        timeout=3600,
    ) as nb:
        nb.inject(
            f"""
                import os
                os.environ['INPUT_DATA'] = "{input_path}"
            """
        )
        nb.execute_cell(list(range(0, len(nb.cells))))
