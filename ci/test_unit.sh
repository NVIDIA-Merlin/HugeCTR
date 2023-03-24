#!/usr/bin/env bash
#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

set -e

container="$1"
devices="$2"
multigpu="$3"

## Test HugeCTR
### Training container
if [ "$container" == "merlin-hugectr" ]; then
    # layers_test && \ Running oom in blossom
    checker_test && \
    device_map_test && \
    loss_test && \
    optimizer_test && \
    regularizers_test
    if [ "$multigpu" == "1" ]; then
         data_reader_test && \
	 parser_test && \
	 auc_test
    fi
    # Deactivated until it is self-contained and it runs
    # inference_test
### TensorFlow Training container
elif [ "$container" == "merlin-tensorflow" ]; then
    pushd /hugectr/sparse_operation_kit/sparse_operation_kit/experiment/test/function_test && \
    bash run_function_test.sh && \
    popd
fi
