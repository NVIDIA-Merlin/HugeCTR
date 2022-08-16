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

from hugectr import Layer_t
from hugectr2onnx.hugectr_loader import ONNX_LAYER_TYPES, EXEMPTION_LAYER_TYPES


def hugectr2onnx_layer_type_test():
    for layer_type in Layer_t.__members__.keys():
        if layer_type not in ONNX_LAYER_TYPES and layer_type not in EXEMPTION_LAYER_TYPES:
            raise RuntimeError(
                "{} layer is not implemented in the ONNX converter".format(layer_type)
            )
            sys.exit(1)
    print("All HugeCTR layers have been implemented in the ONNX converter or exempted")
    print("Exempted layers: {}".format(EXEMPTION_LAYER_TYPES))


if __name__ == "__main__":
    hugectr2onnx_layer_type_test()
