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

import hugectr
from mpi4py import MPI

import hugectr2onnx
import onnxruntime as ort
from utils import read_samples_for_mmoe, compare_array_approx
import numpy as np

ground_truth = "/onnx_converter/hugectr_models/mmoe_parquet_preds.npy"
graph_config = "/onnx_converter/graph_files/mmoe.json"
dense_model = "/onnx_converter/hugectr_models/mmoe_dense_2000.model"
sparse_models = ["/onnx_converter/hugectr_models/mmoe0_sparse_2000.model"]
onnx_model_path = "/onnx_converter/onnx_models/mmoe.onnx"
data_file = "./val/0.parquet"
batch_size = 16384
num_batches = 1
data_source = "./file_names_val.txt"
slot_size_array = [
    91,
    73622,
    17,
    1425,
    3,
    24,
    15,
    5,
    10,
    2,
    3,
    6,
    8,
    133,
    114,
    1675,
    6,
    6,
    51,
    38,
    8,
    47,
    10,
    9,
    10,
    3,
    4,
    7,
    5,
    2,
    52,
    9,
]
slot_shift = np.insert(np.cumsum(slot_size_array), 0, 0)[:-1]

hugectr2onnx.converter.convert(onnx_model_path, graph_config, dense_model, True, sparse_models, "")
label, dense, keys = read_samples_for_mmoe(
    data_file, slot_shift, num_samples=batch_size * num_batches, slot_num=32
)
sess = ort.InferenceSession(onnx_model_path)
res = sess.run(
    output_names=[sess.get_outputs()[0].name, sess.get_outputs()[1].name],
    input_feed={sess.get_inputs()[0].name: dense, sess.get_inputs()[1].name: keys},
)
preds0 = res[0].reshape((batch_size * num_batches, 1))
preds1 = res[1].reshape((batch_size * num_batches, 1))
onnx_preds = np.concatenate((preds0, preds1), axis=1)
print("onnx_preds.shape: ", onnx_preds.shape)

predictions = np.load(ground_truth).reshape(batch_size * num_batches, 2)

print("predictions.shape: ", predictions.shape)

compare_array_approx(onnx_preds, predictions, "mmoe", 1e-2, 1e-1)
