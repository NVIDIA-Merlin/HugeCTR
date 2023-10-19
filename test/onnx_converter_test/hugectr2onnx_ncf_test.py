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
import hugectr2onnx
import onnxruntime as ort
from utils import read_samples_for_ncf, compare_array_approx
import numpy as np


def hugectr2onnx_ncf_test(
    batch_size,
    num_batches,
    data_file,
    graph_config,
    dense_model,
    sparse_models,
    onnx_model_path,
    model_name,
    ground_truth,
):
    hugectr2onnx.converter.convert(onnx_model_path, graph_config, dense_model, True, sparse_models)
    label, keys = read_samples_for_ncf(data_file, batch_size * num_batches, slot_num=2)
    dense = np.zeros([batch_size * num_batches, 0])
    sess = ort.InferenceSession(onnx_model_path)
    res = sess.run(
        output_names=[sess.get_outputs()[0].name],
        input_feed={sess.get_inputs()[0].name: None, sess.get_inputs()[1].name: keys},
    )
    res = res[0].reshape(
        batch_size * num_batches,
    )

    predictions = np.load(ground_truth).reshape(batch_size * num_batches)
    compare_array_approx(res, predictions, model_name, 1e-3, 1e-2)


if __name__ == "__main__":
    hugectr2onnx_ncf_test(
        64,
        100,
        "./movie_len_parquet/val/part_0.parquet",
        "/onnx_converter/graph_files/ncf.json",
        "/onnx_converter/hugectr_models/ncf_dense_2000.model",
        ["/onnx_converter/hugectr_models/ncf0_sparse_2000.model"],
        "/onnx_converter/onnx_models/ncf.onnx",
        "ncf",
        "/onnx_converter/hugectr_models/ncf_preds.npy",
    )
    hugectr2onnx_ncf_test(
        64,
        100,
        "./movie_len_parquet/val/part_0.parquet",
        "/onnx_converter/graph_files/gmf.json",
        "/onnx_converter/hugectr_models/gmf_dense_2000.model",
        ["/onnx_converter/hugectr_models/gmf0_sparse_2000.model"],
        "/onnx_converter/onnx_models/gmf.onnx",
        "gmf",
        "/onnx_converter/hugectr_models/gmf_preds.npy",
    )
    hugectr2onnx_ncf_test(
        64,
        100,
        "./movie_len_parquet/val/part_0.parquet",
        "/onnx_converter/graph_files/neumf.json",
        "/onnx_converter/hugectr_models/neumf_dense_2000.model",
        ["/onnx_converter/hugectr_models/neumf0_sparse_2000.model"],
        "/onnx_converter/onnx_models/neumf.onnx",
        "neumf",
        "/onnx_converter/hugectr_models/neumf_preds.npy",
    )
