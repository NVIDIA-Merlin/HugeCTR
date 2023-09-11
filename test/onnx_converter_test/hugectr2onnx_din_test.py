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
from utils import read_samples_for_din, compare_array_approx
import numpy as np


def hugectr2onnx_din_test(
    batch_size,
    num_batches,
    data_source,
    data_file,
    graph_config,
    dense_model,
    sparse_models,
    onnx_model_path,
    model_name,
    ground_truth,
):
    hugectr2onnx.converter.convert(onnx_model_path, graph_config, dense_model, True, sparse_models)
    dense, user, good, cate = read_samples_for_din(data_file, batch_size * num_batches, slot_num=23)
    sess = ort.InferenceSession(onnx_model_path)
    res = sess.run(
        output_names=[sess.get_outputs()[0].name],
        input_feed={
            sess.get_inputs()[0].name: dense,
            sess.get_inputs()[1].name: user,
            sess.get_inputs()[2].name: good,
            sess.get_inputs()[3].name: cate,
        },
    )
    res = res[0].reshape(
        batch_size * num_batches,
    )

    predictions = np.load(ground_truth).reshape(batch_size * num_batches)

    compare_array_approx(res, predictions, model_name, 1e-2, 1e-1)


if __name__ == "__main__":
    hugectr2onnx_din_test(
        64,
        100,
        "./din_data/valid/_file_list.txt",
        "./din_data/valid/0.ade7fdccb3fe4af0b49d5c8bac1ef534.parquet",
        "/onnx_converter/graph_files/din.json",
        "/onnx_converter/hugectr_models/din_dense_8000.model",
        [
            "/onnx_converter/hugectr_models/din0_sparse_8000.model",
            "/onnx_converter/hugectr_models/din1_sparse_8000.model",
            "/onnx_converter/hugectr_models/din2_sparse_8000.model",
        ],
        "/onnx_converter/onnx_models/din.onnx",
        "din",
        "/onnx_converter/hugectr_models/din_preds.npy",
    )
    hugectr2onnx_din_test(
        64,
        100,
        "./din_data/valid/_file_list.txt",
        "./din_data/valid/0.ade7fdccb3fe4af0b49d5c8bac1ef534.parquet",
        "/onnx_converter/graph_files/din_try.json",
        "/onnx_converter/hugectr_models/din_try_dense_80000.model",
        [
            "/onnx_converter/hugectr_models/din_try0_sparse_80000.model",
            "/onnx_converter/hugectr_models/din_try1_sparse_80000.model",
            "/onnx_converter/hugectr_models/din_try2_sparse_80000.model",
        ],
        "/onnx_converter/onnx_models/din_try.onnx",
        "din_try",
        "/onnx_converter/hugectr_models/din_try_preds.npy",
    )
