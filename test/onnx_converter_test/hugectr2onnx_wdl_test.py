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
from utils import read_samples_for_wdl, compare_array_approx
import numpy as np


def hugectr2onnx_wdl_test(
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
    label, dense, wide_data, deep_data = read_samples_for_wdl(
        data_file, batch_size * num_batches, slot_num=28
    )
    sess = ort.InferenceSession(onnx_model_path)
    res = sess.run(
        output_names=[sess.get_outputs()[0].name],
        input_feed={
            sess.get_inputs()[0].name: dense,
            sess.get_inputs()[1].name: deep_data,
            sess.get_inputs()[2].name: wide_data,
        },
    )
    res = res[0].reshape(
        batch_size * num_batches,
    )

    predictions = np.load(ground_truth).reshape(batch_size * num_batches)
    compare_array_approx(res, predictions, model_name, 1e-3, 1e-2)


if __name__ == "__main__":
    hugectr2onnx_wdl_test(
        64,
        100,
        "./wdl_data_parquet/val/0.9598d8cc5a1e4f85ae31f8068cb47fbb.parquet",
        "/onnx_converter/graph_files/wdl.json",
        "/onnx_converter/hugectr_models/wdl_dense_2000.model",
        [
            "/onnx_converter/hugectr_models/wdl0_sparse_2000.model",
            "/onnx_converter/hugectr_models/wdl1_sparse_2000.model",
        ],
        "/onnx_converter/onnx_models/wdl.onnx",
        "wdl",
        "/onnx_converter/hugectr_models/wdl_preds.npy",
    )
