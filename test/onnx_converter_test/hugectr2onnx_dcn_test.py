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
from utils import read_samples_for_dcn, compare_array_approx
import numpy as np


def hugectr2onnx_dcn_test(
    batch_size,
    num_batches,
    data_file,
    graph_config,
    dense_model,
    sparse_models,
    onnx_model_path,
    model_name,
    ground_truth,
    ntp_file="",
):
    hugectr2onnx.converter.convert(
        onnx_model_path, graph_config, dense_model, True, sparse_models, ntp_file
    )
    label, dense, keys = read_samples_for_dcn(data_file, batch_size * num_batches, slot_num=26)
    sess = ort.InferenceSession(onnx_model_path)
    res = sess.run(
        output_names=[sess.get_outputs()[0].name],
        input_feed={sess.get_inputs()[0].name: dense, sess.get_inputs()[1].name: keys},
    )
    res = res[0].reshape(
        batch_size * num_batches,
    )

    predictions = np.load(ground_truth).reshape(batch_size * num_batches)
    compare_array_approx(res, predictions, model_name, 1e-3, 1e-2)


if __name__ == "__main__":
    hugectr2onnx_dcn_test(
        64,
        100,
        "./deepfm_data_nvt/val/0.35ab81b16b4a409ba42a1baf89dcba52.parquet",
        "/onnx_converter/graph_files/dcn.json",
        "/onnx_converter/hugectr_models/dcn_dense_2000.model",
        ["/onnx_converter/hugectr_models/dcn0_sparse_2000.model"],
        "/onnx_converter/onnx_models/dcn.onnx",
        "dcn",
        "/onnx_converter/hugectr_models/dcn_preds.npy",
    )
    hugectr2onnx_dcn_test(
        64,
        100,
        "./deepfm_data_nvt/val/0.35ab81b16b4a409ba42a1baf89dcba52.parquet",
        "/onnx_converter/graph_files/deepfm.json",
        "/onnx_converter/hugectr_models/deepfm_dense_2000.model",
        ["/onnx_converter/hugectr_models/deepfm0_sparse_2000.model"],
        "/onnx_converter/onnx_models/deepfm.onnx",
        "deepfm",
        "/onnx_converter/hugectr_models/deepfm_preds.npy",
    )
    hugectr2onnx_dcn_test(
        64,
        100,
        "./deepfm_data_nvt/val/0.35ab81b16b4a409ba42a1baf89dcba52.parquet",
        "/onnx_converter/graph_files/dlrm.json",
        "/onnx_converter/hugectr_models/dlrm_dense_2000.model",
        ["/onnx_converter/hugectr_models/dlrm0_sparse_2000.model"],
        "/onnx_converter/onnx_models/dlrm.onnx",
        "dlrm",
        "/onnx_converter/hugectr_models/dlrm_preds.npy",
        "/onnx_converter/hugectr_models/dlrm_dense_2000.model.ntp.json",
    )
    hugectr2onnx_dcn_test(
        64,
        100,
        "./deepfm_data_nvt/val/0.35ab81b16b4a409ba42a1baf89dcba52.parquet",
        "/onnx_converter/graph_files/dlrm_mlp.json",
        "/onnx_converter/hugectr_models/dlrm_mlp_dense_2000.model",
        ["/onnx_converter/hugectr_models/dlrm_mlp0_sparse_2000.model"],
        "/onnx_converter/onnx_models/dlrm_mlp.onnx",
        "dlrm_mlp",
        "/onnx_converter/hugectr_models/dlrm_mlp_preds.npy",
    )
