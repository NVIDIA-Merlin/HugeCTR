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

import hugectr
from hugectr.inference import InferenceParams, CreateInferenceSession
import hugectr2onnx
import onnxruntime as ort
from utils import read_samples_for_dcn, compare_array_approx
import numpy as np

def hugectr2onnx_dcn_test(batch_size, num_batches, data_source, data_file, graph_config, dense_model, sparse_models, onnx_model_path, model_name):
    hugectr2onnx.converter.convert(onnx_model_path, graph_config, dense_model, True, sparse_models)
    label, dense, keys = read_samples_for_dcn(data_file, batch_size*num_batches, slot_num = 26)
    sess = ort.InferenceSession(onnx_model_path)
    res = sess.run(output_names=[sess.get_outputs()[0].name],
                  input_feed={sess.get_inputs()[0].name: dense, sess.get_inputs()[1].name: keys})
    res = res[0].reshape(batch_size*num_batches,)

    inference_params = InferenceParams(model_name = model_name,
                                    max_batchsize = batch_size,
                                    hit_rate_threshold = 1,
                                    dense_model_file = dense_model,
                                    sparse_model_files = sparse_models,
                                    device_id = 0,
                                    use_gpu_embedding_cache = True,
                                    cache_size_percentage = 0.6,
                                    i64_input_key = False)
    inference_session = CreateInferenceSession(graph_config, inference_params)
    predictions = inference_session.predict(num_batches, data_source, hugectr.DataReaderType_t.Norm, hugectr.Check_t.Sum)
    compare_array_approx(res, predictions, model_name, 1e-3, 1e-2)

if __name__ == "__main__":
    hugectr2onnx_dcn_test(64, 100, "./dcn_data/file_list_test.txt",
                    "./dcn_data/val/sparse_embedding0.data",
                    "/onnx_converter/graph_files/dcn.json",
                    "/onnx_converter/hugectr_models/dcn_dense_2000.model",
                    ["/onnx_converter/hugectr_models/dcn0_sparse_2000.model"],
                    "/onnx_converter/onnx_models/dcn.onnx",
                    "dcn")
    hugectr2onnx_dcn_test(64, 100, "./dcn_data/file_list_test.txt",
                    "./dcn_data/val/sparse_embedding0.data",
                    "/onnx_converter/graph_files/deepfm.json",
                    "/onnx_converter/hugectr_models/deepfm_dense_2000.model",
                    ["/onnx_converter/hugectr_models/deepfm0_sparse_2000.model"],
                    "/onnx_converter/onnx_models/deepfm.onnx",
                    "deepfm")
    hugectr2onnx_dcn_test(64, 100, "./dcn_data/file_list_test.txt",
                    "./dcn_data/val/sparse_embedding0.data",
                    "/onnx_converter/graph_files/dlrm.json",
                    "/onnx_converter/hugectr_models/dlrm_dense_2000.model",
                    ["/onnx_converter/hugectr_models/dlrm0_sparse_2000.model"],
                    "/onnx_converter/onnx_models/dlrm.onnx",
                    "dlrm")