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

from hugectr.inference import InferenceParams, CreateInferenceSession
from hugectr.inference import InferenceModel, InferenceParams
import numpy as np

batch_size = 16384
num_batches = 1
data_source = "./mmoe_data/file_names_val.txt"
inference_params = InferenceParams(
    model_name="mmoe",
    max_batchsize=batch_size,
    hit_rate_threshold=1.0,
    dense_model_file="./onnx_converter/hugectr_models/mmoe_dense_2000.model",
    sparse_model_files=["./onnx_converter/hugectr_models/mmoe0_sparse_2000.model"],
    device_id=0,
    use_gpu_embedding_cache=False,
    cache_size_percentage=1.0,
    i64_input_key=False,
    use_mixed_precision=False,
    use_cuda_graph=True,
)
inference_model = InferenceModel("./onnx_converter/graph_files/mmoe.json", inference_params)
inf_predictions = inference_model.predict(
    num_batches=num_batches,
    source=data_source,
    data_reader_type=hugectr.DataReaderType_t.Parquet,
    check_type=hugectr.Check_t.Sum,
)
