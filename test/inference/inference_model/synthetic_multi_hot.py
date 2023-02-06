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

solver = hugectr.CreateSolver(
    model_name="multi_hot",
    max_eval_batches=1,
    batchsize_eval=16384,
    batchsize=16384,
    lr=0.001,
    vvgpu=[[0, 1, 2, 3]],
    i64_input_key=True,
    repeat_dataset=True,
    use_cuda_graph=True,
)
reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.Parquet,
    source=["./multi_hot_parquet/file_list.txt"],
    eval_source="./multi_hot_parquet/file_list_test.txt",
    check_type=hugectr.Check_t.Non,
    slot_size_array=[10000, 10000, 10000],
)
optimizer = hugectr.CreateOptimizer(optimizer_type=hugectr.Optimizer_t.Adam)
model = hugectr.Model(solver, reader, optimizer)
model.add(
    hugectr.Input(
        label_dim=2,
        label_name="label",
        dense_dim=2,
        dense_name="dense",
        data_reader_sparse_param_array=[
            hugectr.DataReaderSparseParam("data1", [2, 1], False, 2),
            hugectr.DataReaderSparseParam("data2", 3, False, 1),
        ],
    )
)
model.add(
    hugectr.SparseEmbedding(
        embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
        workspace_size_per_gpu_in_mb=100,
        embedding_vec_size=16,
        combiner="sum",
        sparse_embedding_name="sparse_embedding1",
        bottom_name="data1",
        optimizer=optimizer,
    )
)
model.add(
    hugectr.SparseEmbedding(
        embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
        workspace_size_per_gpu_in_mb=100,
        embedding_vec_size=16,
        combiner="sum",
        sparse_embedding_name="sparse_embedding2",
        bottom_name="data2",
        optimizer=optimizer,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Reshape,
        bottom_names=["sparse_embedding1"],
        top_names=["reshape1"],
        leading_dim=32,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Reshape,
        bottom_names=["sparse_embedding2"],
        top_names=["reshape2"],
        leading_dim=16,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Concat,
        bottom_names=["reshape1", "reshape2", "dense"],
        top_names=["concat1"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["concat1"],
        top_names=["fc1"],
        num_output=1024,
    )
)
model.add(
    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc1"], top_names=["relu1"])
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["relu1"],
        top_names=["fc2"],
        num_output=2,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.MultiCrossEntropyLoss,
        bottom_names=["fc2", "label"],
        top_names=["loss"],
        target_weight_vec=[0.5, 0.5],
    )
)
model.compile()
model.summary()
model.graph_to_json("/dump_infer/multi_hot.json")
model.fit(
    max_iter=1100,
    display=200,
    eval_interval=1000,
    snapshot=1000,
    snapshot_prefix="/dump_infer/multi_hot",
)
model.export_predictions(
    "/dump_infer/multi_hot_pred_" + str(1000), "/dump_infer/multi_hot_label_" + str(1000)
)

sparse_embedding1 = model.check_out_tensor("sparse_embedding1", hugectr.Tensor_t.Evaluate)
sparse_embedding2 = model.check_out_tensor("sparse_embedding2", hugectr.Tensor_t.Evaluate)

import hugectr
from hugectr.inference import InferenceModel, InferenceParams
import numpy as np
from mpi4py import MPI

model_config = "/dump_infer/multi_hot.json"
inference_params = InferenceParams(
    model_name="multi_hot",
    max_batchsize=16384,
    hit_rate_threshold=1.0,
    dense_model_file="/dump_infer/multi_hot_dense_1000.model",
    sparse_model_files=[
        "/dump_infer/multi_hot0_sparse_1000.model",
        "/dump_infer/multi_hot1_sparse_1000.model",
    ],
    deployed_devices=[0, 1, 2, 3],
    use_gpu_embedding_cache=True,
    cache_size_percentage=0.5,
    i64_input_key=True,
)
inference_model = InferenceModel(model_config, inference_params)
pred = inference_model.predict(
    1,
    "./multi_hot_parquet/file_list_test.txt",
    hugectr.DataReaderType_t.Parquet,
    hugectr.Check_t.Non,
    [10000, 10000, 10000],
)
grount_truth = np.loadtxt("/dump_infer/multi_hot_pred_1000")
print("pred: ", pred)
print("grount_truth: ", grount_truth)
diff = pred.flatten() - grount_truth
mse = np.mean(diff * diff)
print("mse: ", mse)

inference_sparse_embedding1 = inference_model.check_out_tensor("sparse_embedding1")
inference_sparse_embedding2 = inference_model.check_out_tensor("sparse_embedding2")
diff1 = sparse_embedding1.flatten() - inference_sparse_embedding1.flatten()
diff2 = sparse_embedding2.flatten() - inference_sparse_embedding2.flatten()
mse1 = np.mean(diff1 * diff1)
mse2 = np.mean(diff2 * diff2)

if mse > 1e-3 or mse1 > 1e-3 or mse2 > 1e-3:
    raise RuntimeError(
        "Too large mse between synthetic multi hot inference and training: {}, {}, {}".format(
            mse, mse1, mse2
        )
    )
    sys.exit(1)
else:
    print(
        "Synthetic multi hot inference results are consistent with those during training, mse: {}, {}, {}".format(
            mse, mse1, mse2
        )
    )
