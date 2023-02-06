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
    max_eval_batches=100,
    batchsize_eval=1000,
    batchsize=1000,
    lr=0.001,
    vvgpu=[[0]],
    metrics_spec={hugectr.MetricsType.AUC: 1.0},
    repeat_dataset=True,
    i64_input_key=False,
)
reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.Parquet,
    source=["./data/census_parquet/file_names.txt"],
    eval_source="./data/census_parquet/file_names_val.txt",
    check_type=hugectr.Check_t.Sum,
    num_samples=199523,
    eval_num_samples=99762,
    slot_size_array=[
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
    ],
)
optimizer = hugectr.CreateOptimizer(
    optimizer_type=hugectr.Optimizer_t.SGD, update_type=hugectr.Update_t.Local, atomic_update=True
)
model = hugectr.Model(solver, reader, optimizer)
model.add(
    hugectr.Input(
        label_dims=[1, 1],
        label_names=["50k_label", "married_label"],
        dense_dim=0,
        dense_name="dense",
        data_reader_sparse_param_array=[hugectr.DataReaderSparseParam("data", 1, True, 32)],
    )
)
model.add(
    hugectr.SparseEmbedding(
        embedding_type=hugectr.Embedding_t.LocalizedSlotSparseEmbeddingHash,
        workspace_size_per_gpu_in_mb=10000,
        embedding_vec_size=16,
        combiner="sum",
        sparse_embedding_name="embedding",
        bottom_name="data",
        optimizer=optimizer,
    )
)
# Shared layers before split to respective losses
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Reshape,
        bottom_names=["embedding"],
        top_names=["reshape_embedding"],
        leading_dim=512,
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["reshape_embedding"],
        top_names=["shared1"],
        num_output=128,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ReLU, bottom_names=["shared1"], top_names=["relu1"]
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Dropout,
        bottom_names=["relu1"],
        top_names=["dropout1"],
        dropout_rate=0.5,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["dropout1"],
        top_names=["shared2"],
        num_output=256,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ReLU, bottom_names=["shared2"], top_names=["relu2"]
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Dropout,
        bottom_names=["relu2"],
        top_names=["dropout2"],
        dropout_rate=0.5,
    )
)

# Split into separate branches for different loss layers
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Slice,
        bottom_names=["dropout2"],
        top_names=["sliceA", "sliceB"],
        ranges=[(0, 256), (0, 256)],
    )
)

# "A" side and corresponding loss
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["sliceA"],
        top_names=["A_fc1"],
        num_output=64,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ReLU, bottom_names=["A_fc1"], top_names=["A_relu1"]
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Dropout,
        bottom_names=["A_fc1"],
        top_names=["A_dropout1"],
        dropout_rate=0.5,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["A_dropout1"],
        top_names=["A_out"],
        num_output=1,
    )
)

# "B" side and corresponding loss
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["sliceB"],
        top_names=["B_fc1"],
        num_output=64,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ReLU, bottom_names=["B_fc1"], top_names=["B_relu1"]
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Dropout,
        bottom_names=["B_relu1"],
        top_names=["B_dropout1"],
        dropout_rate=0.5,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["B_dropout1"],
        top_names=["B_out"],
        num_output=1,
    )
)

# All loss layers must be declared last
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
        bottom_names=["A_out", "50k_label"],
        top_names=["lossA"],
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
        bottom_names=["B_out", "married_label"],
        top_names=["lossB"],
    )
)

model.compile(loss_names=["50k_label", "married_label"], loss_weights=[0.5, 0.5])
model.summary()
model.fit(max_iter=5000, display=1000, eval_interval=1000, snapshot=1000000, snapshot_prefix="mmoe")
