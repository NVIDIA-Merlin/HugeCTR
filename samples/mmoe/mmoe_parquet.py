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

NUM_EXPERTS = 3
NUM_TASKS = 2

solver = hugectr.CreateSolver(
    max_eval_batches=100,
    batchsize_eval=762,
    batchsize=641,
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
        workspace_size_per_gpu_in_mb=1000,
        embedding_vec_size=16,
        combiner="sum",
        sparse_embedding_name="embedding",
        bottom_name="data",
        optimizer=optimizer,
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Reshape,
        bottom_names=["embedding"],
        top_names=["reshape_embedding"],
        leading_dim=512,
    )
)

# Slice into inputs for Gates and Experts
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Slice,
        bottom_names=["reshape_embedding"],
        top_names=["expert_emb", "gating_emb"],
        ranges=[(0, 512), (0, 512)],
    )
)

### Experts - Reshape and MLP ###
# Using 3 experts, so duplicate embedding for each expert
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Slice,
        bottom_names=["expert_emb"],
        top_names=["expert0_in", "expert1_in", "expert2_in"],
        ranges=[(0, 512), (0, 512), (0, 512)],
    )
)

# Expert 0
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["expert0_in"],
        top_names=["e0_fc1"],
        num_output=256,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ReLU, bottom_names=["e0_fc1"], top_names=["e0_relu1"]
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Dropout,
        bottom_names=["e0_relu1"],
        top_names=["e0_dropout1"],
        dropout_rate=0.5,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["e0_dropout1"],
        top_names=["e0_fc2"],
        num_output=128,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ReLU, bottom_names=["e0_fc2"], top_names=["e0_relu2"]
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Dropout,
        bottom_names=["e0_relu2"],
        top_names=["e0_dropout2"],
        dropout_rate=0.5,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Slice,
        bottom_names=["e0_dropout2"],
        top_names=["e0_out_A", "e0_out_B"],
        ranges=[(0, 128), (0, 128)],
    )
)

# Expert 1
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["expert1_in"],
        top_names=["e1_fc1"],
        num_output=256,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ReLU, bottom_names=["e1_fc1"], top_names=["e1_relu1"]
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Dropout,
        bottom_names=["e1_relu1"],
        top_names=["e1_dropout1"],
        dropout_rate=0.5,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["e1_dropout1"],
        top_names=["e1_fc2"],
        num_output=128,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ReLU, bottom_names=["e1_fc2"], top_names=["e1_relu2"]
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Dropout,
        bottom_names=["e1_relu2"],
        top_names=["e1_dropout2"],
        dropout_rate=0.5,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Slice,
        bottom_names=["e1_dropout2"],
        top_names=["e1_out_A", "e1_out_B"],
        ranges=[(0, 128), (0, 128)],
    )
)

# Expert 2
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["expert2_in"],
        top_names=["e2_fc1"],
        num_output=256,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ReLU, bottom_names=["e2_fc1"], top_names=["e2_relu1"]
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Dropout,
        bottom_names=["e2_relu1"],
        top_names=["e2_dropout1"],
        dropout_rate=0.5,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["e2_dropout1"],
        top_names=["e2_fc2"],
        num_output=128,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ReLU, bottom_names=["e2_fc2"], top_names=["e2_relu2"]
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Dropout,
        bottom_names=["e2_relu2"],
        top_names=["e2_dropout2"],
        dropout_rate=0.5,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Slice,
        bottom_names=["e2_dropout2"],
        top_names=["e2_out_A", "e2_out_B"],
        ranges=[(0, 128), (0, 128)],
    )
)


### Gating Network - gates use Softmax activation ###
# Using 2 output towers, each with its own gate. Duplicate input for each gating network
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Slice,
        bottom_names=["gating_emb"],
        top_names=["gateA_in", "gateB_in"],
        ranges=[(0, 512), (0, 512)],
    )
)

# Gating network A - gates use Softmax activation
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["gateA_in"],
        top_names=["gA_dense"],
        num_output=3,
    )
)  # one scaler for each expert
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Softmax, bottom_names=["gA_dense"], top_names=["gA_softmax"]
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Slice,
        bottom_names=["gA_softmax"],
        top_names=["gA_e0", "gA_e1", "gA_e2"],
        ranges=[(0, 1), (1, 2), (2, 3)],
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Scale,
        bottom_names=["gA_e0"],
        top_names=["gA_e0_scaled"],
        axis=0,
        factor=128,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ElementwiseMultiply,
        bottom_names=["e0_out_A", "gA_e0_scaled"],
        top_names=["e0_A_gated"],
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Scale,
        bottom_names=["gA_e1"],
        top_names=["gA_e1_scaled"],
        axis=0,
        factor=128,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ElementwiseMultiply,
        bottom_names=["e1_out_A", "gA_e1_scaled"],
        top_names=["e1_A_gated"],
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Scale,
        bottom_names=["gA_e2"],
        top_names=["gA_e2_scaled"],
        axis=0,
        factor=128,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ElementwiseMultiply,
        bottom_names=["e2_out_A", "gA_e2_scaled"],
        top_names=["e2_A_gated"],
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Add,
        bottom_names=["e0_A_gated", "e1_A_gated", "e2_A_gated"],
        top_names=["tower_A_input"],
    )
)

# Gating network A - gates use Softmax activation
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["gateB_in"],
        top_names=["gB_dense"],
        num_output=3,
    )
)  # one scaler for each expert
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Softmax, bottom_names=["gB_dense"], top_names=["gB_softmax"]
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Slice,
        bottom_names=["gB_softmax"],
        top_names=["gB_e0", "gB_e1", "gB_e2"],
        ranges=[(0, 1), (1, 2), (2, 3)],
    )
)

# Apply gate scalers to expert outputs for tower B
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Scale,
        bottom_names=["gB_e0"],
        top_names=["gB_e0_scaled"],
        axis=0,
        factor=128,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ElementwiseMultiply,
        bottom_names=["e0_out_B", "gB_e0_scaled"],
        top_names=["e0_B_gated"],
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Scale,
        bottom_names=["gB_e1"],
        top_names=["gB_e1_scaled"],
        axis=0,
        factor=128,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ElementwiseMultiply,
        bottom_names=["e1_out_B", "gB_e1_scaled"],
        top_names=["e1_B_gated"],
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Scale,
        bottom_names=["gB_e2"],
        top_names=["gB_e2_scaled"],
        axis=0,
        factor=128,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ElementwiseMultiply,
        bottom_names=["e2_out_B", "gB_e2_scaled"],
        top_names=["e2_B_gated"],
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Add,
        bottom_names=["e0_B_gated", "e1_B_gated", "e2_B_gated"],
        top_names=["tower_B_input"],
    )
)


# Tower A
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["tower_A_input"],
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
        bottom_names=["A_relu1"],
        top_names=["A_dropout1"],
        dropout_rate=0.5,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["A_dropout1"],
        top_names=["A_fc2"],
        num_output=1,
    )
)

# Tower B
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["tower_B_input"],
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
        top_names=["B_fc2"],
        num_output=1,
    )
)

# All loss layers must be declared last
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
        bottom_names=["A_fc2", "50k_label"],
        top_names=["lossA"],
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
        bottom_names=["B_fc2", "married_label"],
        top_names=["lossB"],
    )
)

model.compile(loss_names=["50k_label", "married_label"], loss_weights=[0.5, 0.5])
model.summary()
model.fit(
    max_iter=10000, display=1000, eval_interval=1000, snapshot=1000000, snapshot_prefix="mmoe"
)
# model.graph_to_json(graph_config_file = "mmoe.json") # Write model to json (optional)
