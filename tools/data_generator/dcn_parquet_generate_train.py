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
from hugectr.tools import DataGenerator, DataGeneratorParams
from mpi4py import MPI

# Norm data generation
data_generator_params = DataGeneratorParams(
    format=hugectr.DataReaderType_t.Parquet,
    label_dim=1,
    dense_dim=13,
    num_slot=26,
    num_files=32,
    eval_num_files=16,
    i64_input_key=True,
    source="./dcn_parquet/file_list.txt",
    eval_source="./dcn_parquet/file_list_test.txt",
    slot_size_array=[
        39884,
        39043,
        17289,
        7420,
        20263,
        3,
        7120,
        1543,
        39884,
        39043,
        17289,
        7420,
        20263,
        3,
        7120,
        1543,
        63,
        63,
        39884,
        39043,
        17289,
        7420,
        20263,
        3,
        7120,
        1543,
    ],
    # for parquet, check_type doesn't make any difference
    check_type=hugectr.Check_t.Non,
    dist_type=hugectr.Distribution_t.PowerLaw,
    power_law_type=hugectr.PowerLaw_t.Short,
)
data_generator = DataGenerator(data_generator_params)
data_generator.generate()

# DCN train
solver = hugectr.CreateSolver(
    max_eval_batches=1280,
    batchsize_eval=1024,
    batchsize=1024,
    lr=0.001,
    vvgpu=[[0]],
    i64_input_key=True,
    repeat_dataset=True,
)
reader = hugectr.DataReaderParams(
    data_reader_type=data_generator_params.format,
    source=[data_generator_params.source],
    eval_source=data_generator_params.eval_source,
    # For parquet, generated dataset doesn't guarantee uniqueness, slot_size_array is still a must
    slot_size_array=data_generator_params.slot_size_array,
    check_type=data_generator_params.check_type,
)
optimizer = hugectr.CreateOptimizer(
    optimizer_type=hugectr.Optimizer_t.Adam, update_type=hugectr.Update_t.Global
)
model = hugectr.Model(solver, reader, optimizer)
model.add(
    hugectr.Input(
        label_dim=data_generator_params.label_dim,
        label_name="label",
        dense_dim=data_generator_params.dense_dim,
        dense_name="dense",
        data_reader_sparse_param_array=[
            hugectr.DataReaderSparseParam("data1", 1, True, data_generator_params.num_slot)
        ],
    )
)
model.add(
    hugectr.SparseEmbedding(
        embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
        workspace_size_per_gpu_in_mb=75,
        embedding_vec_size=16,
        combiner="sum",
        sparse_embedding_name="sparse_embedding1",
        bottom_name="data1",
        optimizer=optimizer,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Reshape,
        bottom_names=["sparse_embedding1"],
        top_names=["reshape1"],
        leading_dim=416,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Concat, bottom_names=["reshape1", "dense"], top_names=["concat1"]
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Slice,
        bottom_names=["concat1"],
        top_names=["slice11", "slice12"],
        ranges=[(0, 429), (0, 429)],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.MultiCross,
        bottom_names=["slice11"],
        top_names=["multicross1"],
        num_layers=6,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["slice12"],
        top_names=["fc1"],
        num_output=1024,
    )
)
model.add(
    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc1"], top_names=["relu1"])
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
        layer_type=hugectr.Layer_t.Concat,
        bottom_names=["dropout1", "multicross1"],
        top_names=["concat2"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["concat2"],
        top_names=["fc2"],
        num_output=1,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
        bottom_names=["fc2", "label"],
        top_names=["loss"],
    )
)
model.compile()
model.summary()
model.fit(max_iter=5120, display=200, eval_interval=1000)
