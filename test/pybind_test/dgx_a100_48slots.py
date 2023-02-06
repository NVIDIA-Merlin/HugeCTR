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

# 1. Create Solver, DataReaderParams and Optimizer
solver = hugectr.CreateSolver(
    max_eval_batches=1,
    batchsize_eval=1024,
    batchsize=256,
    vvgpu=[[0, 1, 2, 3, 4, 5, 6, 7]],
    repeat_dataset=True,
    lr=24.0,
    warmup_steps=2750,
    decay_start=49315,
    decay_steps=27772,
    decay_power=2.0,
    end_lr=0.0,
    use_mixed_precision=True,
    scaler=1024,
    use_cuda_graph=True,
    async_mlp_wgrad=True,
    gen_loss_summary=False,
    train_intra_iteration_overlap=True,
    train_inter_iteration_overlap=True,
    eval_intra_iteration_overlap=True,  # doesn't do anything
    eval_inter_iteration_overlap=True,
    all_reduce_algo=hugectr.AllReduceAlgo.OneShot,
    grouped_all_reduce=False,
    num_iterations_statistics=20,
    metrics_spec={hugectr.MetricsType.AUC: 0.8025},
    perf_logging=True,
    drop_incomplete_batch=False,
)

reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.RawAsync,
    source=["./bing_proxy_raw_small/train.bin"],
    eval_source="./bing_proxy_raw_small/test.bin",
    check_type=hugectr.Check_t.Non,
    num_samples=20000000,
    eval_num_samples=1000000,
    cache_eval_data=1,
    slot_size_array=[
        30,
        1000,
        6000,
        6500,
        52000,
        200000,
        200000,
        240000,
        440000,
        10,
        5,
        2,
        1,
        100,
        1500000,
        200,
        70000,
        200000,
        110000,
        550000,
        120000,
        20000,
        125000,
        50000,
        50000,
        20,
        5,
        100000,
        20000,
        800000,
        60,
        400,
        120,
        4,
        1000,
        140000,
        5,
        50000,
        40000,
        5000,
        2000,
        7000,
        15000,
        1000,
        50,
        300,
        5000,
        30000,
    ],
    async_param=hugectr.AsyncParam(32, 4, 10, 2, 4096, True, hugectr.Alignment_t.Auto),
)
optimizer = hugectr.CreateOptimizer(
    optimizer_type=hugectr.Optimizer_t.SGD, update_type=hugectr.Update_t.Local, atomic_update=True
)
# 2. Initialize the Model instance
model = hugectr.Model(solver, reader, optimizer)
# 3. Construct the Model graph
model.add(
    hugectr.Input(
        label_dim=1,
        label_name="label",
        dense_dim=1,
        dense_name="dense",
        data_reader_sparse_param_array=[hugectr.DataReaderSparseParam("data1", 1, True, 48)],
    )
)
model.add(
    hugectr.SparseEmbedding(
        embedding_type=hugectr.Embedding_t.HybridSparseEmbedding,
        workspace_size_per_gpu_in_mb=15000,
        slot_size_array=[
            30,
            1000,
            6000,
            6500,
            52000,
            200000,
            200000,
            240000,
            440000,
            10,
            5,
            2,
            1,
            100,
            1500000,
            200,
            70000,
            200000,
            110000,
            550000,
            120000,
            20000,
            125000,
            50000,
            50000,
            20,
            5,
            100000,
            20000,
            800000,
            60,
            400,
            120,
            4,
            1000,
            140000,
            5,
            50000,
            40000,
            5000,
            2000,
            7000,
            15000,
            1000,
            50,
            300,
            5000,
            30000,
        ],
        embedding_vec_size=128,
        combiner="sum",
        sparse_embedding_name="sparse_embedding1",
        bottom_name="data1",
        optimizer=optimizer,
        hybrid_embedding_param=hugectr.HybridEmbeddingParam(
            2,
            -1,
            0.03,
            1.3e11,
            2.6e11,
            1.0,
            hugectr.CommunicationType.NVLink_SingleNode,
            hugectr.HybridEmbeddingType.Distributed,
        ),
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.FusedInnerProduct,
        pos_type=hugectr.FcPosition_t.Head,
        bottom_names=["dense"],
        top_names=["fc11", "fc12", "fc13", "fc14"],
        num_output=512,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.FusedInnerProduct,
        pos_type=hugectr.FcPosition_t.Body,
        bottom_names=["fc11", "fc12", "fc13", "fc14"],
        top_names=["fc21", "fc22", "fc23", "fc24"],
        num_output=256,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.FusedInnerProduct,
        pos_type=hugectr.FcPosition_t.Tail,
        bottom_names=["fc21", "fc22", "fc23", "fc24"],
        top_names=["fc3"],
        num_output=128,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Interaction,
        bottom_names=["fc3", "sparse_embedding1"],
        top_names=["interaction1", "interaction_grad"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.FusedInnerProduct,
        pos_type=hugectr.FcPosition_t.Head,
        bottom_names=["interaction1", "interaction_grad"],
        top_names=["fc41", "fc42", "fc43", "fc44"],
        num_output=1024,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.FusedInnerProduct,
        pos_type=hugectr.FcPosition_t.Body,
        bottom_names=["fc41", "fc42", "fc43", "fc44"],
        top_names=["fc51", "fc52", "fc53", "fc54"],
        num_output=1024,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.FusedInnerProduct,
        pos_type=hugectr.FcPosition_t.Body,
        bottom_names=["fc51", "fc52", "fc53", "fc54"],
        top_names=["fc61", "fc62", "fc63", "fc64"],
        num_output=512,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.FusedInnerProduct,
        pos_type=hugectr.FcPosition_t.Body,
        bottom_names=["fc61", "fc62", "fc63", "fc64"],
        top_names=["fc71", "fc72", "fc73", "fc74"],
        num_output=256,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.FusedInnerProduct,
        pos_type=hugectr.FcPosition_t.Tail,
        act_type=hugectr.Activation_t.Non,
        bottom_names=["fc71", "fc72", "fc73", "fc74"],
        top_names=["fc8"],
        num_output=1,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
        bottom_names=["fc8", "label"],
        top_names=["loss"],
    )
)
# 4. Dump the Model graph to JSON
model.graph_to_json(graph_config_file="dlrm.json")
# 5. Compile & Fit
model.compile()
model.summary()
model.fit(
    max_iter=75868, display=1000, eval_interval=3793, snapshot=2000000, snapshot_prefix="dlrm"
)
