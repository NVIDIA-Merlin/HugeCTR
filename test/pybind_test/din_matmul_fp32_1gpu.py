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

solver = hugectr.CreateSolver(
    max_eval_batches=1,
    batchsize_eval=4096,
    batchsize=64,
    lr=0.00001,
    vvgpu=[[0]],
    repeat_dataset=True,
    i64_input_key=True,
    use_cuda_graph=True,
)
reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.Parquet,
    source=["./din_data/train/_file_list.txt"],
    eval_source="./din_data/valid/_file_list.txt",
    check_type=hugectr.Check_t.Non,
    num_workers=1,
    slot_size_array=[
        192403,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        63001,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        801,
    ],
)
optimizer = hugectr.CreateOptimizer(
    optimizer_type=hugectr.Optimizer_t.Adam,
    update_type=hugectr.Update_t.Global,
    beta1=0.9,
    beta2=0.999,
    epsilon=0.000000001,
)
model = hugectr.Model(solver, reader, optimizer)
model.add(
    hugectr.Input(
        label_dim=1,
        label_name="label",
        dense_dim=0,
        dense_name="dense",
        data_reader_sparse_param_array=[
            hugectr.DataReaderSparseParam("UserID", 1, True, 1),
            hugectr.DataReaderSparseParam("GoodID", 1, True, 11),
            hugectr.DataReaderSparseParam("CateID", 1, True, 11),
        ],
    )
)

model.add(
    hugectr.SparseEmbedding(
        embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
        workspace_size_per_gpu_in_mb=84,
        embedding_vec_size=18,
        combiner="sum",
        sparse_embedding_name="sparse_embedding_user",
        bottom_name="UserID",
        optimizer=optimizer,
    )
)
model.add(
    hugectr.SparseEmbedding(
        embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
        workspace_size_per_gpu_in_mb=72,
        embedding_vec_size=18,
        combiner="sum",
        sparse_embedding_name="sparse_embedding_good",
        bottom_name="GoodID",
        optimizer=optimizer,
    )
)
model.add(
    hugectr.SparseEmbedding(
        embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
        workspace_size_per_gpu_in_mb=30,
        embedding_vec_size=18,
        combiner="sum",
        sparse_embedding_name="sparse_embedding_cate",
        bottom_name="CateID",
        optimizer=optimizer,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.FusedReshapeConcat,
        bottom_names=["sparse_embedding_good", "sparse_embedding_cate"],
        top_names=["FusedReshapeConcat_item_his_em", "FusedReshapeConcat_item"],
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Scale,
        bottom_names=["FusedReshapeConcat_item"],
        top_names=["Scale_item"],
        axis=1,
        factor=10,
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Sub,
        bottom_names=["Scale_item", "FusedReshapeConcat_item_his_em"],
        top_names=["sub_ih"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ElementwiseMultiply,
        bottom_names=["Scale_item", "FusedReshapeConcat_item_his_em"],
        top_names=["ElementwiseMul_i"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Concat,
        bottom_names=["Scale_item", "FusedReshapeConcat_item_his_em", "sub_ih", "ElementwiseMul_i"],
        top_names=["concat_i_h"],
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["concat_i_h"],
        top_names=["fc_att_i2"],
        num_output=40,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["fc_att_i2"],
        top_names=["fc_att_i3"],
        num_output=1,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Reshape,
        bottom_names=["fc_att_i3"],
        top_names=["reshape_score"],
        leading_dim=10,
        time_step=1,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Softmax,
        bottom_names=["reshape_score"],
        top_names=["softmax_att_i"],
    )
)
# model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Scale,
#                            bottom_names = ["softmax_att_i"],
#                            top_names = ["Scale_i"],
#                            axis = 0, factor = 36))
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Reshape,
        bottom_names=["FusedReshapeConcat_item_his_em"],
        top_names=["reshape_item_his"],
        leading_dim=36,
        time_step=10,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.MatrixMultiply,  # matmul
        bottom_names=["softmax_att_i", "reshape_item_his"],
        top_names=["MatrixMultiply_ih"],
    )
)
# model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReduceSum,
#                            bottom_names = ["MatrixMultiply_ih"],
#                            top_names = ["reduce_ih"],
#                            axis = 1))
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Reshape,
        bottom_names=["MatrixMultiply_ih"],
        top_names=["reshape_reduce_ih"],
        leading_dim=36,
    )
)
# model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReduceSum,
#                            bottom_names = ["reshape_reduce_ih"],
#                            top_names = ["reduce_ih"],
#                            axis = 1))
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Reshape,
        bottom_names=["FusedReshapeConcat_item_his_em"],
        top_names=["reshape_his"],
        leading_dim=36,
        time_step=10,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ReduceMean,
        bottom_names=["reshape_his"],
        top_names=["reduce_item_his"],
        axis=1,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Reshape,
        bottom_names=["reduce_item_his"],
        top_names=["reshape_reduce_item_his"],
        leading_dim=36,
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Reshape,
        bottom_names=["sparse_embedding_user"],
        top_names=["reshape_user"],
        leading_dim=18,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Concat,
        bottom_names=[
            "reshape_user",
            "reshape_reduce_item_his",
            "reshape_reduce_ih",
            "FusedReshapeConcat_item",
        ],
        top_names=["concat_din_i"],
    )
)
# build_fcn_net
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["concat_din_i"],
        top_names=["fc_din_i1"],
        num_output=200,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.PReLU_Dice,
        bottom_names=["fc_din_i1"],
        top_names=["dice_1"],
        elu_alpha=0.2,
        eps=1e-8,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["dice_1"],
        top_names=["fc_din_i2"],
        num_output=80,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.PReLU_Dice,
        bottom_names=["fc_din_i2"],
        top_names=["dice_2"],
        elu_alpha=0.2,
        eps=1e-8,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["dice_2"],
        top_names=["fc3"],
        num_output=1,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
        bottom_names=["fc3", "label"],
        top_names=["loss"],
    )
)
model.compile()
model.summary()
model.fit(max_iter=88000, display=1000, eval_interval=1000, snapshot=1000000, snapshot_prefix="din")

model.eval()
metrics = model.get_eval_metrics()
print("[HUGECTR][INFO] iter: {}, metrics: {}".format(iter, metrics[0][1]))
if metrics[0][1] < 0.75:
    raise RuntimeError("Cannot reach the AUC threshold {}".format(0.75))
    sys.exit(1)
else:
    print("Successfully reach the AUC threshold {}".format(metrics[0][1]))
