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
    batchsize_eval=6400,
    batchsize=64,
    lr=0.00001,
    vvgpu=[[0]],
    repeat_dataset=True,
    i64_input_key=True,
    use_cuda_graph=True,
)
reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.Parquet,
    source=["./bst_data/train/_file_list.txt"],
    eval_source="./bst_data/valid/_file_list.txt",
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
        dense_dim=1,
        dense_name="dense",
        data_reader_sparse_param_array=[
            hugectr.DataReaderSparseParam("UserID", 1, True, 1),
            hugectr.DataReaderSparseParam("GoodID", 1, True, 10),
            hugectr.DataReaderSparseParam("Target_Good", 1, True, 1),
            hugectr.DataReaderSparseParam("CateID", 1, True, 10),
            hugectr.DataReaderSparseParam("Target_Cate", 1, True, 1),
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
        embedding_vec_size=16,
        combiner="sum",
        sparse_embedding_name="sparse_embedding_good",
        bottom_name="GoodID",
        optimizer=optimizer,
    )
)
model.add(
    hugectr.SparseEmbedding(
        embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
        workspace_size_per_gpu_in_mb=72,
        embedding_vec_size=16,
        combiner="sum",
        sparse_embedding_name="sparse_embedding_item_good",
        bottom_name="Target_Good",
        optimizer=optimizer,
    )
)
model.add(
    hugectr.SparseEmbedding(
        embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
        workspace_size_per_gpu_in_mb=30,
        embedding_vec_size=16,
        combiner="sum",
        sparse_embedding_name="sparse_embedding_cate",
        bottom_name="CateID",
        optimizer=optimizer,
    )
)
model.add(
    hugectr.SparseEmbedding(
        embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
        workspace_size_per_gpu_in_mb=30,
        embedding_vec_size=16,
        combiner="sum",
        sparse_embedding_name="sparse_embedding_item_cate",
        bottom_name="Target_Cate",
        optimizer=optimizer,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.SequenceMask,
        bottom_names=["dense", "dense"],
        top_names=["sequence_mask"],
        max_sequence_len_from=10,
        max_sequence_len_to=10,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Concat,
        bottom_names=["sparse_embedding_cate", "sparse_embedding_good"],
        top_names=["hist_emb_list"],
        axis=2,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["hist_emb_list"],
        top_names=["query_emb"],
        num_output=32,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["hist_emb_list"],
        top_names=["key_emb"],
        num_output=32,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["hist_emb_list"],
        top_names=["value_emb"],
        num_output=32,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.MultiHeadAttention,
        bottom_names=["query_emb", "key_emb", "value_emb", "sequence_mask"],
        top_names=["attention_out"],
        num_attention_heads=4,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Add,
        bottom_names=["attention_out", "query_emb"],
        top_names=["attention_add_shortcut"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.LayerNorm,
        bottom_names=["attention_add_shortcut"],
        top_names=["attention_layer_norm"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["attention_layer_norm"],
        top_names=["attention_ffn1"],
        num_output=128,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["attention_ffn1"],
        top_names=["attention_ffn2"],
        num_output=32,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Add,
        bottom_names=["attention_ffn2", "attention_layer_norm"],
        top_names=["attention_ffn_shortcut"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.LayerNorm,
        bottom_names=["attention_ffn_shortcut"],
        top_names=["attention_ffn_layer_norm"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ReduceMean,
        bottom_names=["attention_ffn_layer_norm"],
        top_names=["reduce_attention_ffn_layer_norm"],
        axis=1,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Reshape,
        bottom_names=["reduce_attention_ffn_layer_norm"],
        top_names=["reshape_attention_out"],
        leading_dim=32,
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
        layer_type=hugectr.Layer_t.Reshape,
        bottom_names=["sparse_embedding_item_good"],
        top_names=["reshape_item_good"],
        leading_dim=16,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Reshape,
        bottom_names=["sparse_embedding_item_cate"],
        top_names=["reshape_item_cate"],
        leading_dim=16,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Concat,
        bottom_names=[
            "reshape_attention_out",
            "reshape_user",
            "reshape_item_good",
            "reshape_item_cate",
        ],
        top_names=["dnn_input"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["dnn_input"],
        top_names=["fc_bst_i1"],
        num_output=256,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.PReLU_Dice,
        bottom_names=["fc_bst_i1"],
        top_names=["dice_1"],
        elu_alpha=0.2,
        eps=1e-8,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["dice_1"],
        top_names=["fc_bst_i2"],
        num_output=128,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.PReLU_Dice,
        bottom_names=["fc_bst_i2"],
        top_names=["dice_2"],
        elu_alpha=0.2,
        eps=1e-8,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["dice_2"],
        top_names=["fc_bst_i3"],
        num_output=64,
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.PReLU_Dice,
        bottom_names=["fc_bst_i3"],
        top_names=["dice_3"],
        elu_alpha=0.2,
        eps=1e-8,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["dice_3"],
        top_names=["fc_bst_i4"],
        num_output=1,
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
        bottom_names=["fc_bst_i4", "label"],
        top_names=["loss"],
    )
)
model.graph_to_json(graph_config_file="/onnx_converter/graph_files/bst_avg_pooling.json")
model.compile()
model.summary()
model.fit(
    max_iter=88000,
    display=1000,
    eval_interval=80000,
    snapshot=80000,
    snapshot_prefix="/onnx_converter/hugectr_models/bst_avg_pooling",
)

import numpy as np

preds = model.check_out_tensor("fc_bst_i4", hugectr.Tensor_t.Evaluate)
np.save("/onnx_converter/hugectr_models/bst_avg_pooling_preds.npy", preds)
