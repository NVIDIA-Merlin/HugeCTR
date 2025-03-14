"""
 Copyright (c) 2024, NVIDIA CORPORATION.
 
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
import argparse

from typing import Dict, TypeVar, List
from enum import Enum, unique


class ModelType(Enum):
    CRITEO = "CRITEO"  # no dense
    DCNV1 = "DCNV1"
    DCNV2 = "DCNV2"
    DEEPFM = "DEEPFM"
    DIN = "DIN"
    DLRM = "DLRM"
    WDL = "WDL"
    BST = "BST"
    MMOE = "MMOE"

    def __str__():
        return self.value


def parse_args(parser):
    DATA_READER_TYPE = {
        "Parquet": hugectr.DataReaderType_t.Parquet,
        "RawAsync": hugectr.DataReaderType_t.RawAsync,
    }
    OPTIMIZER_TYPE = {
        "Adam": hugectr.Optimizer_t.Adam,
        "MomentumSGD": hugectr.Optimizer_t.MomentumSGD,
        "Nesterov": hugectr.Optimizer_t.Nesterov,
        "SGD": hugectr.Optimizer_t.SGD,
    }
    UPDATE_TYPE = {
        "Global": hugectr.Update_t.Global,
        "LazyGlobal": hugectr.Update_t.LazyGlobal,
        "Local": hugectr.Update_t.Local,
    }
    parser.add_argument(
        "--model_type",
        choices=["CRITEO", "DCNV1", "DCNV2", "DEEPFM", "WDL", "BST", "DIN", "MMOE"],
        default="CRITEO",
        help="model type",
    )
    parser.add_argument("--single_slot", action="store_true", help="single slot or not")

    # solver
    # 0,1,2;0,1,2 => 2 nodes, 3gpu per node
    parser.add_argument("--max_eval_batches", type=int, default=1, help="max_eval_batches")
    parser.add_argument("--batchsize_eval", type=int, default=1024, help="batchsize_eval")
    parser.add_argument("--batchsize", type=int, default=1024, help="batchsize_train")
    parser.add_argument(
        "--vvgpu", type=str, required=True, help="vvgpu `:` as node sep `,` as gpu sep"
    )
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning_rate")
    parser.add_argument("--warmup_steps", type=int, default=0, help="learning_rate")
    parser.add_argument("--decay_start", type=int, default=0, help="decay_start")
    parser.add_argument("--decay_steps", type=int, default=1, help="decay_steps")
    parser.add_argument("--decay_power", type=float, default=2.0, help="decay_power")
    parser.add_argument("--end_lr", type=float, default=0.0, help="end_lr")
    parser.add_argument("--i64_input_key", action="store_true", help="i64_input_key")
    parser.add_argument("--use_mixed_precision", action="store_true", help="use_mixed_precision")
    parser.add_argument(
        "--scaler", type=float, default=1.0, help="scaler of mixed precision training"
    )

    # reader
    parser.add_argument(
        "--data_reader_type",
        choices=["Parquet", "RawAsync"],
        default="Parquet",
        help="DataReader type",
    )
    # , separated list 1.txt,2.txt,3.txt
    parser.add_argument("--source", type=str, default="./_file_list.txt", help="training sources")
    parser.add_argument("--eval_source", type=str, default="./_file_list.txt", help="eval sources")
    parser.add_argument("--num_samples", type=int, default=1024, help="num of training samples")
    parser.add_argument("--eval_num_samples", type=int, default=1024, help="num of eval samples")
    parser.add_argument(
        "--float_label_dense", action="store_true", help="dense feature in dataset is float or not"
    )
    # , separated list 1,2,3,4
    parser.add_argument("--slot_size_array", type=str, default="", help="slot size array")

    # optimizer
    parser.add_argument(
        "--optimizer_type",
        choices=["Adam", "MomentumSGD", "Nesterov", "SGD", "Adagrad"],
        default="SGD",
        help="optimizer type",
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for opt")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for opt")
    parser.add_argument("--epsilon", type=float, default=1e-07, help="epsilon for opt")
    parser.add_argument(
        "--update_type",
        choices=["Global", "LazyGlobal", "Local"],
        default="Local",
        help="update_type",
    )
    parser.add_argument("--momentum_factor", type=float, default=0.0, help="update_type")
    parser.add_argument("--atomic_update", action="store_true", help="atomic_update")

    # trainer param
    parser.add_argument("--max_iter", type=int, default=1024, help="num of training iterations")
    parser.add_argument("--display", type=int, default=200, help="num of display iterations")
    parser.add_argument("--eval_interval", type=int, default=200, help="num of eval interval")
    parser.add_argument("--snapshot", type=int, default=1000000, help="snapshot interval")
    parser.add_argument("--auc_threshold", type=float, default=0.5, help="auc threshold")

    args = parser.parse_args()

    gpus_per_node = args.vvgpu.split(":")
    vvgpu = []
    for gpus in gpus_per_node:
        gpus_str = gpus.split(",")
        gpus_int = [int(g) for g in gpus_str]
        vvgpu.append(gpus_int)
    args.vvgpu = vvgpu
    args.source = args.source.split(",")
    if len(args.slot_size_array):
        args.slot_size_array = [int(s) for s in args.slot_size_array.split(",")]
    assert not (args.single_slot ^ len(args.slot_size_array) == 1)

    args.model_type = ModelType[args.model_type]
    args.update_type = UPDATE_TYPE[args.update_type]
    args.optimizer_type = OPTIMIZER_TYPE[args.optimizer_type]
    args.data_reader_type = DATA_READER_TYPE[args.data_reader_type]
    return args


def create_solver(
    max_eval_batches: int,
    batchsize_eval: int,
    batchsize: int,
    vvgpu: List[List[int]],
    lr: float,
    warmup_steps: int,
    decay_start: int,
    decay_steps: int,
    decay_power: float,
    end_lr: float,
    i64_input_key: bool,
    use_mixed_precision: bool,
    scaler: float,
) -> hugectr.Solver:
    solver = hugectr.CreateSolver(
        max_eval_batches=max_eval_batches,
        batchsize_eval=batchsize_eval,
        batchsize=batchsize,
        vvgpu=vvgpu,
        lr=lr,
        warmup_steps=warmup_steps,
        decay_start=decay_start,
        decay_steps=decay_steps,
        decay_power=decay_power,
        end_lr=end_lr,
        i64_input_key=i64_input_key,
        use_mixed_precision=use_mixed_precision,
        scaler=scaler,
    )
    return solver


def create_data_reader(
    data_reader_type,
    source,
    eval_source,
    num_samples,
    eval_num_samples,
    float_label_dense,
    slot_size_array,
) -> hugectr.DataReaderParams:
    reader = hugectr.DataReaderParams(
        data_reader_type=data_reader_type,
        check_type=hugectr.Check_t.Non,
        source=source,
        eval_source=eval_source,
        num_samples=num_samples,
        eval_num_samples=eval_num_samples,
        float_label_dense=float_label_dense,
        slot_size_array=slot_size_array,
    )
    return reader


def create_optimizer(
    optimizer_type,
    beta1,
    beta2,
    epsilon,
    update_type,
    momentum_factor,
    atomic_update,
) -> hugectr.Optimizer_t:
    opt = hugectr.CreateOptimizer(
        optimizer_type=optimizer_type,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        update_type=update_type,
        momentum_factor=momentum_factor,
        atomic_update=atomic_update,
    )
    return opt


# there is no dense, and there num slots can be different
def criteo(
    model: hugectr.Model,
    optimizer: hugectr.Optimizer_t,
    single_slot: bool,
) -> hugectr.Model:
    # slot_size_array = [0] if single_slot else [132, 421, 1398, 1787, 53, 10, 3043, 78, 4, 2192, 2130, 1432, 1805, 25, 1854, 1559, 10, 1113, 515, 4, 1473, 9, 15, 1579, 40, 1193]
    model.add(
        hugectr.Input(
            label_dim=1,
            label_name="label",
            dense_dim=0,
            dense_name="dense",
            data_reader_sparse_param_array=[
                (
                    hugectr.DataReaderSparseParam("data1", 100, False, 1)
                    if single_slot
                    else hugectr.DataReaderSparseParam("data1", 4, False, 26)
                )
            ],
        )
    )
    model.add(
        hugectr.SparseEmbedding(
            embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
            workspace_size_per_gpu_in_mb=1000,
            embedding_vec_size=64,
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
            shape=[-1, 64] if single_slot else [-1, 1664],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["reshape1"],
            top_names=["fc1"],
            num_output=200,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc1"], top_names=["relu1"]
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["relu1"],
            top_names=["fc2"],
            num_output=200,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc2"], top_names=["relu2"]
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["relu2"],
            top_names=["fc3"],
            num_output=200,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc3"], top_names=["relu3"]
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["relu3"],
            top_names=["fc4"],
            num_output=1,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
            bottom_names=["fc4", "label"],
            top_names=["loss"],
        )
    )
    return model


def dcnv1(
    model: hugectr.Model,
    optimizer: hugectr.Optimizer_t,
) -> hugectr.Model:
    model.add(
        hugectr.Input(
            label_dim=1,
            label_name="label",
            dense_dim=13,
            dense_name="dense",
            data_reader_sparse_param_array=[hugectr.DataReaderSparseParam("data1", 1, False, 26)],
        )
    )
    model.add(
        hugectr.SparseEmbedding(
            embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
            workspace_size_per_gpu_in_mb=495,
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
            shape=[-1, 416],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Concat,
            bottom_names=["reshape1", "dense"],
            top_names=["concat1"],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.MultiCross,
            bottom_names=["concat1"],
            top_names=["multicross1"],
            num_layers=6,
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
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc1"], top_names=["relu1"]
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
            top_names=["fc2"],
            num_output=1024,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc2"], top_names=["relu2"]
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
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Concat,
            bottom_names=["dropout2", "multicross1"],
            top_names=["concat2"],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["concat2"],
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
    return model


def dcnv2(
    model: hugectr.Model,
    optimizer: hugectr.Optimizer_t,
) -> hugectr.Model:
    model.add(
        hugectr.Input(
            label_dim=1,
            label_name="label",
            dense_dim=13,
            dense_name="dense",
            data_reader_sparse_param_array=[hugectr.DataReaderSparseParam("data1", 1, False, 26)],
        )
    )
    model.add(
        hugectr.SparseEmbedding(
            embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
            workspace_size_per_gpu_in_mb=495,
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
            shape=[-1, 416],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Concat,
            bottom_names=["reshape1", "dense"],
            top_names=["concat1"],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.MultiCross,
            bottom_names=["concat1"],
            top_names=["multicross1"],
            num_layers=3,
            projection_dim=512,  # dcnv2
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
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc1"], top_names=["relu1"]
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
            top_names=["fc2"],
            num_output=1024,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc2"], top_names=["relu2"]
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
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Concat,
            bottom_names=["dropout2", "multicross1"],
            top_names=["concat2"],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["concat2"],
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
    return model


def wdl(
    model: hugectr.Model,
    optimizer: hugectr.Optimizer_t,
) -> hugectr.Model:
    model.add(
        hugectr.Input(
            label_dim=1,
            label_name="label",
            dense_dim=13,
            dense_name="dense",
            data_reader_sparse_param_array=[
                hugectr.DataReaderSparseParam("deep_data", 1, True, 26),
                hugectr.DataReaderSparseParam("wide_data", 1, True, 2),
            ],
        )
    )
    model.add(
        hugectr.SparseEmbedding(
            embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
            workspace_size_per_gpu_in_mb=27,
            embedding_vec_size=1,
            combiner="sum",
            sparse_embedding_name="sparse_embedding2",
            bottom_name="wide_data",
            optimizer=optimizer,
        )
    )
    model.add(
        hugectr.SparseEmbedding(
            embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
            workspace_size_per_gpu_in_mb=426,
            embedding_vec_size=16,
            combiner="sum",
            sparse_embedding_name="sparse_embedding1",
            bottom_name="deep_data",
            optimizer=optimizer,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Reshape,
            bottom_names=["sparse_embedding1"],
            top_names=["reshape1"],
            shape=[-1, 416],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Reshape,
            bottom_names=["sparse_embedding2"],
            top_names=["reshape_wide"],
            shape=[-1, 2],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReduceSum,
            bottom_names=["reshape_wide"],
            top_names=["reshape2"],
            axis=1,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Concat,
            bottom_names=["reshape1", "dense"],
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
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc1"], top_names=["relu1"]
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
            top_names=["fc2"],
            num_output=1024,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc2"], top_names=["relu2"]
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
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["dropout2"],
            top_names=["fc3"],
            num_output=1,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Add, bottom_names=["fc3", "reshape2"], top_names=["add1"]
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
            bottom_names=["add1", "label"],
            top_names=["loss"],
        )
    )
    return model


def deepfm(
    model: hugectr.Model,
    optimizer: hugectr.Optimizer_t,
) -> hugectr.Model:
    model.add(
        hugectr.Input(
            label_dim=1,
            label_name="label",
            dense_dim=13,
            dense_name="dense",
            data_reader_sparse_param_array=[hugectr.DataReaderSparseParam("data1", 1, True, 26)],
        )
    )
    model.add(
        hugectr.SparseEmbedding(
            embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
            workspace_size_per_gpu_in_mb=495,
            embedding_vec_size=11,
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
            shape=[-1, 11],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Slice,
            bottom_names=["reshape1"],
            top_names=["slice11", "slice12"],
            ranges=[(0, 10), (10, 11)],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Reshape,
            bottom_names=["slice11"],
            top_names=["reshape2"],
            shape=[-1, 260],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Reshape,
            bottom_names=["slice12"],
            top_names=["reshape3"],
            shape=[-1, 26],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Slice,
            bottom_names=["dense"],
            top_names=["slice21", "slice22"],
            ranges=[(0, 13), (0, 13)],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.WeightMultiply,
            bottom_names=["slice21"],
            top_names=["weight_multiply1"],
            weight_dims=[13, 10],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.WeightMultiply,
            bottom_names=["slice22"],
            top_names=["weight_multiply2"],
            weight_dims=[13, 1],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Concat,
            bottom_names=["reshape2", "weight_multiply1"],
            top_names=["concat1"],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Slice,
            bottom_names=["concat1"],
            top_names=["slice31", "slice32"],
            ranges=[(0, 390), (0, 390)],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["slice31"],
            top_names=["fc1"],
            num_output=400,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc1"], top_names=["relu1"]
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
            top_names=["fc2"],
            num_output=400,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc2"], top_names=["relu2"]
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
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["dropout2"],
            top_names=["fc3"],
            num_output=400,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc3"], top_names=["relu3"]
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Dropout,
            bottom_names=["relu3"],
            top_names=["dropout3"],
            dropout_rate=0.5,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["dropout3"],
            top_names=["fc4"],
            num_output=1,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.FmOrder2,
            bottom_names=["slice32"],
            top_names=["fmorder2"],
            out_dim=10,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReduceSum,
            bottom_names=["fmorder2"],
            top_names=["reducesum1"],
            axis=1,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Concat,
            bottom_names=["reshape3", "weight_multiply2"],
            top_names=["concat2"],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReduceSum,
            bottom_names=["concat2"],
            top_names=["reducesum2"],
            axis=1,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Add,
            bottom_names=["fc4", "reducesum1", "reducesum2"],
            top_names=["add"],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
            bottom_names=["add", "label"],
            top_names=["loss"],
        )
    )
    return model


def bst(
    model: hugectr.Model,
    optimizer: hugectr.Optimizer_t,
) -> hugectr.Model:
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
            shape=[-1, 32],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Reshape,
            bottom_names=["sparse_embedding_user"],
            top_names=["reshape_user"],
            shape=[-1, 18],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Reshape,
            bottom_names=["sparse_embedding_item_good"],
            top_names=["reshape_item_good"],
            shape=[-1, 16],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Reshape,
            bottom_names=["sparse_embedding_item_cate"],
            top_names=["reshape_item_cate"],
            shape=[-1, 16],
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
    return model


def construct_model(
    model_type: ModelType, model: hugectr.Model, optimizer: hugectr.Optimizer_t, single_slot: bool
) -> hugectr.Model:
    if model_type == ModelType.CRITEO:
        model = criteo(model, optimizer, single_slot)
    elif model_type == ModelType.DCNV1:
        model = dcnv1(model, optimizer)
    elif model_type == ModelType.DCNV2:
        model = dcnv2(model, optimizer)
    elif model_type == ModelType.WDL:
        model = wdl(model, optimizer)
    elif model_type == ModelType.DEEPFM:
        model = deepfm(model, optimizer)
    elif model_type == ModelType.BST:
        model = bst(model, optimizer)
    else:
        raise ValueError(f"unsupported model {model_type}")
    return model


def _train(model, max_iter, display, max_eval_batches, eval_interval, auc_threshold):
    model.start_data_reading()
    lr_sch = model.get_learning_rate_scheduler()
    reach_auc_threshold = False
    for iter in range(max_iter):
        lr = lr_sch.get_next()
        model.set_learning_rate(lr)
        model.train()
        if iter % display == 0:
            loss = model.get_current_loss()
            print("[HUGECTR][INFO] iter: {}; loss: {}".format(iter, loss))
        if iter % eval_interval == 0 and iter != 0:
            for _ in range(max_eval_batches):
                model.eval()
            metrics = model.get_eval_metrics()
            print("[HUGECTR][INFO] iter: {}, metrics: {}".format(iter, metrics))
            if metrics[0][1] > auc_threshold:
                reach_auc_threshold = True
                break
    if reach_auc_threshold == False:
        raise RuntimeError("Cannot reach the AUC threshold {}".format(auc_threshold))
        sys.exit(1)
    else:
        print("Successfully reach the AUC threshold {}".format(auc_threshold))
    pass


def main(args):
    solver = create_solver(
        max_eval_batches=args.max_eval_batches,
        batchsize_eval=args.batchsize_eval,
        batchsize=args.batchsize,
        vvgpu=args.vvgpu,
        lr=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_start=args.decay_start,
        decay_steps=args.decay_steps,
        decay_power=args.decay_power,
        end_lr=args.end_lr,
        i64_input_key=args.i64_input_key,
        use_mixed_precision=args.use_mixed_precision,
        scaler=args.scaler,
    )
    reader = create_data_reader(
        data_reader_type=args.data_reader_type,
        source=args.source,
        eval_source=args.eval_source,
        num_samples=args.num_samples,
        eval_num_samples=args.eval_num_samples,
        float_label_dense=args.float_label_dense,
        slot_size_array=args.slot_size_array,
    )
    optimizer = create_optimizer(
        optimizer_type=args.optimizer_type,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        update_type=args.update_type,
        momentum_factor=args.momentum_factor,
        atomic_update=args.atomic_update,
    )
    model = hugectr.Model(solver, reader, optimizer)
    model = construct_model(args.model_type, model, optimizer, args.single_slot)
    model.compile()
    model.summary()
    if args.auc_threshold != 0.5:
        _train(
            model,
            args.max_iter,
            args.display,
            args.max_eval_batches,
            args.eval_interval,
            args.auc_threshold,
        )
    else:
        model.fit(
            max_iter=args.max_iter,
            display=args.display,
            eval_interval=args.eval_interval,
            snapshot=args.snapshot,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_args(parser)
    main(args)
