import hugectr
import json
import sys
import argparse
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def create_wdl(solver):
    dataset_path = os.getenv("WDL_DATA_PATH")
    os.symlink(dataset_path, "./wdl_data", target_is_directory=True)
    reader = hugectr.DataReaderParams(
        data_reader_type=hugectr.DataReaderType_t.Norm,
        source=["./wdl_data/file_list.txt"],
        eval_source="./wdl_data/file_list_test.txt",
        check_type=hugectr.Check_t.Sum,
    )
    optimizer = hugectr.CreateOptimizer(
        optimizer_type=hugectr.Optimizer_t.SGD,
        update_type=hugectr.Update_t.Local,
        atomic_update=True,
    )
    model = hugectr.Model(solver, reader, optimizer)
    model.add(
        hugectr.Input(
            label_dim=1,
            label_name="label",
            dense_dim=13,
            dense_name="dense",
            data_reader_sparse_param_array=[
                hugectr.DataReaderSparseParam("wide_data", 30, True, 1),
                hugectr.DataReaderSparseParam("deep_data", 2, False, 26),
            ],
        )
    )
    model.add(
        hugectr.SparseEmbedding(
            embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
            workspace_size_per_gpu_in_mb=23,
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
            workspace_size_per_gpu_in_mb=358,
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
            leading_dim=416,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Reshape,
            bottom_names=["sparse_embedding2"],
            top_names=["reshape2"],
            leading_dim=1,
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
            layer_type=hugectr.Layer_t.Add,
            bottom_names=["fc3", "reshape2"],
            top_names=["add1"],
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


def create_dcn(solver):
    dataset_path = os.getenv("DCN_DATA_PATH")
    os.symlink(dataset_path, "./dcn_data", target_is_directory=True)
    reader = hugectr.DataReaderParams(
        data_reader_type=hugectr.DataReaderType_t.Norm,
        source=["./dcn_data/file_list.txt"],
        eval_source="./dcn_data/file_list_test.txt",
        check_type=hugectr.Check_t.Sum,
    )
    optimizer = hugectr.CreateOptimizer(
        optimizer_type=hugectr.Optimizer_t.SGD,
        update_type=hugectr.Update_t.Local,
        atomic_update=True,
    )
    model = hugectr.Model(solver, reader, optimizer)
    model.add(
        hugectr.Input(
            label_dim=1,
            label_name="label",
            dense_dim=13,
            dense_name="dense",
            data_reader_sparse_param_array=[
                hugectr.DataReaderSparseParam("data1", 2, False, 26)
            ],
        )
    )
    model.add(
        hugectr.SparseEmbedding(
            embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
            workspace_size_per_gpu_in_mb=300,
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


def create_deepfm(solver):
    dataset_path = os.getenv("DCN_DATA_PATH")
    os.symlink(dataset_path, "./dcn_data", target_is_directory=True)
    reader = hugectr.DataReaderParams(
        data_reader_type=hugectr.DataReaderType_t.Norm,
        source=["./dcn_data/file_list.txt"],
        eval_source="./dcn_data/file_list_test.txt",
        check_type=hugectr.Check_t.Sum,
    )
    optimizer = hugectr.CreateOptimizer(
        optimizer_type=hugectr.Optimizer_t.SGD,
        update_type=hugectr.Update_t.Local,
        atomic_update=True,
    )
    model = hugectr.Model(solver, reader, optimizer)
    model.add(
        hugectr.Input(
            label_dim=1,
            label_name="label",
            dense_dim=13,
            dense_name="dense",
            data_reader_sparse_param_array=[
                hugectr.DataReaderSparseParam("data1", 2, False, 26)
            ],
        )
    )
    model.add(
        hugectr.SparseEmbedding(
            embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
            workspace_size_per_gpu_in_mb=61,
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
            leading_dim=11,
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
            leading_dim=260,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Reshape,
            bottom_names=["slice12"],
            top_names=["reshape3"],
            leading_dim=26,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.WeightMultiply,
            bottom_names=["dense"],
            top_names=["weight_multiply1"],
            weight_dims=[13, 10],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.WeightMultiply,
            bottom_names=["dense"],
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
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["concat1"],
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
            bottom_names=["concat1"],
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


def create_din(solver):
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
            workspace_size_per_gpu_in_mb=28,
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
            workspace_size_per_gpu_in_mb=24,
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
            workspace_size_per_gpu_in_mb=10,
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
            layer_type=hugectr.Layer_t.Slice,
            bottom_names=["FusedReshapeConcat_item"],
            top_names=["item1", "item2"],
            ranges=[(0, 36), (0, 36)],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Slice,
            bottom_names=["FusedReshapeConcat_item_his_em"],
            top_names=["item_his1", "item_his2", "item_his3", "item_his4", "item_his5"],
            ranges=[(0, 36), (0, 36), (0, 36), (0, 36), (0, 36)],
        )
    )

    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Scale,
            bottom_names=["item1"],
            top_names=["Scale_item"],
            axis=1,
            factor=10,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Slice,
            bottom_names=["Scale_item"],
            top_names=["Scale_item1", "Scale_item2", "Scale_item3"],
            ranges=[(0, 36), (0, 36), (0, 36)],
        )
    )

    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Sub,
            bottom_names=["Scale_item1", "item_his1"],
            top_names=["sub_ih"],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.DotProduct,
            bottom_names=["Scale_item2", "item_his2"],
            top_names=["DotProduct_i"],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Concat,
            bottom_names=["Scale_item3", "item_his3", "sub_ih", "DotProduct_i"],
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
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Softmax,
            bottom_names=["reshape_score"],
            top_names=["softmax_att_i"],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Scale,
            bottom_names=["softmax_att_i"],
            top_names=["Scale_i"],
            axis=0,
            factor=36,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Reshape,
            bottom_names=["item_his4"],
            top_names=["reshape_item_his"],
            leading_dim=360,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.DotProduct,  # matmul
            bottom_names=["Scale_i", "reshape_item_his"],
            top_names=["DotProduct_ih"],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReduceSum,
            bottom_names=["DotProduct_ih"],
            top_names=["reduce_ih"],
            axis=1,
        )
    )

    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Reshape,
            bottom_names=["item_his5"],
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
                "reduce_ih",
                "item2",
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
    return model


def multi_node_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--batchsize_per_gpu", type=int, required=True)
    parser.add_argument("--node_num", type=int, required=True, default=1)
    parser.add_argument("--gpu_num", type=int, required=True, default=1)
    parser.add_argument("--use_mixed_precision", action='store_true', default=False)
    args = parser.parse_args()

    vvgpu = [[g for g in range(args.gpu_num)] for _ in range(args.node_num)]
    batchsize = args.batchsize_per_gpu * args.node_num * args.gpu_num

    args.i64_input_key = False
    if args.use_mixed_precision:
        args.scaler = 1024
    else:
        args.scaler = 1

    solver = hugectr.CreateSolver(
        max_eval_batches=1,  # we dont evaluate
        batchsize_eval=args.gpu_num * args.node_num,  # we dont evaluate
        batchsize=batchsize,
        vvgpu=vvgpu,
        lr=1e-3,
        i64_input_key=args.i64_input_key,
        use_mixed_precision=args.use_mixed_precision,
        scaler=args.scaler,
    )

    if args.benchmark.lower() == "wdl":
        model = create_wdl(solver)
    if args.benchmark.lower() == "din":
        model = create_din(solver)
    if args.benchmark.lower() == "dcn":
        model = create_dcn(solver)
    if args.benchmark.lower() == "deepfm":
        model = create_deepfm(solver)

    model.compile()
    model.summary()

    model.fit(
        max_iter=2000,
        display=200,
        eval_interval=3000,  # benchmark we dont want evalute
        snapshot=3000,  # benchmark we dont want snapshot
    )


if __name__ == "__main__":
    multi_node_test()
