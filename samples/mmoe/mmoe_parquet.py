import hugectr
from mpi4py import MPI

solver = hugectr.CreateSolver(
    max_eval_batches=100,
    batchsize_eval=1000,
    batchsize=1000,
    lr=0.001,
    vvgpu=[[0]],
    metrics_spec={hugectr.MetricsType.AverageLoss: 0.0},
    repeat_dataset=False,
    i64_input_key=True,
)
reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.Parquet,
    source=["./mmoe_parquet/file_list.txt"],
    eval_source="./mmoe_parquet/file_list_test.txt",
    check_type=hugectr.Check_t.Non,
    slot_size_array=[100000, 100000],
)
optimizer = hugectr.CreateOptimizer(
    optimizer_type=hugectr.Optimizer_t.Adam,
    update_type=hugectr.Update_t.Global,
    beta1=0.25,
    beta2=0.5,
    epsilon=0.0000001,
)
model = hugectr.Model(solver, reader, optimizer)
model.add(
    hugectr.Input(
        label_dims=[1, 1],
        label_names=["labelA", "labelB"],
        dense_dim=1,
        dense_name="dense",
        data_reader_sparse_param_array=[hugectr.DataReaderSparseParam("data", 1, True, 2)],
    )
)
model.add(
    hugectr.SparseEmbedding(
        embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
        workspace_size_per_gpu_in_mb=100,
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
        leading_dim=128,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["reshape_embedding"],
        top_names=["shared1"],
        num_output=256,
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
        ranges=[(0, 127), (128, 255)],
    )
)

# "A" side and corresponding loss
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Reshape,
        bottom_names=["sliceA"],
        top_names=["reshapeA"],
        leading_dim=10,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["reshapeA"],
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
        top_names=["A_out"],
        num_output=1,
    )
)

# "B" side and corresponding loss
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Reshape,
        bottom_names=["sliceB"],
        top_names=["reshapeB"],
        leading_dim=10,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["reshapeB"],
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
        bottom_names=["A_out", "labelA"],
        top_names=["lossA"],
    )
)

model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
        bottom_names=["B_out", "labelB"],
        top_names=["lossB"],
    )
)

model.compile(loss_names=["labelA", "labelB"], loss_weights=[0.5, 0.5])
model.summary()
model.fit(num_epochs=10, display=50, eval_interval=50, snapshot=1000000, snapshot_prefix="mmoe")
