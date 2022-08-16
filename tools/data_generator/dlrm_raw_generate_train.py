import hugectr
from hugectr.tools import DataGenerator, DataGeneratorParams
from mpi4py import MPI

# Raw data generation
data_generator_params = DataGeneratorParams(
    format=hugectr.DataReaderType_t.Raw,
    label_dim=1,
    dense_dim=13,
    num_slot=26,
    i64_input_key=False,
    source="./dlrm_raw/train_data.bin",
    eval_source="./dlrm_raw/test_data.bin",
    slot_size_array=[
        203931,
        18598,
        14092,
        7012,
        18977,
        4,
        6385,
        1245,
        49,
        186213,
        71328,
        67288,
        11,
        2168,
        7338,
        61,
        4,
        932,
        15,
        204515,
        141526,
        199433,
        60919,
        9137,
        71,
        34,
    ],
    nnz_array=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
)
data_generator = DataGenerator(data_generator_params)
data_generator.generate()


# DLRM train
solver = hugectr.CreateSolver(
    max_eval_batches=1280,
    batchsize_eval=1024,
    batchsize=1024,
    lr=0.5,
    warmup_steps=500,
    vvgpu=[[0]],
    repeat_dataset=True,
)
reader = hugectr.DataReaderParams(
    data_reader_type=data_generator_params.format,
    source=[data_generator_params.source],
    eval_source=data_generator_params.eval_source,
    num_samples=data_generator_params.num_samples,
    eval_num_samples=data_generator_params.eval_num_samples,
    check_type=data_generator_params.check_type,
)
optimizer = hugectr.CreateOptimizer(
    optimizer_type=hugectr.Optimizer_t.SGD, update_type=hugectr.Update_t.Local, atomic_update=True
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
        embedding_type=hugectr.Embedding_t.LocalizedSlotSparseEmbeddingOneHot,
        slot_size_array=data_generator_params.slot_size_array,
        workspace_size_per_gpu_in_mb=800,
        embedding_vec_size=128,
        combiner="sum",
        sparse_embedding_name="sparse_embedding1",
        bottom_name="data1",
        optimizer=optimizer,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["dense"],
        top_names=["fc1"],
        num_output=512,
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
        num_output=256,
    )
)
model.add(
    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc2"], top_names=["relu2"])
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["relu2"],
        top_names=["fc3"],
        num_output=128,
    )
)
model.add(
    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc3"], top_names=["relu3"])
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Interaction,
        bottom_names=["relu3", "sparse_embedding1"],
        top_names=["interaction1"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["interaction1"],
        top_names=["fc4"],
        num_output=1024,
    )
)
model.add(
    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc4"], top_names=["relu4"])
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["relu4"],
        top_names=["fc5"],
        num_output=1024,
    )
)
model.add(
    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc5"], top_names=["relu5"])
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["relu5"],
        top_names=["fc6"],
        num_output=512,
    )
)
model.add(
    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc6"], top_names=["relu6"])
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["relu6"],
        top_names=["fc7"],
        num_output=256,
    )
)
model.add(
    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc7"], top_names=["relu7"])
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["relu7"],
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
model.compile()
model.summary()
model.fit(max_iter=5120, display=200, eval_interval=1000)
