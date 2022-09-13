import hugectr
from mpi4py import MPI

solver = hugectr.CreateSolver(
    model_name="wdl",
    max_eval_batches=1,
    batchsize_eval=16384,
    batchsize=16384,
    lr=0.001,
    vvgpu=[[0]],
    repeat_dataset=True,
)
reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.Norm,
    source=["./wdl_data/file_list.txt"],
    eval_source="./wdl_data/file_list_test.txt",
    check_type=hugectr.Check_t.Sum,
)
optimizer = hugectr.CreateOptimizer(
    optimizer_type=hugectr.Optimizer_t.Adam,
    update_type=hugectr.Update_t.Global,
    beta1=0.9,
    beta2=0.999,
    epsilon=0.0000001,
)
model = hugectr.Model(solver, reader, optimizer)
model.add(
    hugectr.Input(
        label_dim=1,
        label_name="label",
        dense_dim=13,
        dense_name="dense",
        data_reader_sparse_param_array=[
            hugectr.DataReaderSparseParam("wide_data", 30, False, 1),
            hugectr.DataReaderSparseParam("deep_data", 2, False, 26),
        ],
    )
)
model.add(
    hugectr.SparseEmbedding(
        embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
        workspace_size_per_gpu_in_mb=69,
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
        workspace_size_per_gpu_in_mb=1074,
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
        layer_type=hugectr.Layer_t.Concat, bottom_names=["reshape1", "dense"], top_names=["concat1"]
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
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["dropout1"],
        top_names=["fc2"],
        num_output=1024,
    )
)
model.add(
    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc2"], top_names=["relu2"])
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
model.compile()
model.summary()
model.graph_to_json(graph_config_file="/dump_infer/wdl.json")
model.fit(
    max_iter=2300, display=200, eval_interval=2000, snapshot=2000, snapshot_prefix="/dump_infer/wdl"
)
model.export_predictions("/dump_infer/wdl_pred_" + str(2000), "/dump_infer/wdl_label_" + str(2000))


from hugectr.inference import InferenceModel, InferenceParams
import numpy as np

batch_size = 16384
num_batches = 1
data_source = "./wdl_data/file_list_test.txt"
inference_params = InferenceParams(
    model_name="wdl",
    max_batchsize=batch_size,
    hit_rate_threshold=1.0,
    dense_model_file="/dump_infer/wdl_dense_2000.model",
    sparse_model_files=["/dump_infer/wdl0_sparse_2000.model", "/dump_infer/wdl1_sparse_2000.model"],
    device_id=0,
    use_gpu_embedding_cache=False,
    cache_size_percentage=1.0,
    i64_input_key=False,
    use_mixed_precision=True,
    use_cuda_graph=True,
)
inference_model = InferenceModel("/dump_infer/wdl.json", inference_params)
predictions = inference_model.predict(
    num_batches=num_batches,
    source=data_source,
    data_reader_type=hugectr.DataReaderType_t.Norm,
    check_type=hugectr.Check_t.Sum,
)
grount_truth = np.loadtxt("/dump_infer/wdl_pred_2000")
diff = predictions.flatten() - grount_truth
mse = np.mean(diff * diff)
if mse > 1e-3:
    raise RuntimeError("Too large mse between WDL multi hot inference and training: {}".format(mse))
    sys.exit(1)
else:
    print(
        "WDL multi hot inference results are consistent with those during training, mse: {}".format(
            mse
        )
    )
