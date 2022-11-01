import hugectr
from mpi4py import MPI

solver = hugectr.CreateSolver(
    model_name="dcn",
    max_eval_batches=1,
    batchsize_eval=16384,
    batchsize=16384,
    lr=0.001,
    vvgpu=[[0]],
    repeat_dataset=True,
    use_mixed_precision=False,
    scaler=1.0,
    use_cuda_graph=True,
    metrics_spec={hugectr.MetricsType.AUC: 1.0},
)
reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.Norm,
    source=["./dcn_data/file_list.txt"],
    eval_source="./dcn_data/file_list_test.txt",
    check_type=hugectr.Check_t.Sum,
    num_workers=16,
)
optimizer = hugectr.CreateOptimizer(
    optimizer_type=hugectr.Optimizer_t.Adam,
    update_type=hugectr.Update_t.Global,
    beta1=0.9,
    beta2=0.999,
    epsilon=0.0001,
)
model = hugectr.Model(solver, reader, optimizer)
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
        workspace_size_per_gpu_in_mb=2400,
        embedding_vec_size=128,
        combiner="sum",
        sparse_embedding_name="sparse_embedding1",
        bottom_name="data1",
        optimizer=optimizer,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.MLP,
        bottom_names=["dense"],
        top_names=["mlp1"],
        num_outputs=[512, 256, 128],
        act_type=hugectr.Activation_t.Relu,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Interaction,
        bottom_names=["mlp1", "sparse_embedding1"],
        top_names=["interaction1"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.MLP,
        bottom_names=["interaction1"],
        top_names=["mlp2"],
        num_outputs=[1024, 1024, 512, 256, 1],
        activations=[
            hugectr.Activation_t.Relu,
            hugectr.Activation_t.Relu,
            hugectr.Activation_t.Relu,
            hugectr.Activation_t.Relu,
            hugectr.Activation_t.Non,
        ],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
        bottom_names=["mlp2", "label"],
        top_names=["loss"],
    )
)
model.compile()
model.summary()
model.graph_to_json(graph_config_file="/dump_infer/dlrm.json")
model.fit(
    max_iter=2300,
    display=200,
    eval_interval=2000,
    snapshot=2000,
    snapshot_prefix="/dump_infer/dlrm",
)
model.export_predictions(
    "/dump_infer/dlrm_pred_" + str(2000), "/dump_infer/dlrm_label_" + str(2000)
)


from hugectr.inference import InferenceModel, InferenceParams
import numpy as np

batch_size = 16384
num_batches = 1
data_source = "./dcn_data/file_list_test.txt"
inference_params = InferenceParams(
    model_name="dlrm",
    max_batchsize=batch_size,
    hit_rate_threshold=1.0,
    dense_model_file="/dump_infer/dlrm_dense_2000.model",
    sparse_model_files=["/dump_infer/dlrm0_sparse_2000.model"],
    device_id=0,
    use_gpu_embedding_cache=False,
    cache_size_percentage=1.0,
    i64_input_key=False,
    use_mixed_precision=False,
    use_cuda_graph=True,
)
inference_model = InferenceModel("/dump_infer/dlrm.json", inference_params)
predictions = inference_model.predict(
    num_batches=num_batches,
    source=data_source,
    data_reader_type=hugectr.DataReaderType_t.Norm,
    check_type=hugectr.Check_t.Sum,
)
grount_truth = np.loadtxt("/dump_infer/dlrm_pred_2000")
diff = predictions.flatten() - grount_truth
mse = np.mean(diff * diff)
if mse > 1e-3:
    raise RuntimeError("Too large mse between DLRM one hot inference and training: {}".format(mse))
    sys.exit(1)
else:
    print(
        "DLRM one hot inference results are consistent with those during training, mse: {}".format(
            mse
        )
    )
