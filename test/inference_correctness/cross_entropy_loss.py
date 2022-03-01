import hugectr
from mpi4py import MPI

solver = hugectr.CreateSolver(
    max_eval_batches = 1,
    batchsize_eval = 1024,
    batchsize = 1024,
    lr = 0.01,
    end_lr = 0.0001,
    warmup_steps = 8000,
    decay_start = 48000,
    decay_steps = 24000,
    vvgpu = [[0]],
    repeat_dataset = True,
    i64_input_key = True
)
reader = hugectr.DataReaderParams(
    data_reader_type = hugectr.DataReaderType_t.Parquet,
    source = ["./cross/data/train/_file_list.txt"],
    eval_source = "./cross/data/test/_file_list.txt",
    check_type = hugectr.Check_t.Sum,
    slot_size_array = [10001, 10001, 10001, 10001]
)
optimizer = hugectr.CreateOptimizer(
    optimizer_type = hugectr.Optimizer_t.Adam,
    update_type = hugectr.Update_t.Local,
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 0.0000001
)
model = hugectr.Model(solver, reader, optimizer)
num_gpus = 1
workspace_size_per_gpu_in_mb = (
    int(
        40004
        * 16
        * 4
        * 3
        / 1000000
    )
    + 10
)
model.add(
    hugectr.Input(
        label_dim=2,
        label_name="label",
        dense_dim=3,
        dense_name="dense",
        data_reader_sparse_param_array=[
            hugectr.DataReaderSparseParam(
                "data1",
                [1,1,1,1],
                False,
                4,
            )
        ],
    )
)
model.add(
    hugectr.SparseEmbedding(
        embedding_type=hugectr.Embedding_t.LocalizedSlotSparseEmbeddingHash,
        workspace_size_per_gpu_in_mb=workspace_size_per_gpu_in_mb,
        embedding_vec_size=16,
        combiner="mean",
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
        num_output=16,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ReLU,
        bottom_names=["fc1"],
        top_names=["relu1"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Interaction,
        bottom_names=["relu1", "sparse_embedding1"],
        top_names=["interaction1"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["interaction1"],
        top_names=["fc4"],
        num_output=32,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ReLU,
        bottom_names=["fc4"],
        top_names=["relu4"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["relu4"],
        top_names=["fc8"],
        num_output=2,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.CrossEntropyLoss,
        bottom_names=["fc8", "label"],
        top_names=["loss"]
    )
)
model.compile()
model.summary()
model.graph_to_json(graph_config_file='/dump_infer/cross_entropy_loss.json')
model.fit(max_iter = 1001, display = 100, eval_interval = 1000, snapshot = 1000, snapshot_prefix = "/dump_infer/cross_entropy_loss")
model.export_predictions("/dump_infer/cross_entropy_loss_pred_" + str(1000), "/dump_infer/cross_entropy_loss_label_" + str(1000))


from hugectr.inference import InferenceParams, CreateInferenceSession
import numpy as np

batch_size = 1024
num_batches = 1
inference_params = InferenceParams(
    model_name = "cross_entropy_loss",
    max_batchsize=batch_size,
    hit_rate_threshold=1.0,
    dense_model_file="/dump_infer/cross_entropy_loss_dense_1000.model",
    sparse_model_files=["/dump_infer/cross_entropy_loss0_sparse_1000.model"],
    device_id=0,
    use_gpu_embedding_cache=True,
    cache_size_percentage=0.5,
    use_mixed_precision=True,
    i64_input_key=True,
)

inference_session = CreateInferenceSession(
    '/dump_infer/cross_entropy_loss.json', inference_params
)

preds = inference_session.predict(
    num_batches=num_batches,
    source="./cross/data/test/_file_list.txt",
    data_reader_type=hugectr.DataReaderType_t.Parquet,
    check_type=hugectr.Check_t.Sum,
    slot_size_array=[10001, 10001, 10001, 10001],
)

ground_truth = np.loadtxt("/dump_infer/cross_entropy_loss_pred_1000")
predictions = preds.flatten()
diff = predictions-ground_truth
mse = np.mean(diff*diff)
if mse > 1e-3:
  raise RuntimeError("Too large mse between cross_entropy_loss inference and training: {}".format(mse))
  sys.exit(1)
else:
  print("cross_entropy_loss inference results are consistent with those during training, mse: {}".format(mse))