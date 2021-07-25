import hugectr
from mpi4py import MPI
solver = hugectr.CreateSolver(max_eval_batches = 100,
                              batchsize_eval = 27700,
                              batchsize = 175480,
                              lr = 0.0045,
                              vvgpu = [[0]],
                              metrics_spec = {hugectr.MetricsType.HitRate: 0.8,
                                              hugectr.MetricsType.AverageLoss:0.0,
                                              hugectr.MetricsType.AUC: 1.0},
                              repeat_dataset = False)
reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,
                                  source = ["./data/ml-20m/train_filelist.txt"],
                                  eval_source = "./data/ml-20m/test_filelist.txt",
                                  check_type = hugectr.Check_t.Non,
                                  num_workers = 10)
optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam,
                                    update_type = hugectr.Update_t.Global,
                                    beta1 = 0.25,
                                    beta2 = 0.5,
                                    epsilon = 0.0000001)
model = hugectr.Model(solver, reader, optimizer)

# MLP side of the NeuMF model
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 1, dense_name = "dense",
                        data_reader_sparse_param_array = 
                        [hugectr.DataReaderSparseParam("data", 1, True, 2)]))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
                            workspace_size_per_gpu_in_mb = 60,
                            embedding_vec_size = 72,
                            combiner = "sum",
                            sparse_embedding_name = "mixed_embedding",
                            bottom_name = "data",
                            optimizer = optimizer))

model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                            bottom_names = ["mixed_embedding"],
                            top_names = ["reshape_embedding"],
                            leading_dim=144))

model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Slice,
                            bottom_names = ["reshape_embedding"],
                            top_names = ["mlp_embedding", "gmf_embedding"],
                            ranges=[(0,127),(128,143)]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["mlp_embedding"],
                            top_names = ["fc1"],
                            num_output=256))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc1"],
                            top_names = ["relu1"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                            bottom_names = ["relu1"],
                            top_names = ["dropout1"],
                            dropout_rate = 0.5))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["dropout1"],
                            top_names = ["fc2"],
                            num_output=256))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc2"],
                            top_names = ["relu2"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                            bottom_names = ["relu2"],
                            top_names = ["dropout2"],
                            dropout_rate = 0.5))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["dropout2"],
                            top_names = ["fc3"],
                            num_output=128))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc3"],
                            top_names = ["relu3"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                            bottom_names = ["relu3"],
                            top_names = ["dropout3"],
                            dropout_rate = 0.5))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["dropout3"],
                            top_names = ["fc4"],
                            num_output=64))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc4"],
                            top_names = ["relu4"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                            bottom_names = ["relu4"],
                            top_names = ["mlp_dropout4"],
                            dropout_rate = 0.5))


# GMF side of the NeuMF model
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Slice,
                            bottom_names = ["gmf_embedding"],
                            top_names = ["user", "item"],
                            ranges=[(0,7),(8,15)]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.DotProduct,
                            bottom_names = ["user", "item"],
                            top_names = ["gmf_multiply"]))


#Combine MLP and GMF outputs for final NeuMF prediction
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,
                            bottom_names = ["gmf_multiply", "mlp_dropout4"],
                            top_names = ["concat"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["concat"],
                            top_names = ["neumf_out"],
                            num_output=1))

model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["neumf_out", "label"],
                            top_names = ["loss"]))
model.compile()
model.summary()
model.fit(num_epochs = 10, display = 200, eval_interval = 200, snapshot = 100000, snapshot_prefix = "neumf")
