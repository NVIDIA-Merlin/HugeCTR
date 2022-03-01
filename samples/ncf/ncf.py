import hugectr
from mpi4py import MPI
solver = hugectr.CreateSolver(max_eval_batches = 100,
                              batchsize_eval = 27700, # 1208 for 1M dataset
                              batchsize = 175480, # 32205 for 1M dataset
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
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 1, dense_name = "dense",
                        data_reader_sparse_param_array = 
                        [hugectr.DataReaderSparseParam("data", 1, True, 2)]
                        ))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
                            workspace_size_per_gpu_in_mb = 147, # 3 for 1M dataset
                            embedding_vec_size = 64,
                            combiner = "sum",
                            sparse_embedding_name = "mlp_embedding",
                            bottom_name = "data",
                            optimizer = optimizer))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                            bottom_names = ["mlp_embedding"],
                            top_names = ["reshape_mlp"],
                            leading_dim=128))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["reshape_mlp"],
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
                            top_names = ["dropout4"],
                            dropout_rate = 0.5))

model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["dropout4"],
                            top_names = ["mlp_out"],
                            num_output=1))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["mlp_out", "label"],
                            top_names = ["loss"]))
model.compile()
model.summary()
model.fit(num_epochs = 10, display = 200, eval_interval = 200, snapshot = 1000000, snapshot_prefix = "ncf")  # display = 50 for 1M
