import hugectr
from mpi4py import MPI
solver = hugectr.CreateSolver(max_eval_batches = 1000,
                              batchsize_eval = 2770,
                              batchsize = 17548,
                              lr = 0.0045,
                              vvgpu = [[0]],
                              metrics_spec = {hugectr.MetricsType.HitRate: 0.8,
                                              hugectr.MetricsType.AverageLoss:0.0,
                                              hugectr.MetricsType.AUC: 1.0},
                              repeat_dataset = True)
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
                        [hugectr.DataReaderSparseParam("data", 1, True, 2)]))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
                            workspace_size_per_gpu_in_mb = 20,
                            embedding_vec_size = 16,
                            combiner = "sum",
                            sparse_embedding_name = "gmf_embedding",
                            bottom_name = "data",
                            optimizer = optimizer))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                            bottom_names = ["gmf_embedding"],
                            top_names = ["reshape1"],
                            leading_dim=32))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Slice,
                            bottom_names = ["reshape1"],
                            top_names = ["user", "item"],
                            ranges=[(0,15),(16,31)]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.DotProduct,
                            bottom_names = ["user", "item"],
                            top_names = ["multiply1"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["multiply1"],
                            top_names = ["gmf_out"],
                            num_output=1))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["gmf_out", "label"],
                            top_names = ["loss"]))
model.graph_to_json("/onnx_converter/graph_files/gmf.json") 
model.compile()
model.summary()
model.fit(max_iter = 2100, display = 200, eval_interval = 1000, snapshot = 2000, snapshot_prefix = "/onnx_converter/hugectr_models//gmf")
