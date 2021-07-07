import hugectr
from mpi4py import MPI
solver = hugectr.CreateSolver(max_eval_batches = 100,
                              batchsize_eval = 2**16,
                              batchsize = 2**16,
                              lr = 0.0045,
                              vvgpu = [[0]],
                              metrics_spec = {hugectr.MetricsType.HitRate: 0.8,
                                              hugectr.MetricsType.AverageLoss:0.0,
                                              hugectr.MetricsType.AUC: 1.0},
                              repeat_dataset = False)
reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,
                                  source = ["./data/ml-1m/train_filelist.txt"],
                                  eval_source = "./data/ml-1m/test_filelist.txt",
                                  check_type = hugectr.Check_t.Non)
optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam,
                                    update_type = hugectr.Update_t.Global,
                                    beta1 = 0.25,
                                    beta2 = 0.5,
                                    epsilon = 0.001)
model = hugectr.Model(solver, reader, optimizer)
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 1, dense_name = "dense",
                        data_reader_sparse_param_array = 
                        [hugectr.DataReaderSparseParam(hugectr.DataReaderSparse_t.Distributed, 2, 1, 2)],
                        sparse_names = ["data"]))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
                            max_vocabulary_size_per_gpu = 12000,
                            embedding_vec_size = 8,
                            combiner = 0,
                            sparse_embedding_name = "gmf_embedding",
                            bottom_name = "data",
                            optimizer = optimizer))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                            bottom_names = ["gmf_embedding"],
                            top_names = ["reshape1"],
                            leading_dim=16))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Slice,
                            bottom_names = ["reshape1"],
                            top_names = ["user", "item"],
                            ranges=[(0,7),(8,15)]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ElementwiseMultiply,
                            bottom_names = ["user", "item"],
                            top_names = ["multiply1"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["multiply1"],
                            top_names = ["gmf_out"],
                            num_output=1))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["gmf_out", "label"],
                            top_names = ["loss"]))
model.compile()
model.summary()
#model.fit(max_iter = 10000, display = 200, eval_interval = 3000, snapshot = 10000, snapshot_prefix = "gmf")
model.fit(num_epochs = 10, display = 100, eval_interval = 100, snapshot = 1000, snapshot_prefix = "gmf")
