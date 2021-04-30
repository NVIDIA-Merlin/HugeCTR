import hugectr
from mpi4py import MPI
solver = hugectr.CreateSolver(max_eval_batches = 300,
                              batchsize_eval = 16384,
                              batchsize = 16384,
                              lr = 0.001,
                              vvgpu = [[0]],
                              repeat_dataset = True)
reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,
                                  source = ["./criteo_data/train/_file_list.txt"],
                                  eval_source = "./criteo_data/val/_file_list.txt",
                                  check_type = hugectr.Check_t.Non)
optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam,
                                    update_type = hugectr.Update_t.Local,
                                    beta1 = 0.9,
                                    beta2 = 0.999,
                                    epsilon = 0.0000001)
model = hugectr.Model(solver, reader, optimizer)
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 0, dense_name = "dense",
                        data_reader_sparse_param_array = 
                        [hugectr.DataReaderSparseParam(hugectr.DataReaderSparse_t.Distributed, 100, 100, 26)],
                        sparse_names = ["data1"]))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
                            max_vocabulary_size_per_gpu = 1737715,
                            embedding_vec_size = 64,
                            combiner = 0,
                            sparse_embedding_name = "sparse_embedding1",
                            bottom_name = "data1",
                            optimizer = optimizer))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                            bottom_names = ["sparse_embedding1"],
                            top_names = ["reshape1"],
                            leading_dim=1664))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["reshape1"],
                            top_names = ["fc1"],
                            num_output=200))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc1"],
                            top_names = ["relu1"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["relu1"],
                            top_names = ["fc2"],
                            num_output=200))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc2"],
                            top_names = ["relu2"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["relu2"],
                            top_names = ["fc3"],
                            num_output=200))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc3"],
                            top_names = ["relu3"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["relu3"],
                            top_names = ["fc4"],
                            num_output=1))                                                                                           
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["fc4", "label"],
                            top_names = ["loss"]))
model.compile()
model.summary()
model.fit(max_iter = 2300, display = 200, eval_interval = 1000, snapshot = 1000000, snapshot_prefix = "deepfm")