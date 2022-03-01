import hugectr

solver = hugectr.CreateSolver(max_eval_batches = 100,
                              batchsize_eval = 2048,
                              batchsize = 2048,
                              vvgpu = [[0]],
                              i64_input_key = False,
                              use_mixed_precision = False,
                              repeat_dataset = True,
                              use_cuda_graph = True)
reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,
                                  source = ["./file_list.txt"],
                                  eval_source = "./file_list_test.txt",
                                  check_type = hugectr.Check_t.Sum)
optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam)
model = hugectr.Model(solver, reader, optimizer)
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 13, dense_name = "dense",
                        data_reader_sparse_param_array = 
                            [hugectr.DataReaderSparseParam("data1", 1, False, 26)]))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
                            workspace_size_per_gpu_in_mb = 321,
                            embedding_vec_size = 16,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding1",
                            bottom_name = "data1",
                            optimizer = optimizer))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                            bottom_names = ["sparse_embedding1"],
                            top_names = ["reshape1"],
                            leading_dim=416))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,
                            bottom_names = ["reshape1", "dense"], top_names = ["concat1"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.MultiCross,
                            bottom_names = ["concat1"],
                            top_names = ["multicross1"],
                            num_layers=6))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["concat1"],
                            top_names = ["fc1"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc1"],
                            top_names = ["relu1"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                            bottom_names = ["relu1"],
                            top_names = ["dropout1"],
                            dropout_rate=0.5))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["dropout1"],
                            top_names = ["fc2"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc2"],
                            top_names = ["relu2"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                            bottom_names = ["relu2"],
                            top_names = ["dropout2"],
                            dropout_rate=0.5))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,
                            bottom_names = ["dropout2", "multicross1"],
                            top_names = ["concat2"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["concat2"],
                            top_names = ["fc3"],
                            num_output=1))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["fc3", "label"],
                            top_names = ["loss"]))
model.compile()
model.summary()
model.fit(max_iter = 12000, display = 200, eval_interval = 1000, snapshot = 100000, snapshot_prefix = "dcn")






