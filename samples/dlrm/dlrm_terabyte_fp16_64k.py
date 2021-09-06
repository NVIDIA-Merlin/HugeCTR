import hugectr
from mpi4py import MPI
solver = hugectr.CreateSolver(max_eval_batches = 1361,
                              batchsize_eval = 65536,
                              batchsize = 65536,
                              lr = 24.0,
                              warmup_steps = 8000,
                              decay_start = 48000,
                              decay_steps = 24000,
                              decay_power = 2.0,
                              end_lr = 0.0,
                              vvgpu = [[0,1,2,3,4,5,6,7]],
                              repeat_dataset = True,
                              use_mixed_precision = True,
                              scaler = 1024)
reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Raw,
                                  source = ["./train_data.bin"],
                                  eval_source = "./test_data.bin",
                                  num_samples = 4195197692,
                                  eval_num_samples = 89137319,
                                  check_type = hugectr.Check_t.Non,
                                  cache_eval_data = 1361)
optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.SGD,
                                    update_type = hugectr.Update_t.Local,
                                    atomic_update = True)
model = hugectr.Model(solver, reader, optimizer)
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 13, dense_name = "dense",
                        data_reader_sparse_param_array = 
                        [hugectr.DataReaderSparseParam("data1", 2, False, 26)]))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.LocalizedSlotSparseEmbeddingOneHot, 
                            slot_size_array = [39884407, 39043, 17289, 7420, 20263, 3, 7120, 1543, 63, 38532952, 2953546, 403346, 10, 2208, 11938, 155, 4, 976, 14, 39979772, 25641295, 39664985, 585935, 12972, 108, 36],
                            workspace_size_per_gpu_in_mb = 91684,
                            embedding_vec_size = 128,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding1",
                            bottom_name = "data1",
                            optimizer = optimizer))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,
                            bottom_names = ["dense"],
                            top_names = ["fc1"],
                            num_output=512))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,
                            bottom_names = ["fc1"],
                            top_names = ["fc2"],
                            num_output=256))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,
                            bottom_names = ["fc2"],
                            top_names = ["fc3"],
                            num_output=128))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Interaction,
                            bottom_names = ["fc3","sparse_embedding1"],
                            top_names = ["interaction1"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,
                            bottom_names = ["interaction1"],
                            top_names = ["fc4"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,
                            bottom_names = ["fc4"],
                            top_names = ["fc5"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,
                            bottom_names = ["fc5"],
                            top_names = ["fc6"],
                            num_output=512))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,
                            bottom_names = ["fc6"],
                            top_names = ["fc7"],
                            num_output=256))                                                  
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["fc7"],
                            top_names = ["fc8"],
                            num_output=1))                                                                                           
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["fc8", "label"],
                            top_names = ["loss"]))
model.compile()
model.summary()
model.fit(max_iter = 64013, display = 1000, eval_interval = 3200, snapshot = 10000000, snapshot_prefix = "dlrm")
