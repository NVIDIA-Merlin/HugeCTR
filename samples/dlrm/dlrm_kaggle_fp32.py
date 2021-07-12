import hugectr
from mpi4py import MPI
solver = hugectr.CreateSolver(max_eval_batches = 70,
                              batchsize_eval = 65536,
                              batchsize = 65536,
                              lr = 0.5,
                              warmup_steps = 300,
                              vvgpu = [[0,1,2,3,4,5,6,7]],
                              repeat_dataset = True)
reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Raw,
                                  source = ["./train_data.bin"],
                                  eval_source = "./test_data.bin",
                                  num_samples = 36634624,
                                  eval_num_samples = 4584062,
                                  check_type = hugectr.Check_t.Non)
optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.SGD,
                                    update_type = hugectr.Update_t.Local,
                                    atomic_update = True)
model = hugectr.Model(solver, reader, optimizer)
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 13, dense_name = "dense",
                        data_reader_sparse_param_array = 
                        [hugectr.DataReaderSparseParam("data1", 2, False, 26)]))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.LocalizedSlotSparseEmbeddingOneHot, 
                            slot_size_array = [1459, 583, 6373320, 1977439, 305, 24, 12513, 633, 3, 92719, 5681, 5666265, 3193, 27, 14986, 4209368, 10, 5652, 2173, 4, 5058596, 18, 15, 282062, 105, 141594],
                            workspace_size_per_gpu_in_mb = 11645,
                            embedding_vec_size = 128,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding1",
                            bottom_name = "data1",
                            optimizer = optimizer))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["dense"],
                            top_names = ["fc1"],
                            num_output=512))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc1"],
                            top_names = ["relu1"]))                           
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["relu1"],
                            top_names = ["fc2"],
                            num_output=256))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc2"],
                            top_names = ["relu2"]))                            
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["relu2"],
                            top_names = ["fc3"],
                            num_output=128))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc3"],
                            top_names = ["relu3"]))                              
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Interaction,
                            bottom_names = ["relu3","sparse_embedding1"],
                            top_names = ["interaction1"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["interaction1"],
                            top_names = ["fc4"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc4"],
                            top_names = ["relu4"]))                              
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["relu4"],
                            top_names = ["fc5"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc5"],
                            top_names = ["relu5"]))                              
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["relu5"],
                            top_names = ["fc6"],
                            num_output=512))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc6"],
                            top_names = ["relu6"]))                               
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["relu6"],
                            top_names = ["fc7"],
                            num_output=256))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc7"],
                            top_names = ["relu7"]))                                                                              
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["relu7"],
                            top_names = ["fc8"],
                            num_output=1))                                                                                           
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["fc8", "label"],
                            top_names = ["loss"]))
model.compile()
model.summary()
model.fit(max_iter = 600, display = 50, eval_interval = 100, snapshot = 10000000, snapshot_prefix = "dlrm")
