import hugectr
from mpi4py import MPI
# 1. Create Solver, DataReaderParams and Optimizer
solver = hugectr.CreateSolver(max_eval_batches = 51,
                              batchsize_eval = 1769472,
                              batchsize = 55296,
                              vvgpu = [[0,1,2,3,4,5,6,7]],
                              repeat_dataset = True,
                              lr = 24.0,
                              warmup_steps = 2750,
                              decay_start = 49315,
                              decay_steps = 27772,
                              decay_power = 2.0,
                              end_lr = 0.0,
                              use_mixed_precision = True,
                              scaler = 1024,
                              use_cuda_graph = False,
                              async_mlp_wgrad = True,
                              gen_loss_summary = False,
                              overlap_lr = True,
                              overlap_init_wgrad = True,
                              overlap_ar_a2a = True,
                              use_holistic_cuda_graph = True,
                              use_overlapped_pipeline = True,
                              all_reduce_algo = hugectr.AllReduceAlgo.OneShot,
                              grouped_all_reduce = False,
                              num_iterations_statistics = 20,
                              metrics_spec = {hugectr.MetricsType.AUC: 0.8025},
                              is_dlrm = True)
reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.RawAsync,
                                  source = ["/raid/datasets/criteo/mlperf/40m.limit_preshuffled/train_data.bin"],
                                  eval_source = "/raid/datasets/criteo/mlperf/40m.limit_preshuffled/test_data.bin",
                                  check_type = hugectr.Check_t.Non,
                                  num_samples = 4195196928,
                                  eval_num_samples = 89137319,
                                  cache_eval_data = 51,
                                  slot_size_array = [39884406,    39043,    17289,     7420,    20263,    3,  7120,     1543,  63, 38532951,  2953546,   403346, 10,     2208,    11938,      155,        4,      976, 14, 39979771, 25641295, 39664984,   585935,    12972,  108,  36],
                                  async_param = hugectr.AsyncParam(32, 4, 552960, 2, 512, True, hugectr.Alignment_t.Auto))
optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.SGD,
                                    update_type = hugectr.Update_t.Local,
                                    atomic_update = True)
# 2. Initialize the Model instance
model = hugectr.ModelPerfExt(solver, reader, optimizer)
# 3. Construct the Model graph
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 13, dense_name = "dense",
                        data_reader_sparse_param_array =
                        [hugectr.DataReaderSparseParam("data1", 1, True, 26)]))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.HybridSparseEmbedding,
                            workspace_size_per_gpu_in_mb = 15000,
                            slot_size_array =  [39884406,    39043,    17289,     7420,    20263,    3,  7120,     1543,  63, 38532951,  2953546,   403346, 10,     2208,    11938,      155,        4,      976, 14, 39979771, 25641295, 39664984,   585935,    12972,  108,  36],
                            embedding_vec_size = 128,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding1",
                            bottom_name = "data1",
                            optimizer = optimizer,
                            hybrid_embedding_param = hugectr.HybridEmbeddingParam(2, -1, 0.01, 1.3e11, 1.9e11, 1.0,
                                                                                hugectr.CommunicationType.NVLink_SingleNode,
                                                                                hugectr.HybridEmbeddingType.Distributed)))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,
                            pos_type = hugectr.FcPosition_t.Head,
                            bottom_names = ["dense"],
                            top_names = ["fc11","fc12", "fc13", "fc14"],
                            num_output=512))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,
                            pos_type = hugectr.FcPosition_t.Body,
                            bottom_names = ["fc11","fc12", "fc13", "fc14"],
                            top_names = ["fc21","fc22", "fc23", "fc24"],
                            num_output=256))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,
                            pos_type = hugectr.FcPosition_t.Tail,
                            bottom_names = ["fc21","fc22", "fc23", "fc24"],
                            top_names = ["fc3"],
                            num_output=128))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Interaction,
                            bottom_names = ["fc3","sparse_embedding1"],
                            top_names = ["interaction1", "interaction_grad"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,
                            pos_type = hugectr.FcPosition_t.Head,
                            bottom_names = ["interaction1", "interaction_grad"],
                            top_names = ["fc41","fc42", "fc43", "fc44"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,
                            pos_type = hugectr.FcPosition_t.Body,
                            bottom_names = ["fc41","fc42", "fc43", "fc44"],
                            top_names = ["fc51","fc52", "fc53", "fc54"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,
                            pos_type = hugectr.FcPosition_t.Body,
                            bottom_names = ["fc51","fc52", "fc53", "fc54"],
                            top_names = ["fc61","fc62", "fc63", "fc64"],
                            num_output=512))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,
                            pos_type = hugectr.FcPosition_t.Body,
                            bottom_names = ["fc61","fc62", "fc63", "fc64"],
                            top_names = ["fc71","fc72","fc73","fc74"],
                            num_output=256))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,
                            pos_type = hugectr.FcPosition_t.Tail,
                            act_type = hugectr.Activation_t.Non,
                            bottom_names = ["fc71","fc72","fc73","fc74"],
                            top_names = ["fc8"],
                            num_output=1))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["fc8", "label"],
                            top_names = ["loss"]))
# 4. Dump the Model graph to JSON
model.graph_to_json(graph_config_file = "dlrm.json")
# 5. Compile & Fit
model.compile()
model.summary()
model.fit(max_iter = 75868, display = 1000, eval_interval = 3793, snapshot = 2000000, snapshot_prefix = "dlrm")
