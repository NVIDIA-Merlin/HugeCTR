import hugectr
from mpi4py import MPI

LABEL_dIM = 1
DENSE_DIM = 13
NUM_SLOT = 26
SLOT_SIZE_ARRAY = [1459, 583, 6373320, 1977439, 305, 24, 12513, 633, 3, 92719, 5681, 5666265, 3193, 27, 14986, 4209368, 10, 5652, 2173, 4, 5058596, 18, 15, 282062, 105, 141594]
NNZ_ARRAY = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
NUM_SAMPLES = 36672493
EVAL_NUM_SAMPLES = 4584062
SOURCE = "./train_data.bin"
EVAL_SOURCE = "./test_data.bin"

ASYNC_MLP_WGRAD = False
GEN_LOSS_SUMMARY = False
OVERLAP_LR = True
OVERLAP_INIT_WGRAD = True
OVERLAP_AR_A2A = True
USE_CUDA_GRAPH = False
USE_HOLISTIC_CUDA_GRAPH = True
USE_OVERLAPPED_PIPELINE = True
ALL_REDUCE_ALGO = hugectr.AllReduceAlgo.OneShot
GROUPED_ALL_REDUCE = False

DATA_READER_TYPE = hugectr.DataReaderType_t.RawAsync
ASYNC_PARAM = hugectr.AsyncParam(32, 4, 552960, 2, 512, True, hugectr.Alignment_t.Auto)
EMBEDDING_TYPE = hugectr.Embedding_t.HybridSparseEmbedding
HYBRID_EMBEDDING_PARAM =  hugectr.HybridEmbeddingParam(2, -1, 0.02, 1.3e11, 1.9e11, 1.0, False, False,
                                                       hugectr.CommunicationType.NVLink_SingleNode,
                                                       hugectr.HybridEmbeddingType.Distributed)
# 1. Create Solver, DataReaderParams and Optimizer
solver = hugectr.CreateSolver(max_eval_batches = 300,
                              batchsize_eval = 16384,
                              batchsize = 16384,
                              vvgpu = [[0,1,2,3,4,5,6,7]],
                              repeat_dataset = True,
                              lr = 0.5,
                              warmup_steps = 300,
                              use_mixed_precision = True,
                              scaler = 1024,
                              use_cuda_graph = USE_CUDA_GRAPH,
                              async_mlp_wgrad = ASYNC_MLP_WGRAD,
                              gen_loss_summary = GEN_LOSS_SUMMARY,
                              overlap_lr = OVERLAP_LR,
                              overlap_init_wgrad = OVERLAP_INIT_WGRAD,
                              overlap_ar_a2a = OVERLAP_AR_A2A,
                              use_holistic_cuda_graph = USE_HOLISTIC_CUDA_GRAPH,
                              use_overlapped_pipeline = USE_OVERLAPPED_PIPELINE,
                              all_reduce_algo = ALL_REDUCE_ALGO,
                              grouped_all_reduce = GROUPED_ALL_REDUCE,
                              num_iterations_statistics = 20,
                              metrics_spec = {hugectr.MetricsType.AUC: 0.8025},
                              is_dlrm = True)
reader = hugectr.DataReaderParams(data_reader_type = DATA_READER_TYPE,
                                  source = [SOURCE],
                                  eval_source = EVAL_SOURCE,
                                  check_type = hugectr.Check_t.Non,
                                  num_samples = NUM_SAMPLES,
                                  eval_num_samples = EVAL_NUM_SAMPLES,
                                  cache_eval_data = 51,
                                  slot_size_array = SLOT_SIZE_ARRAY,
                                  async_param = ASYNC_PARAM)
optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.SGD,
                                    update_type = hugectr.Update_t.Local,
                                    atomic_update = True)
# 2. Initialize the Model instance
model = hugectr.Model(solver, reader, optimizer)
# 3. Construct the Model graph
model.add(hugectr.Input(label_dim = LABEL_dIM, label_name = "label",
                        dense_dim = DENSE_DIM, dense_name = "dense",
                        data_reader_sparse_param_array =
                        [hugectr.DataReaderSparseParam("data1", NNZ_ARRAY, True, NUM_SLOT)]))
model.add(hugectr.SparseEmbedding(embedding_type = EMBEDDING_TYPE,
                            workspace_size_per_gpu_in_mb = 15000,
                            slot_size_array = SLOT_SIZE_ARRAY,
                            embedding_vec_size = 128,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding1",
                            bottom_name = "data1",
                            optimizer = optimizer,
                            hybrid_embedding_param = HYBRID_EMBEDDING_PARAM))
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
                            bottom_names = ["fc3", "sparse_embedding1"],
                            top_names = ["interaction1"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.MultiCross,
                            bottom_names = ["interaction1"],
                            top_names = ["multicross1"],
                            num_layers=6))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["multicross1"],
                            top_names = ["fc4"],
                            num_output=1))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["fc4", "label"],
                            top_names = ["loss"]))
# 4. Dump the Model graph to JSON
model.graph_to_json(graph_config_file = "model.json")
# 5. Compile & Fit
model.compile()
model.summary()
model.fit(max_iter = 2500, display = 200, eval_interval = 1000, snapshot = 2000000, snapshot_prefix = "model")
