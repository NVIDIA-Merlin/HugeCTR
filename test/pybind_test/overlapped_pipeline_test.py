import hugectr
from hugectr.tools import DataGenerator, DataGeneratorParams

LABEL_dIM = 1
DENSE_DIM = 6
NUM_SLOT = 5
SLOT_SIZE_ARRAY = [10000, 40000, 50000, 20000, 30000]
NNZ_ARRAY = [1, 1, 1, 1, 1]
NUM_SAMPLES = 5242880
EVAL_NUM_SAMPLES = 1310720
SOURCE = "./overlapped_raw/train_data.bin"
EVAL_SOURCE = "./overlapped_raw/test_data.bin"

# # One-Hot Raw data generation (The synthetic dataset is already on Selene)
# data_generator_params = DataGeneratorParams(
#   format = hugectr.DataReaderType_t.Raw,
#   label_dim = LABEL_dIM,
#   dense_dim = DENSE_DIM,
#   num_slot = NUM_SLOT,
#   i64_input_key = False,
#   source = SOURCE,
#   eval_source = EVAL_SOURCE,
#   slot_size_array = SLOT_SIZE_ARRAY,
#   nnz_array = NNZ_ARRAY,
#   num_samples = NUM_SAMPLES,
#   eval_num_samples = EVAL_NUM_SAMPLES
# )
# data_generator = DataGenerator(data_generator_params)
# data_generator.generate()

# Train
# 1. Create Solver, DataReaderParams and Optimizer
solver = hugectr.CreateSolver(max_eval_batches = 80,
                              batchsize_eval = 16384,
                              batchsize = 16384,
                              vvgpu = [[0,1,2,3,4,5,6,7]],
                              use_mixed_precision = True,
                              scaler = 1024,
                              repeat_dataset = True,
                              use_cuda_graph = False,
                              async_mlp_wgrad = False, # no interaction layer, thus this knob needs to be turned off
                              gen_loss_summary = True,
                              overlap_lr = True,
                              overlap_init_wgrad = True,
                              overlap_ar_a2a = True,
                              use_overlapped_pipeline = True,
                              use_holistic_cuda_graph = True,
                              all_reduce_algo = hugectr.AllReduceAlgo.OneShot,
                              grouped_all_reduce = False,
                              num_iterations_statistics = 20,
                              metrics_spec = {hugectr.MetricsType.AUC: 0.8025},
                              perf_logging = True,
                              drop_incomplete_batch = False)
reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.RawAsync,
                                  source = [SOURCE],
                                  eval_source = EVAL_SOURCE,
                                  check_type = hugectr.Check_t.Non,
                                  num_samples = NUM_SAMPLES,
                                  eval_num_samples = EVAL_NUM_SAMPLES,
                                  cache_eval_data = 51,
                                  slot_size_array = SLOT_SIZE_ARRAY,
                                  async_param = hugectr.AsyncParam(32, 4, 552960, 2, 512, True, hugectr.Alignment_t.Auto))
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
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.HybridSparseEmbedding, 
                            workspace_size_per_gpu_in_mb = 15000,
                            slot_size_array =  SLOT_SIZE_ARRAY,
                            embedding_vec_size = 128,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding1",
                            bottom_name = "data1",
                            optimizer = optimizer,
                            hybrid_embedding_param = hugectr.HybridEmbeddingParam(2, -1, 0.02, 1.3e11, 1.9e11, 1.0, True, True, 
                                                                                hugectr.CommunicationType.NVLink_SingleNode,
                                                                                hugectr.HybridEmbeddingType.Distributed)))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                            bottom_names = ["sparse_embedding1"],
                            top_names = ["reshape1"],
                            leading_dim=640))
model.add(hugectr.GroupDenseLayer(group_layer_type = hugectr.GroupLayer_t.GroupFusedInnerProduct,
                            bottom_name_list = ["dense"],
                            top_name_list = ["fc1", "fc2", "fc3"],
                            num_outputs = [512, 256, 128],
                            last_act_type = hugectr.Activation_t.Relu))
model.add(hugectr.GroupDenseLayer(group_layer_type = hugectr.GroupLayer_t.GroupFusedInnerProduct,
                            bottom_name_list = ["reshape1"],
                            top_name_list = ["fc4", "fc5", "fc6"],
                            num_outputs = [512, 256, 128],
                            last_act_type = hugectr.Activation_t.Relu))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Add,
                            bottom_names = ["fc3", "fc6"], top_names = ["add"]))
model.add(hugectr.GroupDenseLayer(group_layer_type = hugectr.GroupLayer_t.GroupFusedInnerProduct,
                            bottom_name_list = ["add"],
                            top_name_list = ["fc7", "fc8", "fc9", "fc10"],
                            num_outputs = [1024, 512, 256, 1],
                            last_act_type = hugectr.Activation_t.Non))                                                 
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["fc10", "label"],
                            top_names = ["loss"]))
# 4. Dump the Model graph to JSON
model.graph_to_json(graph_config_file = "overlapped.json")
# 5. Compile & Fit
model.compile()
model.summary()
model.fit(max_iter = 601, display = 100, eval_interval = 100, snapshot = 2000000, snapshot_prefix = "overlapped")