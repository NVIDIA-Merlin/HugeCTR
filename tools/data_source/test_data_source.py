import hugectr
from mpi4py import MPI
from hugectr.data import DataSource, DataSourceParams

data_source_params = DataSourceParams(
    use_hdfs = True, #whether use HDFS to save model files
    namenode = 'localhost', #HDFS namenode IP
    port = 9000, #HDFS port
)

solver = hugectr.CreateSolver(max_eval_batches = 1280,
                              batchsize_eval = 1024,
                              batchsize = 1024,
                              lr = 0.001,
                              vvgpu = [[0]],
                              repeat_dataset = True,
                              data_source_params = data_source_params)
reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,
                                  source = ['./wdl_norm/file_list.txt'],
                                  eval_source = './wdl_norm/file_list_test.txt',
                                  check_type = hugectr.Check_t.Sum)
optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam,
                                    update_type = hugectr.Update_t.Global,
                                    beta1 = 0.9,
                                    beta2 = 0.999,
                                    epsilon = 0.0000001)
model = hugectr.Model(solver, reader, optimizer)
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 13, dense_name = "dense",
                        data_reader_sparse_param_array = 
                        # the total number of slots should be equal to data_generator_params.num_slot
                        [hugectr.DataReaderSparseParam("wide_data", 2, True, 1),
                        hugectr.DataReaderSparseParam("deep_data", 1, True, 26)]))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
                            workspace_size_per_gpu_in_mb = 69,
                            embedding_vec_size = 1,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding2",
                            bottom_name = "wide_data",
                            optimizer = optimizer))
model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
                            workspace_size_per_gpu_in_mb = 1074,
                            embedding_vec_size = 16,
                            combiner = "sum",
                            sparse_embedding_name = "sparse_embedding1",
                            bottom_name = "deep_data",
                            optimizer = optimizer))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                            bottom_names = ["sparse_embedding1"],
                            top_names = ["reshape1"],
                            leading_dim=416))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                            bottom_names = ["sparse_embedding2"],
                            top_names = ["reshape2"],
                            leading_dim=1))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,
                            bottom_names = ["reshape1", "dense"],
                            top_names = ["concat1"]))
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
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["dropout2"],
                            top_names = ["fc3"],
                            num_output=1))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Add,
                            bottom_names = ["fc3", "reshape2"],
                            top_names = ["add1"]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["add1", "label"],
                            top_names = ["loss"]))
model.compile()
model.summary()

model.load_dense_weights('/model/wdl/_dense_1000.model')
model.load_dense_optimizer_states('/model/wdl/_opt_dense_1000.model')
model.load_sparse_weights(['/model/wdl/0_sparse_1000.model', '/model/wdl/1_sparse_1000.model'])
model.load_sparse_optimizer_states(['/model/wdl/0_opt_sparse_1000.model', '/model/wdl/1_opt_sparse_1000.model'])

model.fit(max_iter = 1020, display = 200, eval_interval = 500, snapshot = 1000, snapshot_prefix = "/model/wdl/")