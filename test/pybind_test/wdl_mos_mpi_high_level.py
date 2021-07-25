import hugectr
from mpi4py import MPI
import threading
import sys

def model_oversubscriber_test(json_file, output_dir, use_host_ps):
  dataset = [("file_list."+str(i)+".txt", "file_list."+str(i)+".keyset") for i in range(5)]
  solver = hugectr.CreateSolver(batchsize = 16384,
                                batchsize_eval = 16384,
                                vvgpu = [[0, 1, 2, 3], [4, 5, 6, 7]],
                                use_mixed_precision = False,
                                i64_input_key = False,
                                use_algorithm_search = True,
                                use_cuda_graph = True,
                                repeat_dataset = False)
  reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,
                                  source = ["file_list."+str(i)+".txt" for i in range(5)],
                                  keyset = ["file_list."+str(i)+".keyset" for i in range(5)],
                                  eval_source = "./file_list.5.txt",
                                  check_type = hugectr.Check_t.Sum)
  optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam)
  mos = hugectr.CreateMOS(train_from_scratch = False, use_host_memory_ps=use_host_ps, trained_sparse_models = [output_dir + "/wdl_0_sparse_model", output_dir + "/wdl_1_sparse_model"])
  model = hugectr.Model(solver, reader, optimizer, mos)
  model.construct_from_json(graph_config_file = json_file, include_dense_network = True)
  model.compile()
  model.summary()
  model.fit(num_epochs=1, eval_interval=1000000, display =200)
  if use_host_ps:
    updated_model = model.get_incremental_model()
  model.save_params_to_files("wdl")
  model.set_source(source = ["file_list."+str(i)+".txt" for i in range(6,9)],
                   keyset = ["file_list."+str(i)+".keyset" for i in range(6,9)], 
                   eval_source = "./file_list.5.txt")
  model.fit(num_epochs=1, eval_interval=1000000, display =200)
  if use_host_ps:
    updated_model = model.get_incremental_model()
  model.save_params_to_files("wdl")

if __name__ == "__main__":
  json_file = sys.argv[1]
  use_host_ps = sys.argv[2].lower() == 'true'
  output_dir = sys.argv[3]
  model_oversubscriber_test(json_file, output_dir, use_host_ps)