import hugectr
import sys
from mpi4py import MPI

def model_oversubscriber_test(json_file, temp_dir):
  dataset = [("file_list."+str(i)+".txt", "file_list."+str(i)+".keyset") for i in range(5)]
  solver = hugectr.CreateSolver(batchsize = 16384,
                                batchsize_eval =16384,
                                vvgpu = [[0]],
                                use_mixed_precision = False,
                                i64_input_key = False,
                                use_algorithm_search = True,
                                use_cuda_graph = True,
                                repeat_dataset = False,
                                use_model_oversubscriber = True,
                                temp_embedding_dir = temp_dir)
  reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,
                                  source = ["./file_list.0.txt"],
                                  keyset = ["./file_list.0.keyset"],
                                  eval_source = "./file_list.5.txt",
                                  check_type = hugectr.Check_t.Sum)
  optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam)
  model = hugectr.Model(solver, reader, optimizer)
  model.construct_from_json(graph_config_file = json_file, include_dense_network = True)
  model.compile()
  model.summary()
  lr_sch = model.get_learning_rate_scheduler()
  data_reader_train = model.get_data_reader_train()
  data_reader_eval = model.get_data_reader_eval()
  model_oversubscriber = model.get_model_oversubscriber()
  data_reader_eval.set_source("file_list.5.txt")
  iteration = 0
  for file_list, keyset_file in dataset:
    data_reader_train.set_source(file_list)
    model_oversubscriber.update(keyset_file)
    while True:
      lr = lr_sch.get_next()
      model.set_learning_rate(lr)
      model.train()
      if data_reader_train.is_eof():
        break
      if iteration % 100 == 0:
        batches = 0
        while True:
          batches += 1
          model.eval()
          if batches >= solver.max_eval_batches or data_reader_eval.is_eof():
            break
        if data_reader_eval.is_eof():
          data_reader_eval.set_source()
        metrics = model.get_eval_metrics()
        print("[HUGECTR][INFO] iter: {}, metrics: {}".format(iteration, metrics))
      iteration += 1
    print("[HUGECTR][INFO] trained with data in {}".format(file_list))
  model.save_params_to_files(temp_dir, iteration)

if __name__ == "__main__":
  json_file = sys.argv[1]
  temp_dir = sys.argv[2]
  model_oversubscriber_test(json_file, temp_dir)
