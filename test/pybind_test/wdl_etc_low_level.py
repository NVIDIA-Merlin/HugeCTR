import hugectr
import sys

def embedding_training_cache_test(json_file, output_dir):
  dataset = [("file_list."+str(i)+".txt", "file_list."+str(i)+".keyset") for i in range(5)]
  solver = hugectr.CreateSolver(batchsize = 16384,
                                batchsize_eval =16384,
                                vvgpu = [[0]],
                                use_mixed_precision = False,
                                i64_input_key = False,
                                use_algorithm_search = True,
                                use_cuda_graph = True,
                                repeat_dataset = False)
  reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,
                                  source = ["./file_list.0.txt"],
                                  keyset = ["./file_list.0.keyset"],
                                  eval_source = "./file_list.5.txt",
                                  check_type = hugectr.Check_t.Sum)
  optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam)
  etc = hugectr.CreateETC(ps_types = [hugectr.TrainPSType_t.Staged, hugectr.TrainPSType_t.Staged],
                        sparse_models = [output_dir + "/wdl_0_sparse_model", output_dir + "/wdl_1_sparse_model"])
  model = hugectr.Model(solver, reader, optimizer, etc)
  model.construct_from_json(graph_config_file = json_file, include_dense_network = True)
  model.compile()
  model.summary()
  lr_sch = model.get_learning_rate_scheduler()
  data_reader_train = model.get_data_reader_train()
  data_reader_eval = model.get_data_reader_eval()
  embedding_training_cache = model.get_embedding_training_cache()
  data_reader_eval.set_source("file_list.5.txt")
  data_reader_eval_flag = True
  iteration = 0
  for file_list, keyset_file in dataset:
    data_reader_train.set_source(file_list)
    data_reader_train_flag = True
    embedding_training_cache.update(keyset_file)
    while True:
      lr = lr_sch.get_next()
      model.set_learning_rate(lr)
      data_reader_train_flag = model.train(False)
      if not data_reader_train_flag:
        break
      if iteration % 100 == 0:
        batches = 0
        while data_reader_eval_flag:
          if batches >= solver.max_eval_batches:
            break
          data_reader_eval_flag = model.eval()
          batches += 1
        if not data_reader_eval_flag:
          data_reader_eval.set_source()
          data_reader_eval_flag = True
        metrics = model.get_eval_metrics()
        print("[HUGECTR][INFO] iter: {}, metrics: {}".format(iteration, metrics))
      iteration += 1
    print("[HUGECTR][INFO] trained with data in {}".format(file_list))
  updated_model = model.get_incremental_model()
  model.save_params_to_files("wdl", iteration)

if __name__ == "__main__":
  json_file = sys.argv[1]
  output_dir = sys.argv[2]
  embedding_training_cache_test(json_file, output_dir)
