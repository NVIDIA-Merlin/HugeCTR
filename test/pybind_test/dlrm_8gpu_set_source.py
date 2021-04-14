import hugectr
import sys
from mpi4py import MPI
def set_source_raw_test(json_file):
  train_data = "./train_data.bin"
  test_data = "./test_data.bin"
  solver = hugectr.CreateSolver(max_eval_batches = 5441,
                                batchsize_eval = 16384,
                                batchsize = 16384,
                                vvgpu = [[0,1,2,3,4,5,6,7]],
                                lr = 24.0,
                                warmup_steps = 8000,
                                decay_start = 480000000,
                                decay_steps = 240000000,
                                decay_power = 2.0,
                                end_lr = 0,
                                i64_input_key = False,
                                use_mixed_precision = True,
                                scaler = 1024,
                                repeat_dataset = False,
                                use_cuda_graph = True)
  reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Raw,
                                    source = [train_data],
                                    eval_source = test_data,
                                    check_type = hugectr.Check_t.Non,
                                    slot_size_array = [39884406,    39043,    17289,     7420,    20263,    3,  7120,     1543,  63, 38532951,  2953546,   403346, 10,     2208,    11938,      155,        4,      976, 14, 39979771, 25641295, 39664984,   585935,    12972,  108,  36],
                                    num_samples = 4195155968,
                                    eval_num_samples = 89137319,
                                    cache_eval_data = 1361)
  optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.SGD,
                                      atomic_update = True)
  model = hugectr.Model(solver, reader, optimizer)
  model.construct_from_json(graph_config_file = json_file, include_dense_network = True)
  model.compile()
  model.summary()
  lr_sch = model.get_learning_rate_scheduler()
  data_reader_train = model.get_data_reader_train()
  data_reader_eval = model.get_data_reader_eval()
  data_reader_eval.set_source(test_data)
  iteration = 1
  for cnt in range(2):
    data_reader_train.set_source(train_data)
    print("[HUGECTR][INFO] round: {}".format(cnt), flush = True)
    while True:
      lr = lr_sch.get_next()
      model.set_learning_rate(lr)
      model.train()
      if data_reader_train.is_eof():
        break
      if iteration % 4000 == 0:
        j = 0
        while data_reader_eval.is_eof() == False:
          if j >= solver.max_eval_batches:
            break
          model.eval()
          j += 1
        if data_reader_eval.is_eof():
          data_reader_eval.set_source()
        metrics = model.get_eval_metrics()
        print("[HUGECTR][INFO] iter: {}, metrics: {}".format(iteration, metrics), flush = True)
      iteration += 1
    print("[HUGECTR][INFO] trained with data in {}".format(train_data), flush = True)

if __name__ == "__main__":
  json_file = sys.argv[1]
  set_source_raw_test(json_file)
