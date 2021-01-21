from hugectr import Session, solver_parser_helper, get_learning_rate_scheduler
import sys
from mpi4py import MPI
def set_source_raw_test(json_file):
  train_data = "./train_data.bin"
  test_data = "./test_data.bin"
  solver_config = solver_parser_helper(seed = 0,
                                     batchsize = 16384,
                                     batchsize_eval = 16384,
                                     model_file = "",
                                     embedding_files = [],
                                     vvgpu = [[0,1,2,3,4,5,6,7]],
                                     use_mixed_precision = True,
                                     scaler = 1024,
                                     i64_input_key = False,
                                     use_algorithm_search = True,
                                     use_cuda_graph = True,
                                     repeat_dataset = False
                                    )
  lr_sch = get_learning_rate_scheduler(json_file)
  sess = Session(solver_config, json_file)
  data_reader_train = sess.get_data_reader_train()
  data_reader_eval = sess.get_data_reader_eval()
  data_reader_eval.set_source(test_data)
  iteration = 1
  for cnt in range(2):
    data_reader_train.set_source(train_data)
    print("[HUGECTR][INFO] round: {}".format(cnt))
    while True:
      lr = lr_sch.get_next()
      sess.set_learning_rate(lr)
      good = sess.train()
      if good == False:
        break
      if iteration % 4000 == 0:
        sess.check_overflow()
        sess.copy_weights_for_evaluation()
        data_reader_eval = sess.get_data_reader_eval()
        good_eval = True
        j = 0
        while good_eval:
          if j >= solver_config.max_eval_batches:
            break
          good_eval = sess.eval()
          j += 1
        if good_eval == False:
          data_reader_eval.set_source()
        metrics = sess.get_eval_metrics()
        print("[HUGECTR][INFO] iter: {}, metrics: {}".format(iteration, metrics))
      iteration += 1
    print("[HUGECTR][INFO] trained with data in {}".format(train_data))

if __name__ == "__main__":
  json_file = sys.argv[1]
  set_source_raw_test(json_file)
