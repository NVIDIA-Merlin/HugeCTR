from hugectr import Session, solver_parser_helper, get_learning_rate_scheduler
import sys
from mpi4py import MPI
def session_impl_test(json_file, export_path_prefix):
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
                                     repeat_dataset = True
                                    )
  lr_sch = get_learning_rate_scheduler(json_file)
  sess = Session(solver_config, json_file)
  sess.start_data_reading()
  for i in range(10000):
    lr = lr_sch.get_next()
    sess.set_learning_rate(lr)
    sess.train()
    if (i%100 == 0):
      loss = sess.get_current_loss()
      print("[HUGECTR][INFO] iter: {}; loss: {}".format(i, loss))
    if (i%1000 == 0 and i != 0):
      metrics = sess.evaluation(export_path_prefix + "prediction" + str(i), export_path_prefix + "label" + str(i))
      print("[HUGECTR][INFO] iter: {}, {}".format(i, metrics))
  return

if __name__ == "__main__":
  json_file = sys.argv[1]
  export_path_prefix = sys.argv[2]
  session_impl_test(json_file, export_path_prefix)
