from hugectr import Session, solver_parser_helper, get_learning_rate_scheduler
from mpi4py import MPI
import threading
import sys

def session_impl_test(json_file):
  solver_config = solver_parser_helper(seed = 0,
                                     batchsize = 16384,
                                     batchsize_eval = 16384,
                                     model_file = "",
                                     embedding_files = [],
                                     vvgpu = [[0,1],[2,3],[4,5],[6,7]],
                                     use_mixed_precision = False,
                                     scaler = 1.0,
                                     i64_input_key = False,
                                     use_algorithm_search = True,
                                     use_cuda_graph = True,
                                     repeat_dataset = True
                                    )
  sess = Session(solver_config, json_file)
  sess.start_data_reading()
  lr_sch = get_learning_rate_scheduler(json_file)
  for i in range(10000):
    lr = lr_sch.get_next()
    sess.set_learning_rate(lr)
    sess.train()
    if (i%100 == 0):
      loss = sess.get_current_loss()
      if (rank == 0):
        print("[HUGECTR][INFO] iter: {}; loss: {}".format(i, loss))
    if (i%1000 == 0 and i != 0):
      sess.check_overflow()
      sess.copy_weights_for_evaluation()
      data_reader_eval = sess.get_data_reader_eval()
      for _ in range(solver_config.max_eval_batches):
        sess.eval()
      metrics = sess.get_eval_metrics()
      print("[HUGECTR][INFO] rank: {}, iter: {}, {}".format(rank, i, metrics))
  return

if __name__ == "__main__":
  json_file = sys.argv[1]
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  thread = threading.Thread(target=session_impl_test, args = (json_file,), name='[rank-%d train]' % rank)
  current_thread = threading.currentThread()
  print('[HUGECTR][INFO] %s is main thread: %s' % (current_thread.name, MPI.Is_thread_main()))
  print('[HUGECTR][INFO] before: rank %d '% (rank))
  # start the thread
  thread.start()
  # wait for terminate
  thread.join()
  print('[HUGECTR][INFO] after: rank %d ' % (rank))

