from hugectr import get_learning_rate_scheduler, SolverParser, Session, solver_parser_helper, LrPolicy_t, MetricsType
import hugectr
from mpi4py import MPI
import threading

def session_impl_test():
  json_name = "multi_node_data/criteo_2node_4gpu.json"
  solver_config = solver_parser_helper(batchsize = 40960, batchsize_eval = 40960, lr_policy = LrPolicy_t.fixed, vvgpu = [[0],[1]], i64_input_key = False, metrics_spec = {MetricsType.AUC: 0.8025, MetricsType.AverageLoss: 0.005})
  sess = Session.Create(solver_config, json_name)
  for i in range(10000):
    sess.train()
    if (i%100 == 0):
      loss = sess.get_current_loss()
      if (rank == 0):
        print("iter: {}; loss: {}".format(i, loss))
    if (i%1000 == 0 and i != 0):
      sess.check_overflow()
      metrics = sess.evaluation()
      print("rank: {}, iter: {}, {}".format(rank, i, metrics))
  return

if __name__ == "__main__":
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  thread = threading.Thread(target=session_impl_test, name='[rank-%d train]' % rank)
  current_thread = threading.currentThread()
  print('%s is main thread: %s' % (current_thread.name, MPI.Is_thread_main()))
  print('before: rank %d '% (rank))
  # start the thread
  thread.start()
  # wait for terminate
  thread.join()
  print('after: rank %d ' % (rank))

