from hugectr import Session, solver_parser_helper, SolverParser
from mpi4py import MPI
import threading
import json
import sys

def session_impl_test(json_file):
  #with open(json_file, 'r') as j_file:
  #  j = json.load(j_file)
  #solver = j['solver']
  #batchsize_ = solver.get('batchsize', 40960)
  #batchsize_eval_ = solver.get('batchsize_eval', batchsize_)
  #vvgpu_ = solver.get('gpu', [[0]])
  #if not isinstance(vvgpu_[0], list):
  #  vvgpu_ = [vvgpu_]
  #input_key_type_ = solver.get('input_key_type', 'I32')
  #i64_input_key_ = (input_key_type_ == 'I64')
  #use_mixed_precision_ =  ('mixed_precsion' in solver)
  #scaler_ = solver.get('mixed_precsion', 1.0)
  #use_cuda_graph_ = solver.get('cuda_graph', True)
  #print("batchsize: {}, batchsize_eval: {}, vvgpu: {}, i64_input_key: {}, use_mixed_precison: {}, use_cuda_graph: {}".format(batchsize_, batchsize_eval_, vvgpu_, i64_input_key_, use_mixed_precision_, use_cuda_graph_))
  #solver_config = solver_parser_helper(batchsize = batchsize_, batchsize_eval = batchsize_eval_, vvgpu = vvgpu_, i64_input_key = i64_input_key_, use_mixed_precision = use_mixed_precision_, scaler = scaler_, use_cuda_graph = use_cuda_graph_)
  solver_config = SolverParser(json_file)
  sess = Session(solver_config, json_file)
  for i in range(10000):
    sess.train()
    if (i%100 == 0):
      loss = sess.get_current_loss()
      if (rank == 0):
        print("iter: {}; loss: {}".format(i, loss))
    if (i%1000 == 0 and i != 0):
      metrics = sess.evaluation()
      print("rank: {}, iter: {}, {}".format(rank, i, metrics))
  return

if __name__ == "__main__":
  json_file = sys.argv[1]
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  thread = threading.Thread(target=session_impl_test, args = (json_file,), name='[rank-%d train]' % rank)
  current_thread = threading.currentThread()
  print('%s is main thread: %s' % (current_thread.name, MPI.Is_thread_main()))
  print('before: rank %d '% (rank))
  # start the thread
  thread.start()
  # wait for terminate
  thread.join()
  print('after: rank %d ' % (rank))

