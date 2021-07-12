import hugectr
import json
import sys
import argparse
import threading
from mpi4py import MPI

DATA_READER_TYPE = {"Norm": hugectr.DataReaderType_t.Norm, 
                    "Raw": hugectr.DataReaderType_t.Raw, 
                    "Parquet": hugectr.DataReaderType_t.Parquet}
CHECK_TYPE = {"Sum": hugectr.Check_t.Sum,
              "None": hugectr.Check_t.Non}
OPTIMIZER_TYPE = {"Adam": hugectr.Optimizer_t.Adam,
                  "MomentumSGD": hugectr.Optimizer_t.MomentumSGD,
                  "Nesterov": hugectr.Optimizer_t.Nesterov,
                  "SGD": hugectr.Optimizer_t.SGD}
UPDATE_TYPE = {"Global": hugectr.Update_t.Global,
               "LazyGlobal": hugectr.Update_t.LazyGlobal,
               "Local": hugectr.Update_t.Local}

def parse_args(parser):
  args = parser.parse_args()
  json_config = json.load(open(args.json_file, "rb"))
  solver_config = json_config['solver']
  optimizer_config = json_config['optimizer']
  data_config = json_config['layers'][0]
  args.source = data_config['source']
  args.eval_source = data_config['eval_source']
  if 'format' not in data_config:
    args.data_reader_type = hugectr.DataReaderType_t.Norm
  else:
    args.data_reader_type = DATA_READER_TYPE[data_config.get('format', 'Norm')]
  args.check_type = CHECK_TYPE[data_config['check']]
  args.cache_eval_data = data_config.get('cache_eval_data', 0)
  args.num_samples = data_config.get('num_samples', 0)
  args.eval_num_samples = data_config.get('eval_num_samples', 0)
  args.float_label_dense = data_config.get('float_label_dense', False)
  args.num_workers = data_config.get('num_workers', 16)
  args.slot_size_array = data_config.get('slot_size_array', [])
  args.optimizer_type = OPTIMIZER_TYPE[optimizer_config["type"]]
  args.update_type = UPDATE_TYPE[optimizer_config['update_type']]
  args.learning_rate = 0.001
  args.beta1 = 0.9
  args.beta2 = 0.999
  args.epsilon = 0.0000001
  args.initial_accu_value = 0.0
  args.momentum_factor = 0.0
  args.atomic_update = True
  args.warmup_steps = 1
  args.decay_start = 0
  args.decay_steps = 1
  args.decay_power = 2.0
  args.end_lr = 0.0
  if 'adam_hparam' in optimizer_config:
    args.learning_rate = optimizer_config['adam_hparam']['learning_rate']
    args.beta1 = optimizer_config['adam_hparam']['beta1']
    args.beta2 = optimizer_config['adam_hparam']['beta2']
    args.epsilon = optimizer_config['adam_hparam']['epsilon']
  if 'adagrad_hparam' in optimizer_config:
    args.initial_accu_value = optimizer_config['adagrad_hparam']['initial_accu_value']
    args.epsilon = optimizer_config['adagrad_hparam']['epsilon']
  if 'momentum_sgd_hparam' in optimizer_config:
    args.learning_rate = optimizer_config['momentum_sgd_hparam']['learning_rate']
    args.momentum_factor = optimizer_config['momentum_sgd_hparam']['momentum_factor']
  if 'nesterov_hparam' in optimizer_config:
    args.learning_rate = optimizer_config['nesterov_hparam']['learning_rate']
    args.momentum_factor = optimizer_config['nesterov_hparam']['momentum_factor']
  if 'sgd_hparam' in optimizer_config:
    args.learning_rate = optimizer_config['sgd_hparam']['learning_rate']
    args.warmup_steps = optimizer_config['sgd_hparam'].get('warmup_steps', 1)
    args.decay_start = optimizer_config['sgd_hparam'].get('decay_start', 0)
    args.decay_steps = optimizer_config['sgd_hparam'].get('decay_steps', 1)
    args.decay_power = optimizer_config['sgd_hparam'].get('decay_power', 2.0)
    args.end_lr = optimizer_config['sgd_hparam'].get('end_lr', 0)
  args.batchsize = solver_config.get('batchsize', 2048)
  args.batchsize_eval = solver_config.get('batchsize_eval', args.batchsize)
  args.max_eval_batches = solver_config.get('max_eval_batches', 100)
  args.max_iter = solver_config.get('max_iter', 10000)
  args.eval_interval = solver_config.get('eval_interval', 1000)
  args.display = solver_config.get('display', 200)
  vvgpu = solver_config['gpu']
  if isinstance(vvgpu[0], list):
    args.vvgpu = vvgpu
  else:
    args.vvgpu = [vvgpu]
  args.use_mixed_precision = False
  args.scaler = 1.0
  if 'mixed_precision' in solver_config:
    args.use_mixed_precision = True
    args.scaler = solver_config['mixed_precision']
  args.i64_input_key = False
  if 'input_key_type' in solver_config and solver_config['input_key_type'] == 'I64':
    args.i64_input_key = True
  if 'auc_threshold' in solver_config:
    args.auc_threshold = solver_config['auc_threshold']
    args.auc_check = True
  else:
    args.auc_threshold = 0.5
    args.auc_check = False
  return args

def train(model, max_iter, display, max_eval_batches, eval_interval, auc_threshold):
  model.start_data_reading()
  lr_sch = model.get_learning_rate_scheduler()
  reach_auc_threshold = False
  for iter in range(max_iter):
    lr = lr_sch.get_next()
    model.set_learning_rate(lr)
    model.train()
    if (iter%display == 0):
      loss = model.get_current_loss()
      if rank == 0:
        print("[HUGECTR][INFO] iter: {}; loss: {}".format(iter, loss))
    if (iter%eval_interval == 0 and iter != 0):
      for _ in range(max_eval_batches):
        model.eval()
      metrics = model.get_eval_metrics()
      print("[HUGECTR][INFO] iter: {}, metrics: {}, rank: {} ".format(iter, metrics, rank))
      if metrics[0][1] > auc_threshold:
        reach_auc_threshold = True
        break
  if reach_auc_threshold == False:
    raise RuntimeError("Cannot reach the AUC threshold {}".format(auc_threshold))
    sys.exit(1)
  else:
    print("Successfully reach the AUC threshold {}".format(auc_threshold))
  

def multi_node_test(args):
  solver = hugectr.CreateSolver(max_eval_batches = args.max_eval_batches,
                                batchsize_eval = args.batchsize_eval,
                                batchsize = args.batchsize,
                                vvgpu = args.vvgpu,
                                lr = args.learning_rate,
                                warmup_steps = args.warmup_steps,
                                decay_start = args.decay_start,
                                decay_steps = args.decay_steps,
                                decay_power = args.decay_power,
                                end_lr = args.end_lr,
                                i64_input_key = args.i64_input_key,
                                use_mixed_precision = args.use_mixed_precision,
                                scaler = args.scaler)
  reader = hugectr.DataReaderParams(data_reader_type =  args.data_reader_type,
                                    source = [args.source],
                                    eval_source = args.eval_source,
                                    check_type = args.check_type,
                                    cache_eval_data = args.cache_eval_data,
                                    num_samples = args.num_samples,
                                    eval_num_samples = args.eval_num_samples,
                                    float_label_dense = args.float_label_dense,
                                    num_workers = args.num_workers,
                                    slot_size_array = args.slot_size_array)
  optimizer = hugectr.CreateOptimizer(optimizer_type = args.optimizer_type,
                                      beta1 = args.beta1,
                                      beta2 = args.beta2,
                                      epsilon = args.epsilon,
                                      update_type = args.update_type,
                                      momentum_factor = args.momentum_factor,
                                      atomic_update = args.atomic_update)
  model = hugectr.Model(solver, reader, optimizer)
  model.construct_from_json(graph_config_file = args.json_file, include_dense_network = True)
  model.compile()
  model.summary()
  if args.auc_check:
    train(model, args.max_iter, args.display, args.max_eval_batches, args.eval_interval, args.auc_threshold)
  else:
    model.fit(max_iter = args.max_iter, display = args.display, eval_interval = args.eval_interval)
  return

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--json-file', type=str, required = True, help='JSON configuration file')
  args = parse_args(parser)
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  thread = threading.Thread(target=multi_node_test, args = (args,), name='[rank-%d train]' % rank)
  current_thread = threading.currentThread()
  print('[HUGECTR][INFO] before: rank %d '% (rank))
  # start the thread
  thread.start()
  # wait for terminate
  thread.join()
  print('[HUGECTR][INFO] after: rank %d ' % (rank))

