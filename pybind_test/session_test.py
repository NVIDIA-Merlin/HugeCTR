from hugectr import get_learning_rate_scheduler, SolverParser, Session, solver_parser_helper, LrPolicy_t, MetricsType

def session_impl_test():
  json_name = "./criteo_data/criteo_bin.json"
  solver_config = solver_parser_helper(batchsize = 512, batchsize_eval = 512, lr_policy = LrPolicy_t.fixed, vvgpu = [[0]], i64_input_key = False, metrics_spec = {MetricsType.AUC: 0.8025, MetricsType.AverageLoss: 0.005})
  session_instance = Session.Create(solver_config, json_name)
  for i in range(10000):
    session_instance.train()
    if (i%10 == 0):
      loss = session_instance.get_current_loss()
      print("iter: {}; loss: {}".format(i, loss))
  session_instance.eval()
  metrics = session_instance.get_eval_metrics()
  print(metrics)

def learning_rate_scheduler_test(config_file):
  lr_sch = get_learning_rate_scheduler(config_file)
  for i in range(10000):
    lr = lr_sch.get_next()
    print("iter: {}, lr: {}".format(i, lr))

if __name__ == "__main__":
  session_impl_test()
  #learning_rate_scheduler_test("wdl_1gpu.json")
