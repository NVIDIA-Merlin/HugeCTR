"""
 Copyright (c) 2023, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import hugectr
import sys
import numpy as np


def wdl_test(json_file, export_path_prefix):
    solver = hugectr.CreateSolver(
        max_eval_batches=2048,
        batchsize_eval=16384,
        batchsize=16384,
        vvgpu=[[0, 1, 2, 3, 4, 5, 6, 7]],
        lr=0.001,
        i64_input_key=True,
        use_mixed_precision=True,
        scaler=1024,
        repeat_dataset=True,
        use_cuda_graph=True,
    )
    reader = hugectr.DataReaderParams(
        data_reader_type=hugectr.DataReaderType_t.Parquet,
        source=["./wdl_data_parquet/train/_file_list.txt"],
        eval_source="./wdl_data_parquet/val/_file_list.txt",
        check_type=hugectr.Check_t.Non,
    )
    optimizer = hugectr.CreateOptimizer(
        optimizer_type=hugectr.Optimizer_t.Adam, beta1=0.9, beta2=0.999, epsilon=0.0001
    )
    model = hugectr.Model(solver, reader, optimizer)
    model.construct_from_json(graph_config_file=json_file, include_dense_network=True)
    model.compile()
    model.summary()
    model.start_data_reading()
    lr_sch = model.get_learning_rate_scheduler()
    for i in range(10000):
        lr = lr_sch.get_next()
        model.set_learning_rate(lr)
        model.train()
        if i % 100 == 0:
            loss = model.get_current_loss()
            print("[HUGECTR][INFO] iter: {}; loss: {}".format(i, loss))
        if i % 1000 == 0 and i != 0:
            for _ in range(solver.max_eval_batches):
                model.eval()
            eval_results = model.check_out_tensor("add1", hugectr.Tensor_t.Evaluate)
            np.save(export_path_prefix + "eval" + str(i) + ".npy", eval_results)
            print("[HUGECTR][INFO] eval results shape: {}".format(eval_results.shape))
            metrics = model.get_eval_metrics()
            print("[HUGECTR][INFO] iter: {}, {}".format(i, metrics))
    return


if __name__ == "__main__":
    json_file = sys.argv[1]
    export_path_prefix = sys.argv[2]
    wdl_test(json_file, export_path_prefix)
