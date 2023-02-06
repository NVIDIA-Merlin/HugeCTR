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
import cudf
from mpi4py import MPI
import pandas as pd
import math
import pdb
import cupy
import json
import os


class dcn_model_reader:
    def __init__(self, gpus: list, batchsize: int, file_list: str) -> None:
        self.solver = hugectr.CreateSolver(
            max_eval_batches=300,
            batchsize_eval=batchsize,
            batchsize=batchsize,
            lr=0.001,
            perf_logging=True,
            vvgpu=[gpus],
            repeat_dataset=True,
            i64_input_key=True,
        )
        self.reader = hugectr.DataReaderParams(
            data_reader_type=hugectr.DataReaderType_t.Parquet,
            source=[file_list],
            # read_file_sequentially = True,
            eval_source=file_list,
            slot_size_array=[
                203931,
                18598,
                14092,
                7012,
                18977,
                4,
                6385,
                1245,
                49,
                186213,
                71328,
                67288,
                11,
                2168,
                7338,
                61,
                4,
                932,
                15,
                204515,
                141526,
                199433,
                60919,
                9137,
                71,
                34,
            ],
            check_type=hugectr.Check_t.Non,
        )
        self.optimizer = hugectr.CreateOptimizer(
            optimizer_type=hugectr.Optimizer_t.Adam,
            update_type=hugectr.Update_t.Global,
            beta1=0.9,
            beta2=0.999,
            epsilon=0.0000001,
        )
        self.model = hugectr.Model(self.solver, self.reader, self.optimizer)
        self.model.add(
            hugectr.Input(
                label_dim=1,
                label_name="label",
                dense_dim=13,
                dense_name="dense",
                data_reader_sparse_param_array=[
                    hugectr.DataReaderSparseParam("data1", 1, False, 26)
                ],
            )
        )
        self.model_reader = self.model.get_data_reader_train()
        self.iter = 0

    def read_a_batch(self):
        self.model_reader.read_a_batch_to_device()
        self.iter = self.iter + 1
        return self.model.check_out_tensor("dense", hugectr.Tensor_t.Train)

    # run cudf to see if environment is broken by hctr
    def get_random_df(self):
        some_dict = {
            "_col1": [1, 3, 4, 5, 6, 7, 8],
            "_col2": [1, 3, 4, 5, 6, 0, 8],
        }
        return cudf.DataFrame(some_dict)

    def __del__(self):
        some_dict = {
            "_col1": [1, 3, 4, 5, 6, 7, 8],
            "_col2": [1, 3, 4, 5, 6, 0, 8],
        }
        df = cudf.DataFrame(some_dict)


if __name__ == "__main__":
    print("ENTRANCE: gpu device is ", cupy.cuda.runtime.getDevice())

    with open("./train/_metadata.json") as json_data:
        data = json.load(json_data)
    file_stats = data["file_stats"]
    file_rows = {}
    for mp in file_stats:
        file_rows[mp["file_name"]] = mp["num_rows"]
        # for key,val in file_stats
    gpus = [0, 1]
    batchsize = 32768

    row_start = [0] * len(gpus)
    file_ids = [i for i in range(len(gpus))]

    iters = 10
    dense_col_names = []

    for i in range(1, 14):
        dense_col_names.append("I" + str(i))
    # one file comprise 13056143 samples
    file_list_path = "./train/_file_list.txt"

    model = dcn_model_reader(gpus, batchsize, file_list_path)
    dfs = []
    num_files = -1
    file_list = []
    with open(file_list_path, "r") as f:
        file_contents = f.read().splitlines()
        num_files = int(file_contents[0])
        file_list = file_contents[1 : num_files + 1]
        for i, file in enumerate(file_list):
            dfs.append(pd.read_parquet(file, columns=dense_col_names))

    for i in range(iters):
        gpu_id = i % len(gpus)
        file_id = file_ids[gpu_id]
        file_name = file_list[file_id]

        ndarry_dense = model.read_a_batch()
        print(i, " iter")
        dense_read_df = pd.DataFrame(ndarry_dense, columns=dense_col_names)
        dense_df = dfs[file_id][dense_col_names].iloc[
            row_start[gpu_id] : row_start[gpu_id] + batchsize
        ]
        row_start[gpu_id] = row_start[gpu_id] + batchsize
        dense_df.index = dense_read_df.index
        if row_start[gpu_id] > file_rows[os.path.basename(file_name)]:
            file_ids[gpu_id] = (file_ids[gpu_id] + len(gpus)) % len(gpus)
            row_start[gpu_id] = 0

        sum_batch = (dense_df - dense_read_df).sum(axis=0).sum()
        if abs(sum_batch) > 1e-6:
            print(i, "batchsize is ", batchsize, " err is :", sum_batch)
            exit(-1)

    model.get_random_df()
    del model

print("EXIT: gpu device is ", cupy.cuda.runtime.getDevice())
