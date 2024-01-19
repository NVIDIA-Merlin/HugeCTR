# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from argparse import Namespace
from itertools import chain
from typing import List

from .planner import CostModel, Planner


def generate_plan(
    slot_size_array: List[int],
    multi_hot_sizes: List[int],
    num_nodes: int,
    num_gpus: int,
    args: Namespace,
    log_result: bool,
):
    def sanity_check(shard_matrix, shard_strategy):
        # mainly to make sure all the tables are sharded
        msg = "Not all tables covered in the sharding plan"
        assert set(chain(*shard_matrix)) == set(range(len(slot_size_array))), msg
        shard_strategy_list = [x for strategy_pair in shard_strategy for x in strategy_pair[1]]
        assert set(shard_strategy_list) == set(range(len(slot_size_array))), msg

        for table_list in shard_matrix:
            if len(table_list) == 0:
                raise Exception("Currently no empty shard list is allowed")

    def int_to_string(shard_matrix_int, shard_strategy_int):
        shard_strategy, shard_matrix = [], []
        for pair in shard_strategy_int:
            if len(pair[1]) != 0:
                shard_strategy.append((pair[0], [str(x) for x in pair[1]]))
        for sub_matrix_ in shard_matrix_int:
            shard_matrix.append([str(x) for x in sub_matrix_])
        return shard_matrix, shard_strategy

    if args.sharding_plan in ["round_robin", "uniform"]:
        # sharding strategies that don't exploit system configs
        if args.sharding_plan == "round_robin":
            mp_table = [i for i in range(len(slot_size_array))]
            shard_matrix_ = [[] for _ in range(num_gpus)]
            shard_strategy_ = [("mp", [i for i in mp_table])]

            for i, table_id in enumerate(mp_table):
                target_gpu = i % num_gpus
                shard_matrix_[target_gpu].append(table_id)

        elif args.sharding_plan == "uniform":
            shard_matrix_ = [[x for x in range(len(slot_size_array))] for _ in range(num_gpus)]
            shard_strategy_ = [("mp", [i for i in range(len(slot_size_array))])]

    elif args.sharding_plan in ["auto", "hier_auto"]:
        # sharding strategies that exploit system configs
        dram_cap = args.memory_cap_for_embedding
        if args.optimizer == "adagrad":
            byte_per_elem = 8
        elif args.optimizer == "sgd":
            byte_per_elem = 4

        if args.sharding_plan == "auto":
            cost_model = CostModel(
                1,
                args.mem_comm_bw_ratio / args.mem_comm_work_ratio,
                args.ev_size * byte_per_elem * 1e-9,
                dram_cap,
                slot_size_array,
            )
            planner = Planner(
                multi_hot_sizes,
                num_gpus,
                cost_model,
                log_result=log_result,
                dp_threshold=args.dp_sharding_threshold,
            )
            shard_strategy_, shard_matrix_ = planner.plan()

        elif args.sharding_plan == "hier_auto":
            if num_nodes <= 1:
                raise Exception(
                    "hier_auto plan is only applicable to configs with more than one node"
                )
            cost_model = CostModel(
                1,
                args.mem_comm_bw_ratio / args.mem_comm_work_ratio,
                args.ev_size * byte_per_elem * 1e-9,
                dram_cap * args.num_gpus_per_node,
                slot_size_array,
            )
            planner = Planner(
                multi_hot_sizes,
                num_nodes,
                cost_model,
                log_result=log_result,
                dp_threshold=args.dp_sharding_threshold,
            )
            shard_strategy_, shard_matrix_node_ = planner.plan()
            shard_matrix_ = []
            for node_shard_matrix in shard_matrix_node_:
                for i in range(args.num_gpus_per_node):
                    shard_matrix_.append(node_shard_matrix)
    else:
        raise Exception("unknown sharding plan")

    sanity_check(shard_matrix_, shard_strategy_)
    shard_matrix, shard_strategy = int_to_string(shard_matrix_, shard_strategy_)

    if log_result:
        logging.info("Provided system info: ")
        logging.info("num_gpu_per_nodes: %d", args.num_gpus_per_node)
        logging.info("Memory to communication BW ratio: %f", args.mem_comm_bw_ratio)
        logging.info("Memory to communication work ratio: %f", args.mem_comm_work_ratio)
        logging.info("DRAM capacity: %f GB", args.memory_cap_for_embedding)
        logging.info("shard_matrix:")
        logging.info(shard_matrix)
        logging.info("\n")

    return shard_matrix, shard_strategy
