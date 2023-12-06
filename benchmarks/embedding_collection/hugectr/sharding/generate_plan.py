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
from itertools import chain, product
from typing import List
import numpy as np
from .planner import CostModel, Planner


def get_byte_per_elem(args):
    if args.optimizer == "adagrad":
        byte_per_elem = 8
    elif args.optimizer == "sgd":
        byte_per_elem = 4
    else:
        raise Exception("not supported optimizer")
    return byte_per_elem


def int_to_string(
    shard_matrix_int, shard_strategy_int, unique_table_ids_int, reduction_table_ids_int
):
    shard_strategy, shard_matrix, unique_table_ids, reduction_table_ids = [], [], [], []
    for pair in shard_strategy_int:
        if len(pair[1]) != 0:
            tmp_table_ids = []
            for i in range(len(pair[1])):
                tmp_table_info = pair[1][i]
                if isinstance(tmp_table_info, tuple):
                    tmp_table_ids.append((str(tmp_table_info[0]), tmp_table_info[1]))
                else:
                    tmp_table_ids.append(str(tmp_table_info))

            shard_strategy.append((pair[0], tmp_table_ids))
    for sub_matrix_ in shard_matrix_int:
        shard_matrix.append([str(x) for x in sub_matrix_])
    for table_id in unique_table_ids_int:
        unique_table_ids.append(str(table_id))
    for table_id in reduction_table_ids_int:
        reduction_table_ids.append(str(table_id))
    return shard_matrix, shard_strategy, unique_table_ids, reduction_table_ids


def generate_plan_ragged_ev_size(
    table_id_list: List[int],
    slot_size_array: List[int],
    multi_hot_sizes: List[int],
    ev_size_list: List[int],
    sharding_plan: str,
    num_nodes: int,
    num_gpus_per_node: int,
    args: Namespace,
    log_result: bool,
):
    num_gpus = num_nodes * num_gpus_per_node
    assert len(table_id_list) == len(slot_size_array)
    assert len(table_id_list) == len(multi_hot_sizes)
    assert len(table_id_list) == len(ev_size_list)

    def sanity_check(shard_matrix, shard_strategy):
        # mainly to make sure all the tables are sharded
        msg = "Not all tables covered in the sharding plan"
        assert set(chain(*shard_matrix)) == set(range(len(slot_size_array))), msg
        shard_strategy_list_raw = [x for strategy_pair in shard_strategy for x in strategy_pair[1]]
        shard_strategy_list = []
        for i in range(len(shard_strategy_list_raw)):
            if isinstance(shard_strategy_list_raw[i], tuple):
                shard_strategy_list.append(shard_strategy_list_raw[i][0])
            else:
                shard_strategy_list.append(shard_strategy_list_raw[i])
        assert set(shard_strategy_list) == set(range(len(slot_size_array))), msg

    def replace_table_id(shard_matrix, shard_strategy):
        shard_matrix_after_replacement = [[] for _ in range(num_gpus)]
        shard_strategy_after_replacement = []
        for gpu_id in range(num_gpus):
            for i in range(len(shard_matrix[gpu_id])):
                virtual_table_id = shard_matrix[gpu_id][i]
                shard_matrix_after_replacement[gpu_id].append(table_id_list[virtual_table_id])
        for strategy, table_ids in shard_strategy:
            table_ids_after_replacement = []
            for table_id in table_ids:
                if isinstance(table_id, tuple):
                    table_id_replacement = (table_id_list[table_id[0]], table_id[1])
                    table_ids_after_replacement.append(table_id_replacement)
                else:
                    table_ids_after_replacement.append(table_id_list[table_id])
            shard_strategy_after_replacement.append((strategy, table_ids_after_replacement))
        return shard_matrix_after_replacement, shard_strategy_after_replacement

    byte_per_elem = get_byte_per_elem(args)
    if sharding_plan in ["round_robin", "uniform", "table_row_wise"]:
        # sharding strategies that don't exploit system configs
        mp_table = [i for i in range(len(slot_size_array))]
        if sharding_plan == "round_robin":
            shard_matrix_ = [[] for _ in range(num_gpus)]

            for i, table_id in enumerate(mp_table):
                target_gpu = i % num_gpus
                shard_matrix_[target_gpu].append(table_id)

        elif sharding_plan == "uniform":
            shard_matrix_ = [mp_table for _ in range(num_gpus)]

        elif sharding_plan == "table_row_wise":
            shard_matrix_ = [[] for _ in range(num_gpus)]

            for i, table_id in enumerate(mp_table):
                if (
                    slot_size_array[i] * ev_size_list[i] * byte_per_elem / 1000 / 1000 / 1000
                    > args.memory_cap_for_embedding * args.num_gpus_per_node
                ):
                    for gpu_id in range(num_gpus):
                        shard_matrix_[gpu_id].append(table_id)
                else:
                    target_node = i % num_nodes
                    num_local_gpus = num_gpus // num_nodes

                    for gpu_id in range(num_local_gpus):
                        target_gpu = target_node * num_local_gpus + gpu_id
                        shard_matrix_[target_gpu].append(table_id)

        shard_strategy_ = [("mp", [i for i in range(len(slot_size_array))])]
        shard_column_wise_nums_ = []

    elif sharding_plan in ["auto", "hier_auto"]:
        # sharding strategies that exploit system configs
        dram_cap = args.memory_cap_for_embedding

        if sharding_plan == "auto":
            cost_model = CostModel(
                1,
                args.mem_comm_bw_ratio,
                args.mem_comm_work_ratio,
                args.dense_comm_work_ratio,
                args.batchsize,
                np.array(ev_size_list).astype(np.float64) * byte_per_elem / 1024 / 1024 / 1024,
                np.array(ev_size_list),
                dram_cap,
                slot_size_array,
                1,
            )
            planner = Planner(
                multi_hot_sizes,
                np.array(ev_size_list),
                num_nodes,
                num_gpus_per_node,
                args.batchsize,
                False,
                cost_model,
                log_result=log_result,
                use_column_wise_sharding=args.use_column_wise_shard,
            )
            shard_strategy_, shard_matrix_, shard_column_wise_nums_ = planner.plan()

        elif sharding_plan == "hier_auto":
            if num_nodes <= 1:
                raise Exception(
                    "hier_auto plan is only applicable to configs with more than one node"
                )
            cost_model = CostModel(
                1,
                args.mem_comm_bw_ratio,
                args.mem_comm_work_ratio,
                args.dense_comm_work_ratio,
                args.batchsize,
                np.array(ev_size_list).astype(np.float64) * byte_per_elem / 1024 / 1024 / 1024,
                np.array(ev_size_list),
                dram_cap * args.num_gpus_per_node,
                slot_size_array,
                1,
            )
            planner = Planner(
                multi_hot_sizes,
                np.array(ev_size_list),
                num_nodes,
                num_gpus_per_node,
                args.batchsize,
                True,
                cost_model,
                log_result=log_result,
                use_column_wise_sharding=args.use_column_wise_shard,
            )
            shard_strategy_, shard_matrix_node_, shard_column_wise_nums_ = planner.plan()
            shard_matrix_ = []
            for node_shard_matrix in shard_matrix_node_:
                for i in range(args.num_gpus_per_node):
                    shard_matrix_.append(node_shard_matrix)
    else:
        raise Exception("unknown sharding plan")
    sanity_check(shard_matrix_, shard_strategy_)

    shard_matrix, shard_strategy = replace_table_id(shard_matrix_, shard_strategy_)

    if log_result:
        logging.info("slot_size_array:")
        logging.info(slot_size_array)
        logging.info("multi_hot_sizes:")
        logging.info(multi_hot_sizes)
        logging.info("shard_matrix:")
        logging.info(shard_matrix)
        logging.info("\n")
        logging.info("shard_strategy:")
        logging.info(shard_strategy)
        logging.info("\n")
    return shard_matrix, shard_strategy, shard_column_wise_nums_


def generate_plan(
    slot_size_array: List[int],
    multi_hot_sizes: List[int],
    ev_size_list: List[int],
    num_nodes: int,
    num_gpus_per_node: int,
    args: Namespace,
    log_result: bool,
):
    # filter:
    # 1. dp table
    # 2. dense table
    byte_per_elem = get_byte_per_elem(args)
    num_gpus = num_nodes * num_gpus_per_node

    num_table = len(slot_size_array)

    # 1. select dp tables based on sorting table_size
    def filter_dp_tables(candidate_table_ids, threshold):
        candidate_table_meta = []
        for table_id in candidate_table_ids:
            candidate_table_meta.append(
                (
                    table_id,
                    (
                        slot_size_array[table_id] * ev_size_list[table_id],
                        -1 * multi_hot_sizes[table_id],
                    ),
                )
            )
        sorted_table_meta = sorted(candidate_table_meta, key=lambda x: x[1])

        if len(sorted_table_meta) > threshold:
            sorted_table_meta = sorted_table_meta[:threshold]

        dp_table_ids = [v[0] for v in sorted_table_meta]
        rest_table_ids = [i for i in candidate_table_ids if i not in set(dp_table_ids)]
        return dp_table_ids, rest_table_ids

    dp_table_ids, rest_table_ids = filter_dp_tables(
        [i for i in range(num_table)], args.dp_threshold
    )
    dp_table_memory_per_gpu = (
        sum(slot_size_array[table_id] * ev_size_list[table_id] for table_id in dp_table_ids)
        * byte_per_elem
        / 1024
        / 1024
        / 1024
    )
    args.memory_cap_for_embedding -= dp_table_memory_per_gpu

    # 2. select dense tables based on sorting rows
    def filter_dense_tables(candidate_table_ids, threshold):
        candidate_table_meta = []
        for table_id in candidate_table_ids:
            candidate_table_meta.append(
                (
                    table_id,
                    (multi_hot_sizes[table_id], slot_size_array[table_id] * ev_size_list[table_id]),
                )
            )
        sorted_table_meta = sorted(candidate_table_meta, key=lambda x: x[1])

        if len(sorted_table_meta) > threshold:
            sorted_table_meta = sorted_table_meta[:threshold]

        dense_table_ids = [v[0] for v in sorted_table_meta]
        rest_table_ids = [i for i in candidate_table_ids if i not in set(dense_table_ids)]

        # inject COMBINERS
        has_multi_hot = False
        for dense_table_id in dense_table_ids:
            if multi_hot_sizes[dense_table_id] > 1:
                has_multi_hot = True

        args.COMBINERS = []
        for table_id in range(num_table):
            if table_id in set(dense_table_ids) and not has_multi_hot:
                args.COMBINERS.append("concat")
            else:
                args.COMBINERS.append("sum")
        return dense_table_ids, rest_table_ids

    if len(rest_table_ids) > 0:
        dense_table_ids, rest_table_ids = filter_dense_tables(rest_table_ids, args.dense_threshold)
        dense_table_memory_per_gpu = (
            sum(slot_size_array[table_id] * ev_size_list[table_id] for table_id in dense_table_ids)
            * byte_per_elem
            / 1024
            / 1024
            / 1024
            / num_gpus
        )
        args.memory_cap_for_embedding -= dense_table_memory_per_gpu
    else:
        dense_table_ids = []

    # 3. sharding on reduction-based table
    sparse_table_ids = rest_table_ids
    if len(rest_table_ids) > 0:
        sparse_slot_size_array = [slot_size_array[i] for i in rest_table_ids]
        sparse_multi_hot_sizes = [multi_hot_sizes[i] for i in rest_table_ids]
        sparse_ev_size_list = [ev_size_list[i] for i in rest_table_ids]

        (
            sparse_table_shard_matrix,
            sparse_table_shard_strategy,
            sparse_table_shard_column_wise_nums,
        ) = generate_plan_ragged_ev_size(
            sparse_table_ids,
            sparse_slot_size_array,
            sparse_multi_hot_sizes,
            sparse_ev_size_list,
            args.sharding_plan,
            num_nodes,
            num_gpus_per_node,
            args,
            log_result,
        )
        assert len(sparse_table_shard_strategy) == 1, "only mp in sparse_table_shard_strtegy"
    else:
        sparse_table_shard_matrix = [[] for _ in range(num_gpus)]
        sparse_table_shard_strategy = []
        sparse_table_shard_column_wise_nums = []

    shard_matrix = [[] for _ in range(num_gpus)]
    for gpu_id in range(num_gpus):
        shard_matrix[gpu_id] += dp_table_ids
        shard_matrix[gpu_id] += dense_table_ids
        shard_matrix[gpu_id] += sparse_table_shard_matrix[gpu_id]

    shard_strategy = [
        ("dp", dp_table_ids),
        ("mp", dense_table_ids + sparse_table_shard_strategy[0][1]),
    ]

    unique_table_ids = dense_table_ids
    reduction_table_ids = sparse_table_ids

    shard_matrix, shard_strategy, unique_table_ids, reduction_table_ids = int_to_string(
        shard_matrix, shard_strategy, unique_table_ids, reduction_table_ids
    )

    if log_result:
        logging.info("Provided system info: ")
        logging.info("num_gpu_per_nodes: %d", args.num_gpus_per_node)
        logging.info("Memory to communication BW ratio: %f", args.mem_comm_bw_ratio)
        logging.info("Memory to communication work ratio: %f", args.mem_comm_work_ratio)
        logging.info("DRAM capacity: %f GB", args.memory_cap_for_embedding)
        logging.info("shard_matrix:")
        logging.info(shard_matrix)
        logging.info("shard_strategy:")
        logging.info(shard_strategy)
        logging.info("unique_table_ids:")
        logging.info(unique_table_ids)
        logging.info("reduction_table_ids:")
        logging.info(reduction_table_ids)
        logging.info("COMBINERS:")
        logging.info(args.COMBINERS)
        logging.info("dense_table_dimension:")
        logging.info([ev_size_list[table_id] for table_id in dense_table_ids])
        logging.info("sparse_table_shard_column_wise_nums:")
        logging.info(sparse_table_shard_column_wise_nums)
        logging.info("\n")
    return shard_matrix, shard_strategy, unique_table_ids, reduction_table_ids
