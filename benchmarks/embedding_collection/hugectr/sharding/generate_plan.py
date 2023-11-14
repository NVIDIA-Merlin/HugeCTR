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


def filter_sparse_dense_table(args, table_ids, slot_sizes, multi_hots, ev_sizes):
    sparse_table_ids = []
    sparse_slot_sizes = []
    sparse_multi_hots = []
    sparse_ev_sizes = []
    dense_table_ids = []
    for i in range(len(sparse_table_ids)):
        if sparse_multi_hots[i] <= args.sd_threshold:
            dense_table_ids.append(table_ids[i])
        else:
            sparse_table_ids.append(table_ids[i])
            sparse_slot_sizes.append(slot_sizes[i])
            sparse_multi_hots.append(multi_hots[i])
            sparse_ev_sizes.append(ev_sizes[i])

    return sparse_table_ids, sparse_slot_sizes, sparse_multi_hots, sparse_ev_sizes, dense_table_ids


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
        dp_threshold: int = 0,
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
                    table_id_replacement = (table_id_list[table_id], table_id[1])
                    table_ids_after_replacement.append(table_id_replacement)
                else:
                    table_ids_after_replacement.append(table_id_list[table_id])
            shard_strategy_after_replacement.append((strategy, table_ids_after_replacement))
        return shard_matrix_after_replacement, shard_strategy_after_replacement

    def int_to_string(shard_matrix_int, shard_strategy_int):
        shard_strategy, shard_matrix = [], []
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
        return shard_matrix, shard_strategy

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
        if args.optimizer == "adagrad":
            byte_per_elem = 8
        elif args.optimizer == "sgd":
            byte_per_elem = 4

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
                unique_ratio=0.4
            )
            planner = Planner(multi_hot_sizes, np.array(ev_size_list), num_nodes, num_gpus_per_node, args.batchsize,
                              False, cost_model, log_result=log_result, dp_threshold=dp_threshold,use_column_wise_sharding=args.use_column_wise_shard)
            shard_strategy_, shard_matrix_, shard_column_wise_nums_ = planner.plan()

        elif sharding_plan == "hier_auto":
            if num_nodes <= 1:
                raise Exception("hier_auto plan is only applicable to configs with more than one node")
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
                unique_ratio=0.4
            )
            planner = Planner(multi_hot_sizes, np.array(ev_size_list), num_nodes, num_gpus_per_node, args.batchsize,
                              True, cost_model, log_result=log_result, dp_threshold=dp_threshold,use_column_wise_sharding=args.use_column_wise_shard)
            shard_strategy_, shard_matrix_node_, shard_column_wise_nums_ = planner.plan()
            shard_matrix_ = []
            for node_shard_matrix in shard_matrix_node_:
                for i in range(args.num_gpus_per_node):
                    shard_matrix_.append(node_shard_matrix)
    else:
        raise Exception("unknown sharding plan")
    sanity_check(shard_matrix_, shard_strategy_)

    shard_matrix, shard_strategy = replace_table_id(shard_matrix_, shard_strategy_)
    shard_matrix, shard_strategy = int_to_string(shard_matrix, shard_strategy)

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
        combiner_list: List[str],
        num_nodes: int,
        num_gpus_per_node: int,
        args: Namespace,
        log_result: bool,
        dp_threshold: int = 0,
):
    combiner_set = sorted(list(set(combiner_list)))

    num_table = len(slot_size_array)
    assert num_table == len(multi_hot_sizes)
    assert num_table == len(ev_size_list)
    assert num_table == len(combiner_list)
    table_id_list = list(range(num_table))
    num_gpus = num_nodes * num_gpus_per_node
    shard_matrix = [[] for _ in range(num_gpus)]
    sparse_mp_shard_table_ids = []
    dense_mp_shard_table_ids = []
    dp_shard_table_ids = []
    dense_sparse_table_ids = []
    for combiner in combiner_set:
        filtered_table_id_list = []
        filtered_slot_size_array = []
        filtered_multi_hot_sizes = []
        filtered_ev_sizes = []
        for i in range(num_table):
            if combiner != combiner_list[i]:
                continue
            filtered_table_id_list.append(table_id_list[i])
            filtered_slot_size_array.append(slot_size_array[i])
            filtered_multi_hot_sizes.append(multi_hot_sizes[i])
            filtered_ev_sizes.append(ev_size_list[i])
        if len(filtered_table_id_list) == 0:
            continue

        if combiner == 'concat':
            one_shard_matrix, one_shard_strategy, one_shard_column_wise_nums = generate_plan_ragged_ev_size(
                filtered_table_id_list,
                filtered_slot_size_array,
                filtered_multi_hot_sizes,
                filtered_ev_sizes,
                "uniform",
                num_nodes,
                num_gpus_per_node,
                args,
                log_result,
                dp_threshold
            )
        elif combiner == 'sum':
            # first filter which table use dense do sparse
            filtered_table_id_list, filtered_slot_size_array, filtered_multi_hot_sizes, filtered_ev_sizes, tmp_dense_sparse_table_ids = filter_sparse_dense_table(
                args, filtered_table_id_list, filtered_slot_size_array, filtered_multi_hot_sizes, filtered_ev_sizes)
            dense_sparse_table_ids += tmp_dense_sparse_table_ids
            one_shard_matrix, one_shard_strategy, one_shard_column_wise_nums = generate_plan_ragged_ev_size(
                filtered_table_id_list,
                filtered_slot_size_array,
                filtered_multi_hot_sizes,
                filtered_ev_sizes,
                args.sharding_plan,
                num_nodes,
                num_gpus_per_node,
                args,
                log_result,
                dp_threshold
            )
            # add spase dense table id to shard_matrix
            if len(dense_sparse_table_ids) > 0:
                for gpu_id in range(num_gpus):
                    shard_matrix[gpu_id] += tmp_dense_sparse_table_ids
        else:
            raise
        for gpu_id in range(num_gpus):
            shard_matrix[gpu_id] += one_shard_matrix[gpu_id]
        sparse_mp_shard_table_ids.append(tmp_dense_sparse_table_ids)
        # add sparse dense
        for strategy, table_ids in one_shard_strategy:
            if strategy == 'dp':
                dp_shard_table_ids += table_ids
            if strategy == 'mp' and combiner == "sum":
                sparse_mp_shard_table_ids.append(table_ids)
            if strategy == 'mp' and combiner == "concat":
                dense_mp_shard_table_ids += table_ids

    shard_strategy = []
    if len(dp_shard_table_ids) > 0:
        shard_strategy.append(
            ('dp', dp_shard_table_ids)
        )

    if len(sparse_mp_shard_table_ids) > 0:
        if args.disable_fuse_sparse_embedding:
            for each_mp_shard_table_ids in sparse_mp_shard_table_ids:
                shard_strategy.append(
                    ('mp', each_mp_shard_table_ids)
                )
        else:
            fused_sparse_mp_shard_table_ids = list(chain.from_iterable(sparse_mp_shard_table_ids))
            shard_strategy.append(
                ('mp', fused_sparse_mp_shard_table_ids)
            )

    if len(dense_mp_shard_table_ids) > 0:
        shard_strategy.append(
            ('mp', dense_mp_shard_table_ids)
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
        logging.info("\n")
    return shard_matrix, shard_strategy, dense_sparse_table_ids
