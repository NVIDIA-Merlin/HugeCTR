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
import time
from typing import List, Tuple

import numpy as np
import copy


def ev_size_compensation(ev_sizes_array):
    # use some magic num to evaluation ev_size conpensation
    align_128 = (np.floor((ev_sizes_array - 1) / 128)).astype(np.int32)
    align_128 = (align_128 * 128).astype(np.int32)
    remainders = (ev_sizes_array - align_128).astype(np.int32)

    intercept = int(96)
    slope_range = int(32)

    ev_sizes_compensation = (
        align_128 + intercept + (np.floor((remainders / 128) * slope_range)).astype(np.int32)
    )
    return ev_sizes_array


class ShardingState:
    """
    Containing the state of a sharding process.
    The plan iteratively update the sharding state based on a given heuristic and obtain
    solutions.
    """

    def __init__(
        self,
        array_hotness: np.array,
        array_evsizes: np.array,
        num_bucket: int,
        num_nodes: int,
        num_gpus_per_node: int,
        batch_size: int,
        cost,
        is_hier: bool = False,
        use_column_wise_sharding: bool = False,
    ) -> None:
        if is_hier:
            self.num_bucket = num_nodes
        else:
            self.num_bucket = num_nodes * num_gpus_per_node
        self.num_nodes = num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        self.is_hier = is_hier
        self.use_column_wise_sharding = use_column_wise_sharding

        mp_table_id = np.arange(array_hotness.size)
        array_hotness_mp = array_hotness[mp_table_id]
        array_evsizes_mp = array_evsizes[mp_table_id]
        array_cost_mp = cost.get_cost_per_lookup(
            array_hotness_mp, ev_size_compensation(array_evsizes_mp)
        )

        sorted_idx = np.argsort(array_cost_mp)[::-1]  # hotness idx from max to min
        self.array_unshard_hotness = array_hotness  # raw hotness array
        self.array_unshard_hotness_mp = array_hotness_mp

        self.array_unshard_evsizes = array_evsizes
        self.array_unshard_evsizes_update = copy.deepcopy(array_evsizes)

        self.array_unshard_evsizes_mp = array_evsizes_mp
        self.array_unshard_evsizes_mp_update = array_evsizes_mp

        self.array_unshard_cost_mp = array_cost_mp

        self.array_evsizes = array_evsizes_mp[sorted_idx]
        self.array_hotness = array_hotness_mp[sorted_idx]  # hotness array and sorted
        self.array_cost = array_cost_mp[sorted_idx]  # hotness array and sorted
        self.array_table_id = mp_table_id[sorted_idx]

        self.array_num_split = np.zeros(self.array_unshard_hotness.size, dtype=int)
        self.array_num_split[mp_table_id] = 1
        self.shard_ll = [[] for i in range(self.num_bucket)]  # N device list
        self.is_hier = is_hier

    def split_hot_shard(self, cost, is_column_wise=False):
        """
        split the shard with the largest hotness
        """
        # shards are sorted based on the hotness. Find the first hot shard that
        # can be split further
        # TODO:maybe we can change the algo of split, we need consider the column-wise sharding
        tmp_embedding_length_cost = ev_size_compensation(self.array_evsizes) * self.array_hotness
        tmp_embedding_com_cost = self.array_evsizes * 80
        # tmp_ratio = tmp_embedding_length_cost/tmp_embedding_com_cost
        # tmp_ratio = tmp_embedding_length_cost
        tmp_ratio = tmp_embedding_length_cost + tmp_embedding_com_cost
        # tmp_sorted_idx = np.argsort(tmp_embedding_length_cost)[::-1]
        tmp_sorted_idx = np.argsort(tmp_ratio)[::-1]
        # print("tmp_ratio = ",tmp_ratio," tmp_sorted_idx = ",tmp_sorted_idx)
        # print("self.array_evsizes = ",self.array_evsizes," self.array_hotness = ",self.array_hotness)
        tmp_embedding_length_cost = tmp_embedding_length_cost[tmp_sorted_idx]
        tmp_table_ids = self.array_table_id[tmp_sorted_idx]
        # print("tmp_table_ids = ",tmp_table_ids," is_column_wise = ",is_column_wise)
        for shard_id in range(tmp_table_ids.size):
            table_id = tmp_table_ids[shard_id]
            hotness = self.array_unshard_hotness[table_id]
            split_num_pre = (
                self.array_unshard_evsizes[table_id]
                / self.array_unshard_evsizes_update[table_id]
                * self.array_num_split[table_id]
            )
            if split_num_pre * 2 <= self.num_bucket:
                # if this table can be further split and we can put it into
                # more buckets
                idx = np.where(self.array_table_id == table_id)[0]
                self.array_hotness = np.delete(self.array_hotness, idx)
                self.array_table_id = np.delete(self.array_table_id, idx)
                self.array_evsizes = np.delete(self.array_evsizes, idx)
                if is_column_wise:
                    tmp_table_size = self.array_unshard_evsizes_update[table_id]
                    if tmp_table_size == 0 or tmp_table_size % 2 != 0 or tmp_table_size <= 32:
                        self.array_num_split[table_id] *= 2
                    else:
                        # return back on row-wise
                        self.array_unshard_evsizes_update[table_id] /= 2
                else:
                    self.array_num_split[table_id] *= 2

                split_num = int(
                    self.array_num_split[table_id]
                    * self.array_unshard_evsizes[table_id]
                    / self.array_unshard_evsizes_update[table_id]
                )
                self.array_hotness = np.concatenate(
                    (
                        self.array_hotness,
                        np.ones(split_num) * (hotness / self.array_num_split[table_id]),
                    )
                )
                self.array_table_id = np.concatenate(
                    (self.array_table_id, np.ones(split_num, dtype=int) * table_id)
                )
                self.array_evsizes = np.concatenate(
                    (
                        self.array_evsizes,
                        np.ones(split_num) * (self.array_unshard_evsizes_update[table_id]),
                    )
                )
                # TODO: do we need a flag that means we can know if we already split a table?
                break

        self.array_cost = cost.get_cost_per_lookup(
            self.array_hotness, ev_size_compensation(self.array_evsizes)
        )
        # sort after splitting to maintain the shard hotness in order
        sorted_idx = np.argsort(self.array_cost)[::-1]
        self.array_cost = self.array_cost[sorted_idx]
        self.array_hotness = self.array_hotness[sorted_idx]
        self.array_table_id = self.array_table_id[sorted_idx]
        self.array_evsizes = self.array_evsizes[sorted_idx]
        # print("final self.array_evsizes = ",self.array_evsizes," self.array_hotness = ",self.array_hotness," self.array_table_id = ",self.array_table_id)
        # print("final self.array_num_split = ",self.array_num_split)

    def split_oom_shard(self, table_id, cost, is_column_wise=False):
        hotness = self.array_unshard_hotness[table_id]
        split_num_pre = (
            self.array_unshard_evsizes[table_id]
            / self.array_unshard_evsizes_update[table_id]
            * self.array_num_split[table_id]
        )
        if split_num_pre * 2 <= self.num_bucket:
            idx = np.where(self.array_table_id == table_id)[0]
            self.array_hotness = np.delete(self.array_hotness, idx)
            self.array_table_id = np.delete(self.array_table_id, idx)
            self.array_evsizes = np.delete(self.array_evsizes, idx)
            if is_column_wise:
                tmp_table_size = self.array_unshard_evsizes_update[table_id]
                if tmp_table_size == 0 and tmp_table_size % 2 != 0 or tmp_table_size <= 32:
                    self.array_num_split[table_id] *= 2
                    # return False
                else:
                    self.array_unshard_evsizes_update[table_id] /= 2
            else:
                self.array_num_split[table_id] *= 2
                # split_num = self.array_num_split[table_id]
            split_num = int(
                self.array_num_split[table_id]
                * self.array_unshard_evsizes[table_id]
                / self.array_unshard_evsizes_update[table_id]
            )

            self.array_hotness = np.concatenate(
                (
                    self.array_hotness,
                    np.ones(split_num) * (hotness / self.array_num_split[table_id]),
                )
            )
            self.array_table_id = np.concatenate(
                (self.array_table_id, np.ones(split_num, dtype=int) * table_id)
            )

            self.array_evsizes = np.concatenate(
                (
                    self.array_evsizes,
                    np.ones(split_num) * (self.array_unshard_evsizes_update[table_id]),
                )
            )

            self.array_cost = cost.get_cost_per_lookup(self.array_hotness, self.array_evsizes)
            # sort after splitting to maintain the shard hotness in order
            sorted_idx = np.argsort(self.array_cost)[::-1]
            self.array_cost = self.array_cost[sorted_idx]
            self.array_hotness = self.array_hotness[sorted_idx]
            self.array_table_id = self.array_table_id[sorted_idx]
            self.array_evsizes = self.array_evsizes[sorted_idx]
            return True
        else:
            return False

    def update_split_num(self):
        self.array_num_split = np.zeros_like(self.array_unshard_hotness)
        for shard_list in self.shard_ll:
            for table_id in shard_list:
                self.array_num_split[table_id] += 1

    def reset_shard_ll(self):
        self.shard_ll = [[] for i in range(self.num_bucket)]

    def get_column_wise_sharding_nums(self):
        return self.array_unshard_evsizes / self.array_unshard_evsizes_update

    def push_bucket(
        self,
        bucket_id: int,
        table_id: int,
    ) -> None:
        self.shard_ll[bucket_id].append(table_id)

    def pop_bucket(
        self,
        bucket_id: int,
    ) -> None:
        self.shard_ll[bucket_id].pop()


class Cost:
    def __init__(
        self,
        cost: np.array(float),
        hotness_cost: np.array(float),
        comm_cost: np.array(float),
        mem_cost: np.array(float),
    ) -> None:
        self.cost = cost
        self.hotness_cost = hotness_cost
        self.comm_cost = comm_cost
        self.mem_cost = mem_cost


class CostModel:
    def __init__(
        self,
        hotness_cost: float,
        band_width_ratio: float,
        sparse_work_ratio: float,
        dense_work_ratio: float,
        batchsize: int,
        mem_cost,
        ev_sizes: np.array(int),
        mem_capacity: float,
        table_size: List[int],
        key_embedding_type_ratio: int = 1,
    ) -> None:
        self.unit_hotness_cost = hotness_cost
        self.band_width_ratio = band_width_ratio
        self.sparse_work_ratio = sparse_work_ratio
        self.dense_work_ratio = dense_work_ratio
        self.batchsize = batchsize
        self.unit_mem_cost = mem_cost
        self.ev_sizes = ev_sizes
        self.mem_capacity = mem_capacity
        self.array_table_size = np.array(table_size)
        self.key_embedding_type_ratio = key_embedding_type_ratio

    def get_cost(
        self,
        ss: ShardingState,
    ) -> Tuple[Cost, bool]:
        list_cost = []
        list_hotness_cost = []
        list_comm_cost = []
        list_mem_cost = []

        for shard_list in ss.shard_ll:
            hotness_cost = (
                self.unit_hotness_cost
                * self.sparse_work_ratio
                * self.batchsize
                * (
                    ss.array_unshard_hotness[shard_list]
                    * ev_size_compensation(ss.array_unshard_evsizes_update[shard_list])
                    / np.array(ss.array_num_split)[shard_list]
                ).sum()
            )
            comm_cost = (
                self.band_width_ratio * self.batchsize * ss.array_unshard_evsizes_update[shard_list]
            ).sum()
            # TODO:add a data distributor cost to comm_cost

            if ss.is_hier:
                comm_cost = comm_cost / ss.num_gpus_per_node
                # TODO:give a ratio of nvlink bw compare to NIC bw
                comm_cost = comm_cost * 3 / 2
                hotness_cost = hotness_cost / ss.num_gpus_per_node
            if len(shard_list) > 0:
                mem_cost = (
                    self.unit_mem_cost[shard_list]
                    / (
                        ss.array_unshard_evsizes[shard_list]
                        / ss.array_unshard_evsizes_update[shard_list]
                    )
                    * self.array_table_size[shard_list]
                    / np.array(ss.array_num_split)[shard_list]
                ).sum()
            else:
                mem_cost = 0
            list_cost.append(hotness_cost + comm_cost)
            list_hotness_cost.append(hotness_cost)
            list_comm_cost.append(comm_cost)
            list_mem_cost.append(mem_cost)

        return (
            Cost(
                np.array(list_cost),
                np.array(list_hotness_cost),
                np.array(list_comm_cost),
                np.array(list_mem_cost),
            ),
            max(list_mem_cost) > self.mem_capacity,
        )

    def get_cost_per_lookup(self, hotness_array: np.array, ev_sizes_array: np.array) -> np.array:
        hotness_cost = (
            self.unit_hotness_cost
            * self.sparse_work_ratio
            * self.batchsize
            * ev_size_compensation(ev_sizes_array)
            * hotness_array
        )
        comm_cost = self.band_width_ratio * self.batchsize * ev_sizes_array
        return hotness_cost + comm_cost


class Planner:
    """
    The planner work out a series of plans iteratively.
    In each iteration, the planner tries to split the hottest shard and place the shards into
    a bucket based on a give heuristic. When the shard is too large to fit into the best bucket
    suggested by the heuristic, it finds the next best bucket until it iterates through all the
    buckets. In that case, it tries to split the shard further. If the shard cannot be split
    further, the planner aborts and returns the default sharding plan.
    """

    def __init__(
        self,
        list_hotness: list,
        ev_sizes: np.array,
        num_nodes: int,
        num_gpus_per_node: int,
        batchsize: int,
        is_hier: bool,
        cost_model: CostModel,
        max_search_iter: int = 20,
        use_column_wise_sharding: bool = False,
        log_result: bool = False,
    ) -> None:
        self.array_hotness = np.array(list_hotness)
        self.ev_sizes = np.array(ev_sizes)
        self.num_nodes = num_nodes  # numGPUS or numNodes
        self.num_gpus_per_node = num_gpus_per_node  # numGPUS or numNodesa
        self.batchsize = batchsize
        self.use_column_wise_sharding = use_column_wise_sharding
        if is_hier:
            self.num_bucket = num_nodes  # numGPUS or numNodes
        else:
            self.num_bucket = num_nodes * num_gpus_per_node  # numGPUS or numNodes
        self.cost_model = cost_model
        self.list_candidate = []
        self.max_search_iter = max_search_iter
        self.log_result = log_result

        # Create the default sharding plan. Throw if even this default sharding plan cannot fit, as
        # it should be the most memory-efficient
        sharding_state_default = ShardingState(
            self.array_hotness,
            self.ev_sizes,
            self.num_bucket,
            self.num_nodes,
            self.num_gpus_per_node,
            self.batchsize,
            self.cost_model,
            is_hier,
        )

        # num device
        for b in range(self.num_bucket):
            # num hotness
            for t in range(self.array_hotness.size):
                sharding_state_default.push_bucket(b, t)
        sharding_state_default.update_split_num()
        cost, oom = self.cost_model.get_cost(sharding_state_default)
        if oom:
            raise Exception("OOM even with the most memory-efficient sharding plan")
        self.list_candidate.append(
            (
                cost.cost.max(),
                cost.hotness_cost,
                cost.comm_cost,
                cost.mem_cost,
                sharding_state_default.get_column_wise_sharding_nums(),
                sharding_state_default.shard_ll,
            )
        )

        # Create DP sharding plan based on the DP threshold

        self.mp_table_id = np.arange(self.array_hotness.size)
        self.sharding_state = sharding_state_default = ShardingState(
            self.array_hotness,
            self.ev_sizes,
            self.num_bucket,
            self.num_nodes,
            self.num_gpus_per_node,
            self.batchsize,
            self.cost_model,
            is_hier,
            self.use_column_wise_sharding,
        )

        logging.basicConfig(level=logging.INFO, format="%(message)s")

    def greedy_plan(self, ss):
        """
        This is a heuristic based on greedy policy. The shard is placed to the bucket with the
        lowest hotness cost
        """
        array_cost = np.zeros(ss.num_bucket)

        ss.reset_shard_ll()

        for i in range(ss.array_cost.size):  # mp table num
            sorted_idx = np.argsort(array_cost)
            sharded = False
            for bucket_id in sorted_idx:
                if (
                    ss.array_table_id[i] not in ss.shard_ll[bucket_id]
                ):  # ss.array_table_id mp table_id
                    # for now, only uniform sharding is supported. Hence cannot put two shards
                    # from the same table into the same bucket
                    ss.push_bucket(bucket_id, ss.array_table_id[i])
                    cost, oom = self.cost_model.get_cost(ss)
                    if not oom:
                        sharded = True
                        array_cost = cost.cost
                        break
                    else:
                        # Current bucket cannot fit. Iterate to the next best bucket
                        ss.pop_bucket(bucket_id)
            if not sharded:
                # This means the shard is too large to fit within any bucket
                return ss.array_table_id[i], ss, cost
        return None, ss, cost

    def plan(self):
        t0 = time.time()
        for i in range(self.max_search_iter):
            oom_table_id, self.sharding_state, cost = self.greedy_plan(self.sharding_state)

            if oom_table_id is None:
                self.list_candidate.append(
                    (
                        cost.cost.max(),
                        cost.hotness_cost,
                        cost.comm_cost,
                        cost.mem_cost,
                        self.sharding_state.get_column_wise_sharding_nums(),
                        self.sharding_state.shard_ll,
                    )
                )
                # print(" self.max_search_iter ",self.max_search_iter,"i = ",i," cost.cost.max() = ",cost.cost.max() , "cost.hotness_cost ",cost.hotness_cost , "cost.comm_cost = ",cost.comm_cost)

                if self.use_column_wise_sharding:
                    # we have two choice , first is row split , second is column split
                    sharding_state_row = copy.deepcopy(self.sharding_state)
                    sharding_state_col = copy.deepcopy(self.sharding_state)
                    my_dict = vars(sharding_state_row)
                    ss_dict = vars(self.sharding_state)
                    sharding_state_row.split_hot_shard(cost=self.cost_model, is_column_wise=False)
                    oom_table_id_row, sharding_state_row, cost_row = self.greedy_plan(
                        sharding_state_row
                    )
                    sharding_state_col.split_hot_shard(cost=self.cost_model, is_column_wise=True)
                    oom_table_id_col, sharding_state_col, cost_col = self.greedy_plan(
                        sharding_state_col
                    )

                    # 2 split method is success ,compare them cost
                    if oom_table_id_row is None and oom_table_id_col is None:
                        cost_max_row = cost_row.cost.max()
                        cost_max_col = cost_col.cost.max()
                        if cost_max_row > cost_max_col:
                            self.sharding_state.split_hot_shard(
                                cost=self.cost_model, is_column_wise=True
                            )
                        else:
                            self.sharding_state.split_hot_shard(
                                cost=self.cost_model, is_column_wise=False
                            )
                        continue

                    # row split is success , split row
                    if oom_table_id_row is None:
                        self.sharding_state.split_hot_shard(
                            cost=self.cost_model, is_column_wise=True
                        )
                        continue

                    # col split is success , split col
                    if oom_table_id_col is None:
                        self.sharding_state.split_hot_shard(
                            cost=self.cost_model, is_column_wise=False
                        )
                        continue

                    # 2 split method is failure , split row by default
                    if oom_table_id_row is not None and oom_table_id_col is not None:
                        self.sharding_state.split_hot_shard(
                            cost=self.cost_model, is_column_wise=False
                        )

                else:
                    self.sharding_state.split_hot_shard(cost=self.cost_model)
            else:
                if self.use_column_wise_sharding:
                    sharding_state_row = copy.deepcopy(self.sharding_state)
                    sharding_state_col = copy.deepcopy(self.sharding_state)

                    oom_table_can_split_row = sharding_state_row.split_oom_shard(
                        oom_table_id, cost=self.cost_model, is_column_wise=False
                    )
                    oom_table_can_split_col = sharding_state_col.split_oom_shard(
                        oom_table_id, cost=self.cost_model, is_column_wise=True
                    )
                    if (not oom_table_can_split_row) and (not oom_table_can_split_col):
                        oom_table_can_split = False
                    elif (oom_table_can_split_row) and (not oom_table_can_split_col):
                        oom_table_can_split = self.sharding_state.split_oom_shard(
                            oom_table_id, cost=self.cost_model, is_column_wise=False
                        )
                    elif (not oom_table_can_split_row) and (oom_table_can_split_col):
                        oom_table_can_split = self.sharding_state.split_oom_shard(
                            oom_table_id, cost=self.cost_model, is_column_wise=True
                        )
                    else:
                        oom_table_can_split = True
                        oom_table_id_row, sharding_state_row, cost_row = self.greedy_plan(
                            sharding_state_row
                        )
                        oom_table_id_col, sharding_state_col, cost_col = self.greedy_plan(
                            sharding_state_col
                        )

                        cost_max_row = cost_row.cost.max()
                        cost_max_col = cost_col.cost.max()
                        if cost_max_row > cost_max_col:
                            _ = self.sharding_state.split_oom_shard(
                                oom_table_id, cost=self.cost_model, is_column_wise=True
                            )
                        else:
                            _ = self.sharding_state.split_oom_shard(
                                oom_table_id, cost=self.cost_model, is_column_wise=False
                            )
                else:
                    oom_table_can_split = self.sharding_state.split_oom_shard(
                        oom_table_id, cost=self.cost_model, is_column_wise=True
                    )
                if not oom_table_can_split:
                    break

        self.list_candidate.sort(key=lambda x: x[0])

        sparse_cost = self.list_candidate[0][0]
        print("sparse cost = ", sparse_cost)

        shard_matrix = self.list_candidate[0][-1]
        shard_column_wise_nums = np.array(self.list_candidate[0][-2])
        shard_column_wise_nums_mp = shard_column_wise_nums[self.mp_table_id].astype(np.int32)
        mp_shard_info = list(zip(self.mp_table_id.tolist(), shard_column_wise_nums_mp.tolist()))
        shard_strategy = [("mp", mp_shard_info)]
        if self.log_result:
            logging.info("Planner took %f sec" % (time.time() - t0))
            logging.info(shard_strategy)
            logging.info(shard_matrix)
            logging.info("hotness cost is:")
            logging.info(self.list_candidate[0][1])
            logging.info("table cost is:")
            logging.info(self.list_candidate[0][2])
            logging.info("mem cost is:")
            logging.info(self.list_candidate[0][3])
        return shard_strategy, shard_matrix, shard_column_wise_nums
