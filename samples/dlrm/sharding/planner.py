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


class ShardingState:
    """
    Containing the state of a sharding process.
    The plan iteratively update the sharding state based on a given heuristic and obtain
    solutions.
    """

    def __init__(
        self,
        array_hotness: np.array,
        num_bucket: int,
        dp_table_id: np.array(int) = np.array([]),
    ) -> None:
        mp_table_id = np.setdiff1d(np.arange(array_hotness.size), dp_table_id)
        array_hotness_mp = array_hotness[mp_table_id]
        sorted_idx = np.argsort(array_hotness_mp)[::-1]
        self.array_unshard_hotness = array_hotness
        self.array_hotness = array_hotness_mp[sorted_idx]
        self.num_bucket = num_bucket
        self.array_table_id = mp_table_id[sorted_idx]
        self.array_num_split = np.zeros(self.array_unshard_hotness.size, dtype=int)
        self.array_num_split[mp_table_id] = 1
        self.shard_ll = [[] for i in range(self.num_bucket)]

    def split_hot_shard(self):
        """
        split the shard with the largest hotness
        """
        # shards are sorted based on the hotness. Find the first hot shard that
        # can be split further
        for shard_id in range(self.array_table_id.size):
            table_id = self.array_table_id[shard_id]
            hotness = self.array_unshard_hotness[table_id]
            if self.array_num_split[table_id] * 2 <= self.num_bucket:
                # if this table can be further split and we can put it into
                # more buckets
                idx = np.where(self.array_table_id == table_id)[0]
                self.array_hotness = np.delete(self.array_hotness, idx)
                self.array_table_id = np.delete(self.array_table_id, idx)
                self.array_num_split[table_id] *= 2
                self.array_hotness = np.concatenate(
                    (
                        self.array_hotness,
                        np.ones(self.array_num_split[table_id])
                        * (hotness / self.array_num_split[table_id]),
                    )
                )
                self.array_table_id = np.concatenate(
                    (
                        self.array_table_id,
                        np.ones(self.array_num_split[table_id], dtype=int) * table_id,
                    )
                )
                break

        # sort after splitting to maintain the shard hotness in order
        sorted_idx = np.argsort(self.array_hotness)[::-1]
        self.array_hotness = self.array_hotness[sorted_idx]
        self.array_table_id = self.array_table_id[sorted_idx]

    def split_oom_shard(self, table_id: int) -> bool:
        hotness = self.array_unshard_hotness[table_id]
        if self.array_num_split[table_id] * 2 <= self.num_bucket:
            idx = np.where(self.array_table_id == table_id)[0]
            self.array_hotness = np.delete(self.array_hotness, idx)
            self.array_table_id = np.delete(self.array_table_id, idx)
            self.array_num_split[table_id] *= 2
            self.array_hotness = np.concatenate(
                (
                    self.array_hotness,
                    np.ones(self.array_num_split[table_id])
                    * (hotness / self.array_num_split[table_id]),
                )
            )
            self.array_table_id = np.concatenate(
                (self.array_table_id, np.ones(self.array_num_split[table_id], dtype=int) * table_id)
            )
            sorted_idx = np.argsort(self.array_hotness)[::-1]
            self.array_hotness = self.array_hotness[sorted_idx]
            self.array_table_id = self.array_table_id[sorted_idx]
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
        table_cost: np.array(float),
        mem_cost: np.array(float),
    ) -> None:
        self.cost = cost
        self.hotness_cost = hotness_cost
        self.table_cost = table_cost
        self.mem_cost = mem_cost


class CostModel:
    def __init__(
        self,
        hotness_cost: float,
        table_cost: float,
        mem_cost: float,
        mem_capacity: float,
        table_size: List[int],
    ) -> None:
        self.unit_hotness_cost = hotness_cost
        self.unit_table_cost = table_cost
        self.unit_mem_cost = mem_cost
        self.mem_capacity = mem_capacity
        self.array_table_size = np.array(table_size)

    def get_cost(
        self,
        ss: ShardingState,
    ) -> Tuple[Cost, bool]:
        list_cost = []
        list_hotness_cost = []
        list_table_cost = []
        list_mem_cost = []

        for shard_list in ss.shard_ll:
            hotness_cost = (
                self.unit_hotness_cost
                * (
                    ss.array_unshard_hotness[shard_list] / np.array(ss.array_num_split)[shard_list]
                ).sum()
            )
            table_cost = self.unit_table_cost * len(shard_list)
            mem_cost = (
                self.unit_mem_cost
                * (
                    self.array_table_size[shard_list] / np.array(ss.array_num_split)[shard_list]
                ).sum()
            )
            list_cost.append(hotness_cost + table_cost)
            list_hotness_cost.append(hotness_cost)
            list_table_cost.append(table_cost)
            list_mem_cost.append(mem_cost)

        return (
            Cost(
                np.array(list_cost),
                np.array(list_hotness_cost),
                np.array(list_table_cost),
                np.array(list_mem_cost),
            ),
            max(list_mem_cost) > self.mem_capacity,
        )

    def deduct_mem_cap_for_dp(
        self,
        dp_table_id: list,
    ) -> None:
        self.mem_capacity -= self.array_table_size[dp_table_id].sum() * self.unit_mem_cost
        if self.mem_capacity < 0:
            raise Exception("OOM due to DP. Please considering increase the DP threshold")


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
        num_bucket: int,
        cost_model: CostModel,
        dp_threshold: int = 0,
        max_search_iter: int = 20,
        log_result: bool = False,
    ) -> None:
        self.array_hotness = np.array(list_hotness)
        self.num_bucket = num_bucket
        self.cost_model = cost_model
        self.list_candidate = []
        self.max_search_iter = max_search_iter
        self.log_result = log_result

        # Create the default sharding plan. Throw if even this default sharding plan cannot fit, as
        # it should be the most memory-efficient
        sharding_state_default = ShardingState(self.array_hotness, self.num_bucket)
        for b in range(self.num_bucket):
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
                cost.table_cost,
                cost.mem_cost,
                sharding_state_default.shard_ll,
            )
        )

        # Create DP sharding plan based on the DP threshold
        self.dp_table_id = np.where(
            cost_model.array_table_size < dp_threshold / cost_model.unit_mem_cost
        )[0]
        self.mp_table_id = np.setdiff1d(np.arange(self.array_hotness.size), self.dp_table_id)
        self.sharding_state = ShardingState(self.array_hotness, self.num_bucket, self.dp_table_id)
        self.cost_model.deduct_mem_cap_for_dp(self.dp_table_id)

        logging.basicConfig(level=logging.INFO, format="%(message)s")

    def greedy_plan(self, ss):
        """
        This is a heuristic based on greedy policy. The shard is placed to the bucket with the
        lowest hotness cost
        """
        array_cost = np.zeros(ss.num_bucket)
        ss.reset_shard_ll()
        for i in range(ss.array_hotness.size):
            sorted_idx = np.argsort(array_cost)
            sharded = False
            for bucket_id in sorted_idx:
                if ss.array_table_id[i] not in ss.shard_ll[bucket_id]:
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
                        cost.table_cost,
                        cost.mem_cost,
                        self.sharding_state.shard_ll,
                    )
                )
                self.sharding_state.split_hot_shard()
            else:
                oom_table_can_split = self.sharding_state.split_oom_shard(oom_table_id)
                if not oom_table_can_split:
                    break

        self.list_candidate.sort(key=lambda x: x[0])

        shard_strategy = [("mp", self.mp_table_id.tolist())]
        shard_strategy.append(("dp", self.dp_table_id.tolist()))
        shard_matrix = self.list_candidate[0][-1]
        for table_id in self.dp_table_id:
            for shard_list in shard_matrix:
                shard_list.append(table_id)
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
        return shard_strategy, shard_matrix
