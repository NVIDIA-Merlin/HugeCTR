import inspect
import os
import sys
import unittest
from itertools import chain

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from sharding import CostModel, Planner

list_table_size = [
    40000000,
    39060,
    17295,
    7424,
    20265,
    3,
    7122,
    1543,
    63,
    40000000,
    3067956,
    405282,
    10,
    2209,
    11938,
    155,
    4,
    976,
    14,
    40000000,
    40000000,
    40000000,
    590152,
    12973,
    108,
    36,
]
list_hotness = [
    3,
    2,
    1,
    2,
    6,
    1,
    1,
    1,
    1,
    7,
    3,
    8,
    1,
    6,
    9,
    5,
    1,
    1,
    1,
    12,
    100,
    27,
    10,
    3,
    1,
    1,
]


def sanity_check(shard_matrix, shard_strategy):
    # To make sure all the tables are sharded
    assert set(chain(*shard_matrix)) == set(
        [x for x in range(len(list_table_size))]
    ), "Not all tables covered in the sharding plan"
    shard_strategy_list = [x for strategy_pair in shard_strategy for x in strategy_pair[1]]
    assert set(shard_strategy_list) == set(
        [x for x in range(len(list_table_size))]
    ), "Not all tables covered in the sharding plan"

    # To make sure no duplicated shards on one GPU
    for shard_list in shard_matrix:
        assert len(set(shard_list)) == len(shard_list)


class TestShardingPlanner(unittest.TestCase):
    def test_single_node(self):
        print("sigle-node case")
        cost_model = CostModel(1, 2 / 300e9 * 2e12 / 10, 128 * 8 * 1e-9, 60, list_table_size)
        planner = Planner(list_hotness, 8, cost_model)
        shard_strategy, shard_matrix = planner.plan()
        sanity_check(shard_matrix, shard_strategy)
        print()

    def test_two_node(self):
        print("two-node case")
        cost_model = CostModel(1, 2 / 50e9 * 2e12 / 10, 128 * 8 * 1e-9, 60, list_table_size)
        planner = Planner(list_hotness, 16, cost_model)
        shard_strategy, shard_matrix = planner.plan()
        sanity_check(shard_matrix, shard_strategy)

    def test_four_node(self):
        print("four-node case")
        cost_model = CostModel(1, 2 / 25e9 * 2e12 / 10, 128 * 8 * 1e-9, 60, list_table_size)
        planner = Planner(list_hotness, 32, cost_model)
        shard_strategy, shard_matrix = planner.plan()
        sanity_check(shard_matrix, shard_strategy)

    def test_oom_throw(self):
        # test throw from an OOM case
        list_table_size_oom = list(list_table_size)
        list_table_size_oom[0] = 4e8
        cost_model = CostModel(1, 2 / 300e9 * 2e12 / 10, 128 * 8 * 1e-9, 60, list_table_size_oom)
        self.assertRaises(Exception, Planner, list_hotness, 8, cost_model)


if __name__ == "__main__":
    unittest.main()
