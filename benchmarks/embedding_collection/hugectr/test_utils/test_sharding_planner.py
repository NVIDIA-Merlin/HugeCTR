import inspect
import os
import sys
import unittest
from itertools import chain
import numpy as np


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

ev_sizes = np.array([256 - 4 * x for x in range(len(list_hotness))])


def sanity_check(shard_matrix, shard_strategy):
    # To make sure all the tables are sharded
    assert set(chain(*shard_matrix)) == set(
        [x for x in range(len(list_table_size))]
    ), "Not all tables covered in the sharding plan"
    shard_strategy_list_raw = [x for strategy_pair in shard_strategy for x in strategy_pair[1]]
    shard_strategy_list = []
    for i in range(len(shard_strategy_list_raw)):
        if isinstance(shard_strategy_list_raw[i], tuple):
            shard_strategy_list.append(shard_strategy_list_raw[i][0])
        else:
            shard_strategy_list.append(shard_strategy_list_raw[i])

    assert set(shard_strategy_list) == set(
        [x for x in range(len(list_table_size))]
    ), "Not all tables covered in the sharding plan"

    # To make sure no duplicated shards on one GPU
    for shard_list in shard_matrix:
        assert len(set(shard_list)) == len(shard_list)


class TestShardingPlanner(unittest.TestCase):
    def test_single_node(self):
        print("######################sigle-node case")
        band_width_ratio = 7
        sparse_work_ratio = 4
        dense_work_ratio = 4
        batchsize = 2048
        mem_cost = ev_sizes * 8 * 1e-9
        cost_model = CostModel(
            1,
            band_width_ratio,
            sparse_work_ratio,
            dense_work_ratio,
            batchsize,
            mem_cost,
            ev_sizes,
            60,
            list_table_size,
            1,
            0.3,
        )
        planner = Planner(list_hotness, ev_sizes, 1, 8, batchsize, False, cost_model)
        shard_strategy, shard_matrix, shard_column_wise_num = planner.plan()
        print("shard_strategy = ", shard_strategy)
        print("**********************shard_matrix = ", shard_matrix)
        sanity_check(shard_matrix, shard_strategy)

    def test_two_node(self):
        print("######################two-node case")
        band_width_ratio = 72
        sparse_work_ratio = 4
        dense_work_ratio = 4
        batchsize = 2048
        mem_cost = ev_sizes * 8 * 1e-9

        cost_model = CostModel(
            1,
            band_width_ratio,
            sparse_work_ratio,
            dense_work_ratio,
            batchsize,
            mem_cost,
            ev_sizes,
            60,
            list_table_size,
            1,
            0.3,
        )
        planner = Planner(list_hotness, ev_sizes, 2, 8, batchsize, False, cost_model)
        shard_strategy, shard_matrix, shard_column_wise_num = planner.plan()
        print("shard_strategy = ", shard_strategy)
        print("**********************shard_matrix = ", shard_matrix)
        sanity_check(shard_matrix, shard_strategy)

    def test_two_node_hier(self):
        print("######################two-node hier case")
        band_width_ratio = 72
        sparse_work_ratio = 4
        dense_work_ratio = 4
        batchsize = 2048
        mem_cost = ev_sizes * 8 * 1e-9

        cost_model = CostModel(
            1,
            band_width_ratio,
            sparse_work_ratio,
            dense_work_ratio,
            batchsize,
            mem_cost,
            ev_sizes,
            480,
            list_table_size,
            1,
            0.3,
        )
        planner = Planner(list_hotness, ev_sizes, 2, 8, batchsize, True, cost_model)
        shard_strategy, shard_matrix, shard_column_wise_num = planner.plan()
        print("shard_strategy = ", shard_strategy)
        print("**********************shard_matrix = ", shard_matrix)
        sanity_check(shard_matrix, shard_strategy)

    def test_four_node(self):
        print("######################four-node case")
        band_width_ratio = 72
        sparse_work_ratio = 4
        dense_work_ratio = 4
        batchsize = 2048
        mem_cost = ev_sizes * 8 * 1e-9

        cost_model = CostModel(
            1,
            band_width_ratio,
            sparse_work_ratio,
            dense_work_ratio,
            batchsize,
            mem_cost,
            ev_sizes,
            60,
            list_table_size,
            1,
            0.3,
        )
        planner = Planner(list_hotness, ev_sizes, 4, 8, batchsize, False, cost_model)
        shard_strategy, shard_matrix, shard_column_wise_num = planner.plan()
        print("shard_strategy = ", shard_strategy)
        print("**********************shard_matrix = ", shard_matrix)
        sanity_check(shard_matrix, shard_strategy)

    def test_four_node_hier(self):
        print("######################four-node hier case")
        band_width_ratio = 72
        sparse_work_ratio = 4
        dense_work_ratio = 4
        batchsize = 2048
        mem_cost = ev_sizes * 8 * 1e-9

        cost_model = CostModel(
            1,
            band_width_ratio,
            sparse_work_ratio,
            dense_work_ratio,
            batchsize,
            mem_cost,
            ev_sizes,
            480,
            list_table_size,
            1,
            0.3,
        )
        planner = Planner(list_hotness, ev_sizes, 4, 8, batchsize, True, cost_model)
        shard_strategy, shard_matrix, shard_column_wise_num = planner.plan()
        print("shard_strategy = ", shard_strategy)
        print("**********************shard_matrix = ", shard_matrix)
        sanity_check(shard_matrix, shard_strategy)

    def test_two_node_with_column_wise_sharding(self):
        print("######################two-node case with column wise sharding")
        band_width_ratio = 72
        sparse_work_ratio = 4
        dense_work_ratio = 4
        batchsize = 2048
        mem_cost = ev_sizes * 8 * 1e-9

        cost_model = CostModel(
            1,
            band_width_ratio,
            sparse_work_ratio,
            dense_work_ratio,
            batchsize,
            mem_cost,
            ev_sizes,
            60,
            list_table_size,
            1,
            0.3,
        )
        planner = Planner(
            list_hotness,
            ev_sizes,
            2,
            8,
            batchsize,
            False,
            cost_model,
            use_column_wise_sharding=True,
        )
        shard_strategy, shard_matrix, shard_column_wise_num = planner.plan()
        print("shard_strategy = ", shard_strategy)
        print(
            "**********************shard_matrix = ",
            shard_matrix,
            " shard_column_wise_num = ",
            shard_column_wise_num,
        )
        sanity_check(shard_matrix, shard_strategy)

    def test_two_node_hier_with_column_wise_sharding(self):
        print("######################two-node hier case with column wise sharding")
        band_width_ratio = 72
        sparse_work_ratio = 4
        dense_work_ratio = 4
        batchsize = 2048
        mem_cost = ev_sizes * 8 * 1e-9

        cost_model = CostModel(
            1,
            band_width_ratio,
            sparse_work_ratio,
            dense_work_ratio,
            batchsize,
            mem_cost,
            ev_sizes,
            480,
            list_table_size,
            1,
            0.3,
        )
        planner = Planner(
            list_hotness, ev_sizes, 2, 8, batchsize, True, cost_model, use_column_wise_sharding=True
        )
        shard_strategy, shard_matrix, shard_column_wise_num = planner.plan()
        print("shard_strategy = ", shard_strategy)
        print(
            "**********************shard_matrix = ",
            shard_matrix,
            " shard_column_wise_num = ",
            shard_column_wise_num,
        )
        sanity_check(shard_matrix, shard_strategy)

    def test_four_node_with_column_wise_sharding(self):
        print("######################four-node case with column wise sharding")
        band_width_ratio = 72
        sparse_work_ratio = 4
        dense_work_ratio = 4
        batchsize = 2048
        mem_cost = ev_sizes * 8 * 1e-9

        cost_model = CostModel(
            1,
            band_width_ratio,
            sparse_work_ratio,
            dense_work_ratio,
            batchsize,
            mem_cost,
            ev_sizes,
            60,
            list_table_size,
            1,
            0.3,
        )
        planner = Planner(
            list_hotness,
            ev_sizes,
            4,
            8,
            batchsize,
            False,
            cost_model,
            use_column_wise_sharding=True,
        )
        shard_strategy, shard_matrix, shard_column_wise_num = planner.plan()
        print("shard_strategy = ", shard_strategy)
        print(
            "**********************shard_matrix = ",
            shard_matrix,
            " shard_column_wise_num = ",
            shard_column_wise_num,
        )
        sanity_check(shard_matrix, shard_strategy)

    def test_four_node_hier_with_column_wise_sharding(self):
        print("######################four-node hier case with column wise sharding")
        band_width_ratio = 72
        sparse_work_ratio = 4
        dense_work_ratio = 4
        batchsize = 2048
        mem_cost = ev_sizes * 8 * 1e-9

        cost_model = CostModel(
            1,
            band_width_ratio,
            sparse_work_ratio,
            dense_work_ratio,
            batchsize,
            mem_cost,
            ev_sizes,
            480,
            list_table_size,
            1,
            0.3,
        )
        planner = Planner(
            list_hotness, ev_sizes, 4, 8, batchsize, True, cost_model, use_column_wise_sharding=True
        )
        shard_strategy, shard_matrix, shard_column_wise_num = planner.plan()
        print("shard_strategy = ", shard_strategy)
        print(
            "**********************shard_matrix = ",
            shard_matrix,
            " shard_column_wise_num = ",
            shard_column_wise_num,
        )
        sanity_check(shard_matrix, shard_strategy)

    def test_four_node_hier_with_column_wise_sharding_with_dp(self):
        print("######################four-node hier case with column wise sharding dp")
        band_width_ratio = 75
        sparse_work_ratio = 4
        dense_work_ratio = 4
        batchsize = 2048
        mem_cost = ev_sizes * 8 * 1e-9

        cost_model = CostModel(
            1,
            band_width_ratio,
            sparse_work_ratio,
            dense_work_ratio,
            batchsize,
            mem_cost,
            ev_sizes,
            480,
            list_table_size,
            1,
            0.3,
        )
        planner = Planner(
            list_hotness,
            ev_sizes,
            4,
            8,
            batchsize,
            True,
            cost_model,
            use_column_wise_sharding=True,
            dp_threshold=5,
        )
        shard_strategy, shard_matrix, shard_column_wise_num = planner.plan()
        print("shard_strategy = ", shard_strategy)
        print(
            "**********************shard_matrix = ",
            shard_matrix,
            " shard_column_wise_num = ",
            shard_column_wise_num,
        )
        sanity_check(shard_matrix, shard_strategy)

    def test_oom_throw(self):
        # test throw from an OOM case
        print("######################oom raise error")
        band_width_ratio = 72
        sparse_work_ratio = 4
        dense_work_ratio = 4
        batchsize = 2048
        mem_cost = ev_sizes * 8 * 1e-9
        list_table_size_oom = list(list_table_size)
        list_table_size_oom[0] = 4e8
        cost_model = CostModel(
            1,
            band_width_ratio,
            sparse_work_ratio,
            dense_work_ratio,
            batchsize,
            mem_cost,
            ev_sizes,
            60,
            list_table_size_oom,
            1,
            0.4,
        )

        self.assertRaises(
            Exception, Planner, list_hotness, ev_sizes, 1, 8, batchsize, False, cost_model
        )
        print("######################oom raise error end")


if __name__ == "__main__":
    unittest.main()
