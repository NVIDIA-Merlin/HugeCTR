from collections import defaultdict
import enum
from itertools import accumulate
from traceback import print_tb
import typing
import abc
import random
from common import *
import numpy as np


class EmbeddingTableParam(typing.NamedTuple):
    id_space: int
    vocabulary_size: int
    ev_size: int

class ILookup(abc.ABC):
    @abc.abstractmethod
    def lookup(
        self,
        keys: typing.List[int],
        num_keys: int,
        id_space_offset: typing.List[int] or None,
        num_offset: typing.List[int] or None,
        id_space_list: typing.List[int],
    ) -> typing.List[typing.List[float]]:
        pass


class IEmbeddingTable(ILookup):
    @abc.abstractmethod
    def lookup(
        self,
        keys: typing.List[int],
        num_keys: int,
        id_space_offset: typing.List[int] or None,
        num_offset: typing.List[int] or None,
        id_space_list: typing.List[int],
    ) -> typing.List[typing.List[float]]:
        pass

    def update(
        self, 
        keys: typing.List[int],
        num_keys: int,
        offset: typing.List[int],
        num_offset: int,
        grad_ev: typing.List[float],
        grad_ev_id_space_offset: typing.List[int],
        d_ev_size_list: typing.List[int],
        id_space_list: typing.List[int],
    ):
        pass

    # after init, num of key can not be changed
    def init(
        self, 
        keys: typing.List[int],
        num_keys: int,
        offset: typing.List[int],
        num_offset: int
    ):
        pass
    
    def dump(self):
        pass

# a helper function
def global_init_key(num_gpus, global_embedding_sharding_param_list, global_emb_tables, embedding_collection_param, init_keys):
    for gpu_id in range(num_gpus):
        local_embedding_sharding_param_list = global_embedding_sharding_param_list[gpu_id]
        local_emb_table_list = global_emb_tables[gpu_id]
        assert len(local_embedding_sharding_param_list) == len(local_emb_table_list)
        for idx in range(len(local_embedding_sharding_param_list)):
            embedding_sharding_param = local_embedding_sharding_param_list[idx]
            embedding_table = local_emb_table_list[idx]
            sharding_keys = []
            offset = [0]
            id_space_set = set()
            for local_embedding_id in embedding_sharding_param.local_embedding_list:
                id_space = embedding_collection_param.embedding_params[local_embedding_id].id_space
                if id_space in id_space_set:
                    continue
                id_space_set.add(id_space)
                for key in init_keys[id_space]:
                    if key % embedding_sharding_param.num_sharding == embedding_sharding_param.sharding_id:
                        sharding_keys.append(key)
                offset.append(len(sharding_keys))
            embedding_table.init(sharding_keys, len(sharding_keys), offset, len(offset))

    # hack for dp. embedding table should be responsible for init same ev for same key on differente device
    for gpu_id in range(num_gpus):
        local_embedding_sharding_param_list = global_embedding_sharding_param_list[gpu_id]
        local_emb_table_list = global_emb_tables[gpu_id]
        assert len(local_embedding_sharding_param_list) == len(local_emb_table_list)
        for idx in range(len(local_embedding_sharding_param_list)):
            embedding_sharding_param = local_embedding_sharding_param_list[idx]
            embedding_table = local_emb_table_list[idx]
            if embedding_sharding_param.table_placement_strategy == 'dp':
                embedding_table.embedding_table = global_emb_tables[0][idx].embedding_table


def global_dump_emb_table(num_gpus, global_embedding_sharding_param_list, global_emb_tables):
    emb_table_dict = defaultdict(dict)
    for gpu_id in range(num_gpus):
        local_embedding_sharding_param_list = global_embedding_sharding_param_list[gpu_id]
        local_emb_table_list = global_emb_tables[gpu_id]
        assert len(local_embedding_sharding_param_list) == len(local_emb_table_list)
        for idx in range(len(local_embedding_sharding_param_list)):
            embedding_sharding_param = local_embedding_sharding_param_list[idx]
            embedding_table = local_emb_table_list[idx]
            if embedding_sharding_param.table_placement_strategy == 'dp' and gpu_id > 0:
                continue
            keys, id_space_offset_list, ev_list, ev_size_list, id_space_list = embedding_table.dump()
            ev_offset = 0
            for i in range(len(id_space_list)):
                id_space = id_space_list[i]
                start = id_space_offset_list[i]
                end = id_space_offset_list[i + 1]
                ev_size = ev_size_list[i]
                for r in range(start, end):
                    k = keys[r]
                    ev = ev_list[ev_offset: ev_offset + ev_size]
                    emb_table_dict[id_space][k] = ev
                    ev_offset += ev_size
    # for id_space in emb_table_dict:
    #     for key in emb_table_dict[id_space]:
    #         print('id_space:{}, key:{}, ev_size:{}'.format(id_space, key, len(emb_table_dict[id_space][key])))
    return emb_table_dict
            

class GroupedStaticEmbeddingTable(IEmbeddingTable):
    def __init__(self, param_list: typing.List[EmbeddingTableParam]) -> None:
        super().__init__()
        if len(param_list) > 0:
            assert any([p.vocabulary_size != -1 for p in param_list])  # do not support dynamic size
            self.id_space_list = [p.id_space for p in param_list]
            self.id_space_idx = [-1 for _ in range(max(self.id_space_list) + 1)]
            for idx, id_space in enumerate(self.id_space_list):
                self.id_space_idx[id_space] = idx

            self.ev_size_list = [p.ev_size for p in param_list]
            self.max_vocabulary_size = sum([p.vocabulary_size for p in param_list])
        else:
            self.id_space_list = []
            self.id_space_idx = []

            self.ev_size_list = []
            self.max_vocabulary_size = 0

    def lookup(
        self,
        keys: typing.List[int],
        num_keys: int,
        offset: typing.List[int],
        num_offset: int,
        id_space_list: typing.List[int],
    ) -> typing.List[typing.List[float]]:
        emb_vec = [None for _ in range(num_keys)]
        for idx in range(num_offset - 1):
            id_space = id_space_list[idx]
            start = offset[idx]
            end = offset[idx + 1]
            local_idx = self.id_space_idx[id_space]
            table_size_offset = self.embedding_table_size_offset[local_idx]
            ev_size = self.ev_size_list[local_idx]

            for i in range(start, end):
                key = keys[i]
                location = self.key_location[id_space][key]
                emb_vec[i] = [0. for _ in range(ev_size)]
                for ev_id in range(ev_size):
                    emb_vec[i][ev_id] = self.embedding_table[table_size_offset + location * ev_size + ev_id]
        return emb_vec

    def update(
        self, 
        keys: typing.List[int],
        num_keys: int,
        offset: typing.List[int],
        num_offset: int,
        d_ev_size_list: typing.List[int],
        d_ev_size_offset: typing.List[int],
        id_space_list: typing.List[int],
        grad_emb_vec: typing.List[float]
    ):
        pass

    # after init, num of key can not be changed
    def init(
        self, 
        keys: typing.List[int],
        num_keys: int,
        offset: typing.List[int],
        num_offset: int
    ):
        assert num_offset == len(self.id_space_list) + 1
        emb_table_size_list = []
        self.vocabulary_size_list = []
        self.key_location = defaultdict(dict)
        self.id_space_offset_list = [0]
        for idx in range(num_offset - 1):
            start = offset[idx]
            end = offset[idx + 1]
            vocabulary_size = end - start
            id_space = self.id_space_list[idx]

            self.vocabulary_size_list.append(vocabulary_size)
            emb_table_size_list.append(vocabulary_size * self.ev_size_list[idx])

            id_space_offset = self.id_space_offset_list[-1]
            for i in range(start, end):
                key = keys[i]
                self.key_location[id_space][key] = i - start
            self.id_space_offset_list.append(id_space_offset + end - start)
        emb_table_size = sum(emb_table_size_list)

        self.keys = keys
        self.embedding_table = np.random.rand(emb_table_size)
        self.embedding_table_size_offset = list(accumulate([0] + emb_table_size_list))
        self.vocabulary_size_offset = list(accumulate([0] + self.vocabulary_size_list))

    def dump(self):
        return self.keys, self.id_space_offset_list, self.embedding_table, self.ev_size_list, self.id_space_list

# A helper class to test single table use case
class EmbeddingTable(IEmbeddingTable):
    def __init__(self, param: EmbeddingTableParam) -> None:
        self.param = param
        self.embedding_table = np.asarray([[key for _ in range(param.ev_size)] for key in range(param.vocabulary_size)])

    def lookup(
        self,
        keys: typing.List[int],
        num_keys: int,
        id_space_offset: typing.List[int],
        num_offset: typing.List[int],
        id_space_list: typing.List[int],
    ) -> typing.List[typing.List[float]]:
        emb_vec = []
        for i in range(num_keys):
            emb_vec.append(self.embedding_table[keys[i]].tolist())
        return emb_vec


class GroupedEmbeddingTable(IEmbeddingTable):
    def __init__(self, emb_tables_dict) -> None:
        self.emb_table_dict = emb_tables_dict
        for id_space, emb_table in self.emb_table_dict.items():
            print(id_space, emb_table.param)

    def lookup(
        self,
        keys: typing.List[int],
        num_keys: int,
        offset: typing.List[int],
        num_offset: int,
        id_space_list: typing.List[int],
    ) -> typing.List[typing.List[float]]:
        emb_vec = [None for _ in range(num_keys)]
        for idx in range(num_offset - 1):
            idx_start = offset[idx]
            idx_end = offset[idx + 1]
            id_space = id_space_list[idx]
            emb_table = self.emb_table_dict[id_space]
            select_key_in_this_id_space = [keys[idx_key] for idx_key in range(idx_start, idx_end)]

            emb_vec_in_this_id_space = emb_table.lookup(select_key_in_this_id_space, idx_end - idx_start, None, None, None)
            for i, ev in enumerate(emb_vec_in_this_id_space):
                emb_vec[idx_start + i] = ev
        return emb_vec

# creator of embedding table for multiple use case
class EmbeddingTableCreator:
    # this is used for create table to be consistent with embedding
    @staticmethod
    def create_static_embedding_table(gpu_id, embedding_collection_param: EmbeddingCollectionParam, embedding_table_param_list: typing.List[EmbeddingTableParam], global_embedding_sharding_param_list: typing.List[typing.List[EmbeddingShardingParam]]):
        local_embedding_sharding_param_list = global_embedding_sharding_param_list[gpu_id]

        embedding_table_list = []
        for local_embedding_sharding_param in local_embedding_sharding_param_list:
            local_embedding_list = local_embedding_sharding_param.local_embedding_list
            num_sharding = local_embedding_sharding_param.num_sharding

            local_emb_table_param_list = []
            local_emb_table_id_space_set = set()
            for embedding_id in local_embedding_list:
                id_space = embedding_collection_param.embedding_params[embedding_id].id_space
                emb_table_param = embedding_table_param_list[id_space]
                assert emb_table_param.vocabulary_size != -1
                if id_space not in local_emb_table_id_space_set:
                    local_emb_table_param_list.append(EmbeddingTableParam(
                        id_space=id_space,
                        vocabulary_size=emb_table_param.vocabulary_size // num_sharding,
                        ev_size=emb_table_param.ev_size
                    ))
                    local_emb_table_id_space_set.add(id_space)
            group_static_embedding_table = GroupedStaticEmbeddingTable(local_emb_table_param_list)
            embedding_table_list.append(group_static_embedding_table)
        return embedding_table_list
    
    # this is used for grouping single table to be consistent with embedding
    @staticmethod
    def group_embedding_table(gpu_id, emb_tables, embedding_collection_param: EmbeddingCollectionParam, global_embedding_sharding_param_list: typing.List[typing.List[EmbeddingParam]]):
        local_embedding_sharding_param_list = global_embedding_sharding_param_list[gpu_id]
        
        embedding_table_list = []
        for local_embedding_sharding_param in local_embedding_sharding_param_list:
            local_embedding_list = local_embedding_sharding_param.local_embedding_list

            id_space_set = set()
            for embedding_id in local_embedding_list:
                id_space_set.add(embedding_collection_param.embedding_params[embedding_id].id_space)
            id_space_set = sorted(list(id_space_set))
            embedding_table_list.append(GroupedEmbeddingTable(emb_tables_dict={id_space: emb_tables[id_space] for id_space in id_space_set}))
        return embedding_table_list

    # this is used for extend fused table to be consistent with embedding
    @staticmethod
    def extend_embedding_table(gpu_id, emb_table, global_embedding_sharding_param_list: typing.List[typing.List[EmbeddingShardingParam]]):
        local_embedding_sharding_param_list = global_embedding_sharding_param_list[gpu_id]

        embedding_table_list = []
        for _ in len(local_embedding_sharding_param_list):
            embedding_table_list.append(emb_table)
        return embedding_table_list