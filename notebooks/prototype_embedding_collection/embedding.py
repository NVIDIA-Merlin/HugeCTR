import typing

from common import *
from op import *
from embedding_table import *
import json

class EmbeddingPlanner:
    def __init__(self, num_gpus, embedding_collection_param: EmbeddingCollectionParam) -> None:
        self.num_gpus = num_gpus
        self.embedding_collection_param = embedding_collection_param
        self.global_embedding_sharding_param_list = [[] for _ in range(num_gpus)]

    def generate_embedding_plan_from_json_file(self, plan_file):
        with open(plan_file, 'r') as f:
            plan = json.load(f)
        assert len(plan) == self.num_gpus
        for gpu_id in range(self.num_gpus):
            local_plan = plan[gpu_id]
            for embedding_plan in local_plan:
                local_embedding_list = embedding_plan['local_embedding_list']
                global_embedding_list = embedding_plan['global_embedding_list']
                sharding_id = 0 if 'sharding_id' not in embedding_plan else embedding_plan['sharding_id']
                num_sharding = 1 if 'num_sharding' not in embedding_plan else embedding_plan['num_sharding']
                table_placement_strategy = embedding_plan['table_placement_strategy']
                if table_placement_strategy == 'dp' and num_sharding > 1:
                    raise RuntimeError("dp embedding can not shard")
                self.global_embedding_sharding_param_list[gpu_id].append(
                    EmbeddingShardingParam(
                        local_embedding_list=local_embedding_list,
                        global_embedding_list=global_embedding_list,
                        sharding_id=sharding_id,
                        num_sharding=num_sharding,
                        table_placement_strategy=table_placement_strategy
                    )
                )
        print('global embedding plan:', self.global_embedding_sharding_param_list)
        # we should add some check to varify the effectiveness of plan file

    def generate_embedding_plan(self, strategy):
        id_space_list = list(set([p.id_space for p in self.embedding_collection_param.embedding_params]))

        def generate_dp_embedding_sharding_params():
            embedding_list = []
            for embedding_param in self.embedding_collection_param.embedding_params:
                if embedding_param.id_space in id_space_list:
                    embedding_list.append(embedding_param.embedding_id)
            embedding_list = sorted(embedding_list)
            return [
                EmbeddingShardingParam(
                    local_embedding_list=embedding_list,
                    global_embedding_list=[embedding_list for _ in range(self.num_gpus)],
                    sharding_id=0,
                    num_sharding=1,
                    table_placement_strategy="dp",
                )
                for i in range(self.num_gpus)
            ]

        def generate_mp_embedding_sharding_params():
            global_embedding_list = [[] for _ in range(self.num_gpus)]
            for id_space_idx, id_space in enumerate(id_space_list):
                gpu_id = id_space_idx % self.num_gpus

                for embedding_param in self.embedding_collection_param.embedding_params:
                    if embedding_param.id_space == id_space:
                        global_embedding_list[gpu_id].append(embedding_param.embedding_id)
                global_embedding_list[gpu_id] = sorted(global_embedding_list[gpu_id])

            return [
                EmbeddingShardingParam(
                    local_embedding_list=global_embedding_list[i],
                    global_embedding_list=global_embedding_list,
                    sharding_id=0,
                    num_sharding=1,
                    table_placement_strategy="localized",
                )
                for i in range(self.num_gpus)
            ]
        
        if strategy == 'dp':
            embedding_sharding_params = generate_dp_embedding_sharding_params()
        elif strategy == 'localized':
            embedding_sharding_params = generate_mp_embedding_sharding_params()

        for gpu_id in range(self.num_gpus):
            self.global_embedding_sharding_param_list.append(embedding_sharding_params[gpu_id])

    def create_embedding_collection(self, gpu_id):
        return EmbeddingCollection(self.embedding_collection_param, self.global_embedding_sharding_param_list[gpu_id], gpu_id, self.num_gpus)


class LocalizedEmbedding:
    def __init__(
        self, gpu_id, num_gpus, embedding_collection_param: EmbeddingCollectionParam, embedding_sharding_param: EmbeddingShardingParam
    ) -> None:
        # param init
        self.gpu_id = gpu_id
        self.num_gpus = num_gpus
        self.embedding_collection_param = embedding_collection_param
        self.embedding_sharding_param = embedding_sharding_param
        self.embedding_params = embedding_collection_param.embedding_params
        self.universal_batch_size = embedding_collection_param.universal_batch_size
        self.current_embedding_params = [self.embedding_params[b] for b in self.embedding_sharding_param.local_embedding_list]
        self.id_space_list = [
            self.embedding_collection_param.embedding_params[b].id_space for b in self.embedding_sharding_param.local_embedding_list
        ]

        # on host
        self.h_local_embedding_list = self.embedding_sharding_param.local_embedding_list
        self.h_local_hotness_list = [self.embedding_params[b].hotness for b in self.embedding_sharding_param.local_embedding_list]
        self.h_hotness_list = [b.hotness for b in self.embedding_params]
        self.h_local_ev_size_list = [self.embedding_params[b].ev_size for b in self.embedding_sharding_param.local_embedding_list]
        self.h_local_ev_size_offset = list(accumulate([0] + self.h_local_ev_size_list))
        self.h_ev_size_list = [b.ev_size for b in self.embedding_params]
        self.h_ev_size_offset = list(accumulate([0] + self.h_ev_size_list))
        self.h_local_combiner_list = [self.embedding_params[b].combiner for b in self.embedding_sharding_param.local_embedding_list]
        self.h_combiner_list = [b.combiner for b in self.embedding_params]

        # on device
        self.d_local_embedding_list = self.h_local_embedding_list
        self.d_local_hotness_list = self.h_local_hotness_list
        self.d_hotness_list = self.h_hotness_list
        self.d_local_ev_size_offset = self.h_local_ev_size_offset
        self.d_ev_size_offset = self.h_ev_size_offset
        self.d_local_combiner_list = self.h_local_combiner_list
        self.d_combiner_list = self.h_combiner_list

        assert not has_concat_embedding(
            [self.embedding_collection_param.embedding_params[i] for i in self.embedding_sharding_param.local_embedding_list]
        ), "localized embedding not support concat conbiner"

        # op init
        self.model_index_calculation = ModelIndexCalcualtion(
            len(self.h_local_embedding_list),
            self.h_local_hotness_list,
            self.h_hotness_list,
            self.universal_batch_size,
            self.embedding_collection_param.key_type,
            self.embedding_collection_param.offset_type
        )
        self.network_index_calculation = NetworkIndexCalculation()
        self.model_backward_index_calculation = ModelBackwardIndexCalculation(
            num_gpus,
            len(self.h_local_embedding_list),
            self.h_local_hotness_list,
            self.universal_batch_size,
            self.embedding_collection_param.key_type,
            self.embedding_collection_param.offset_type
        )
        self.id_space_list = [
            embedding_collection_param.embedding_params[embedding_id].id_space
            for embedding_id in embedding_sharding_param.local_embedding_list
        ]
        self.compress_offset = CompressOffset(len(self.embedding_sharding_param.local_embedding_list))
        self.model_forward = ModelForward(num_gpus, self.embedding_sharding_param.local_embedding_list)
        self.all2all_comm = All2All(gpu_id, num_gpus)
        self.network_forward = NetworkForward(num_gpus)

        self.network_backward = NetworkBackward(num_gpus)

        self.model_backward = ModelBackward(
            num_gpus,
            len(self.h_local_embedding_list),
            self.h_local_hotness_list,
            self.h_local_ev_size_list,
            self.universal_batch_size,
            self.embedding_collection_param.emb_type
        )
        # buffer init
        self.model_comm_buffer = self.init_model_comm_buffer()

        self.network_comm_buffer = self.init_network_comm_buffer()

        (
            self.dst_embedding_id_list,
            self.num_dst_embedding_id,
            self.network_idx,
            self.network_gpu_idx,
            self.network_offset,
        ) = self.network_index_calculation.compute(
            self.num_gpus, self.embedding_sharding_param.global_embedding_list, self.h_ev_size_offset
        )

    def init_model_comm_buffer(self):
        model_comm_buffer_size = self.get_model_comm_buffer_size(self.universal_batch_size)

        return [[0.0 for _ in range(size)] for size in model_comm_buffer_size if self.embedding_collection_param.emb_type == 'float' or self.embedding_collection_param.emb_type == 'half']

    def init_network_comm_buffer(self):
        network_comm_buffer_size = self.get_network_comm_buffer_size(self.universal_batch_size)

        return [[0.0 for _ in range(size)] for size in network_comm_buffer_size if self.embedding_collection_param.emb_type == 'float' or self.embedding_collection_param.emb_type == 'half']

    def get_model_comm_buffer_size(self, batch_size):
        num_buffer_size = 0
        batch_size_per_gpu = batch_size // self.num_gpus
        for embedding_id in self.embedding_sharding_param.local_embedding_list:
            embedding_param = self.embedding_collection_param.embedding_params[embedding_id]
            ev_size = embedding_param.ev_size
            num_buffer_size += ev_size * batch_size_per_gpu
        return [num_buffer_size for _ in range(self.num_gpus)]

    def get_network_comm_buffer_size(self, batch_size):
        batch_size_per_gpu = batch_size // self.num_gpus
        network_comm_buffer_size = []
        for gpu_id in range(self.num_gpus):
            remote_embedding_list = self.embedding_sharding_param.global_embedding_list[gpu_id]
            num_buffer_size = 0
            for embedding_id in remote_embedding_list:
                embedding_param = self.embedding_collection_param.embedding_params[embedding_id]
                num_buffer_size += embedding_param.ev_size * batch_size_per_gpu
            network_comm_buffer_size.append(num_buffer_size)
        return network_comm_buffer_size

    def index_calculation_per_gpu(self, key, bucket_range):
        self.batch_size = (len(bucket_range) - 1) // self.embedding_collection_param.num_embeddings
        self.model_key, self.model_offsets, self.num_model_key, self.num_key_in_bucket_for_combiner = self.model_index_calculation.compute(
            key,
            bucket_range,
            self.d_local_embedding_list,
            self.embedding_sharding_param.sharding_id,
            self.embedding_sharding_param.num_sharding,
            self.batch_size,
        )
        self.id_space_offset = self.compress_offset.compute(self.model_offsets, self.batch_size)
        (
            self.unique_key,
            self.num_unique_key,
            self.unique_key_id_space_offset,
            self.unique_key_bucket_idx,
            self.unique_key_bucket_idx_offset,
            self.unique_key_ev_size_offset,
            self.unique_key_ev_id_space_offset,
        ) = self.model_backward_index_calculation.compute(
            self.model_key,
            self.num_model_key,
            self.model_offsets,
            self.id_space_offset,
            self.d_local_ev_size_offset,
            self.batch_size,
        )
        # print(
        #     "localized index_calculation_per_gpu",
        #     self.gpu_id,
        #     self.model_key,
        #     self.model_offsets,
        #     self.id_space_offset,
        #     self.h_local_combiner_list,
        #     self.h_local_ev_size_list,
        # )

    def model_forward_per_gpu(self, embedding_table):
        self.mp_ev = embedding_table.lookup(
            self.model_key,
            self.num_model_key,
            self.id_space_offset,
            len(self.embedding_sharding_param.local_embedding_list) + 1,
            self.id_space_list,
        )
        self.model_forward.compute(
            self.mp_ev,
            self.model_offsets,
            self.num_key_in_bucket_for_combiner,
            self.model_comm_buffer,
            self.d_local_combiner_list,
            self.d_local_ev_size_offset,
            self.batch_size,
        )
        # print("localized model_forward_per_gpu", self.gpu_id, self.model_comm_buffer)

    def forward_communication_per_gpu(self):
        self.model_comm_buffer_size = self.get_model_comm_buffer_size(self.batch_size)
        self.network_comm_buffer_size = self.get_network_comm_buffer_size(self.batch_size)
        self.all2all_comm.communication(
            self.model_comm_buffer, self.model_comm_buffer_size, self.network_comm_buffer, self.network_comm_buffer_size
        )
        # print("localized forward_communication_per_gpu", self.gpu_id, self.network_comm_buffer)

    def network_forward_per_gpu(self, output_buffer):
        self.network_forward.compute(
            self.network_comm_buffer,
            self.dst_embedding_id_list,
            self.num_dst_embedding_id,
            self.network_idx,
            self.network_gpu_idx,
            self.network_offset,
            output_buffer,
            self.d_combiner_list,
            self.d_ev_size_offset,
            self.batch_size,
        )

    def forward_per_gpu(self, key, bucket_range, embedding_table, output_buffer):
        self.index_calculation_per_gpu(key, bucket_range)
        self.model_forward_per_gpu(embedding_table)
        self.forward_communication_per_gpu()
        self.network_forward_per_gpu(output_buffer)

    def network_backward_per_gpu(self, top_grad):
        self.network_backward.compute(
            top_grad,
            self.network_comm_buffer,
            self.dst_embedding_id_list,
            self.num_dst_embedding_id,
            self.network_idx,
            self.network_gpu_idx,
            self.network_offset,
            self.d_ev_size_offset,
            self.batch_size,
        )
        # print("localized network_backward_per_gpu", self.gpu_id, self.network_comm_buffer)

    def backward_communication_per_gpu(self):
        self.all2all_comm.communication(
            self.network_comm_buffer, self.network_comm_buffer_size, self.model_comm_buffer, self.model_comm_buffer_size
        )
        # print("localized backward_communication_per_gpu", self.gpu_id, self.model_comm_buffer)

    def model_backward_per_gpu(self):
        self.grad_ev = self.model_backward.compute(
            self.model_comm_buffer,
            self.unique_key_ev_size_offset,
            self.unique_key_bucket_idx,
            self.unique_key_bucket_idx_offset,
            self.num_unique_key,
            self.d_local_ev_size_offset,
            self.batch_size,
        )
        return (
            self.unique_key,
            self.num_unique_key,
            self.unique_key_id_space_offset,
            self.grad_ev,
            self.unique_key_ev_id_space_offset,
        )

    def backward_per_gpu(self, top_grad, do_allreduce):
        self.network_backward_per_gpu(top_grad)
        self.backward_communication_per_gpu()
        return self.model_backward_per_gpu()


class DPEmbedding:
    def __init__(
        self, gpu_id, num_gpus, embedding_collection_param: EmbeddingCollectionParam, embedding_sharding_param: EmbeddingShardingParam
    ) -> None:
        self.gpu_id = gpu_id
        self.num_gpus = num_gpus
        self.embedding_params = embedding_collection_param.embedding_params
        self.embedding_collection_param = embedding_collection_param
        self.universal_batch_size = embedding_collection_param.universal_batch_size
        self.embedding_sharding_param = embedding_sharding_param

        # on host
        self.h_local_embedding_list = self.embedding_sharding_param.local_embedding_list
        self.h_local_hotness_list = [self.embedding_params[b].hotness for b in self.embedding_sharding_param.local_embedding_list]
        self.h_hotness_list = [b.hotness for b in self.embedding_params]
        self.h_local_ev_size_list = [self.embedding_params[b].ev_size for b in self.embedding_sharding_param.local_embedding_list]
        self.h_local_ev_size_offset = list(accumulate([0] + self.h_local_ev_size_list))
        self.h_ev_size_offset = list(accumulate([0] + [b.ev_size for b in self.embedding_params]))
        self.h_local_combiner_list = [self.embedding_params[b].combiner for b in self.embedding_sharding_param.local_embedding_list]
        self.h_combiner_list = [b.combiner for b in self.embedding_params]

        # on device
        self.d_local_embedding_list = self.h_local_embedding_list
        self.d_local_hotness_list = self.h_local_hotness_list
        self.d_hotness_list = self.h_hotness_list
        self.d_local_ev_size_offset = self.h_local_ev_size_offset
        self.d_ev_size_offset = self.h_ev_size_offset
        self.d_local_combiner_list = self.h_local_combiner_list
        self.d_combiner_list = self.h_combiner_list

        assert not has_concat_embedding(
            [self.embedding_collection_param.embedding_params[i] for i in self.embedding_sharding_param.local_embedding_list]
        ), "dp embedding not support concat conbiner"

        self.index_calculation = DPIndexCalculation(
            gpu_id,
            num_gpus,
            len(self.h_local_embedding_list),
            self.h_local_hotness_list,
            self.h_hotness_list,
            self.universal_batch_size,
            self.embedding_collection_param.key_type,
            self.embedding_collection_param.offset_type
        )
        self.dp_local_reduce_index_calculation = DPLocalReduceIndexCalculation(
            len(self.h_local_embedding_list),
            self.embedding_collection_param.num_embeddings,
            self.h_local_hotness_list,
            self.h_hotness_list,
            self.universal_batch_size,
            self.embedding_collection_param.key_type,
            self.embedding_collection_param.offset_type
        )
        self.id_space_list = [
            embedding_collection_param.embedding_params[embedding_id].id_space
            for embedding_id in embedding_sharding_param.local_embedding_list
        ]
        self.compress_offset = CompressOffset(len(self.embedding_sharding_param.local_embedding_list))
        self.dp_model_forward = DPModelForward(num_gpus, self.h_local_embedding_list)
        self.dp_local_reduce = DPLocalReduce(
            gpu_id,
            num_gpus,
            len(self.h_local_embedding_list),
            self.h_local_hotness_list,
            self.h_local_ev_size_list,
            self.embedding_collection_param.universal_batch_size,
            self.embedding_collection_param.emb_type,
        )
        self.all_reduce = AllreduceInplace(gpu_id, num_gpus)

    def forward_per_gpu(self, key, bucket_range, embedding_table, output_buffer):
        batch_size = (len(bucket_range) - 1) // self.embedding_collection_param.num_embeddings
        self.dp_key, self.dp_offset, self.num_dp_key = self.index_calculation.compute(
            key, bucket_range, self.d_local_embedding_list, batch_size
        )

        self.id_space_offset = self.compress_offset.compute(self.dp_offset, batch_size // self.num_gpus)
        (
            self.unique_key,
            self.num_unique_key,
            self.unique_key_id_space_offset,
            self.unique_key_bucket_idx,
            self.unique_key_bucket_idx_offset,
            self.unique_key_ev_size_offset,
            self.unique_key_ev_id_space_offset,
        ) = self.dp_local_reduce_index_calculation.compute(
            key, bucket_range, self.d_local_embedding_list, self.d_local_ev_size_offset, batch_size
        )

        self.dp_ev = embedding_table.lookup(
            self.dp_key,
            self.num_dp_key,
            self.id_space_offset,
            len(self.embedding_sharding_param.local_embedding_list) + 1,
            self.id_space_list,
        )
        self.dp_model_forward.compute(
            self.dp_ev,
            self.dp_offset,
            output_buffer,
            self.d_local_embedding_list,
            self.d_local_combiner_list,
            self.d_ev_size_offset,
            batch_size,
        )

    def backward_per_gpu(self, top_grad, do_allreduce):
        batch_size_per_gpu = len(top_grad) // sum([b.ev_size for b in self.embedding_collection_param.embedding_params])
        batch_size = batch_size_per_gpu * self.num_gpus
        self.grad_ev = self.dp_local_reduce.compute(
            top_grad,
            self.unique_key_ev_size_offset,
            self.unique_key_bucket_idx,
            self.unique_key_bucket_idx_offset,
            self.num_unique_key,
            self.d_ev_size_offset,
            batch_size,
        )
        if do_allreduce:
            self.all_reduce.communication(self.grad_ev, self.unique_key_ev_size_offset[self.num_unique_key])
        return (
            self.unique_key,
            self.num_unique_key,
            self.unique_key_id_space_offset,
            self.grad_ev,
            self.unique_key_ev_id_space_offset,
        )


class EmbeddingCollection:
    def __init__(
        self,
        embedding_collection_param: EmbeddingCollectionParam,
        embedding_sharding_params: typing.List[EmbeddingShardingParam],
        gpu_id,
        num_gpus,
    ) -> None:
        self.gpu_id = gpu_id
        self.num_gpus = num_gpus
        self.embedding_params = embedding_collection_param.embedding_params
        self.embedding_collection_param = embedding_collection_param
        self.embedding_sharding_params = embedding_sharding_params
        self.create_embedding()

    def create_embedding(self):
        self.embeddings = []
        for embedding_sharding_param in self.embedding_sharding_params:
            if embedding_sharding_param.table_placement_strategy == "dp":
                self.embeddings.append(
                    DPEmbedding(self.gpu_id, self.num_gpus, self.embedding_collection_param, embedding_sharding_param)
                )
            elif embedding_sharding_param.table_placement_strategy == "localized":
                self.embeddings.append(
                    LocalizedEmbedding(self.gpu_id, self.num_gpus, self.embedding_collection_param, embedding_sharding_param)
                )

    def forward_per_gpu(self, key, bucket_range, embedding_tables, output_buffer):
        for embedding_id in range(len(self.embedding_sharding_params)):
            embedding = self.embeddings[embedding_id]
            embedding.forward_per_gpu(key, bucket_range, embedding_tables[embedding_id], output_buffer)

    def backward_per_gpu(
        self, top_grad, grad_key, grad_key_offset, grad_ev, grad_ev_offset, grad_id_space_list, do_allreduce
    ):
        for embedding in self.embeddings:
            local_id_space_list = embedding.id_space_list
            grad_id_space_list.extend(local_id_space_list)
            (
                local_grad_key,
                num_local_grad_key,
                local_grad_key_offset,
                local_grad_ev,
                local_grad_ev_offset,
            ) = embedding.backward_per_gpu(top_grad, do_allreduce)
            # print("local_grad_key_offset", local_grad_key_offset)
            # print("local_grad_ev_offset", local_grad_ev_offset)

            grad_key.extend(local_grad_key[:num_local_grad_key])
            grad_ev.extend(local_grad_ev[: local_grad_ev_offset[-1]])
            for idx in range(len(local_id_space_list)):
                grad_key_offset.append(local_grad_key_offset[idx + 1] - local_grad_key_offset[idx])
                grad_ev_offset.append(local_grad_ev_offset[idx + 1] - local_grad_ev_offset[idx])

        # direct copy can not pass grad_key_offset to outside
        cumsum_grad_key_offset = list(accumulate(grad_key_offset))
        for i in range(len(cumsum_grad_key_offset)):
            grad_key_offset[i] = cumsum_grad_key_offset[i]
        cumsum_grad_ev_offset = list(accumulate(grad_ev_offset))
        for i in range(len(cumsum_grad_ev_offset)):
            grad_ev_offset[i] = cumsum_grad_ev_offset[i]
