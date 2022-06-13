from ast import arg
from concurrent.futures import thread
import typing
from itertools import accumulate, chain
import collections
import threading
from embedding import *
from common import *
from embedding_table import *



class Model:
    def __init__(self, embedding_planner: EmbeddingPlanner, num_gpus) -> None:
        self.num_gpus = num_gpus
        self.gpus = [i for i in range(self.num_gpus)]
        self.embedding_planner = embedding_planner
        self.embedding_planner.generate_embedding_plan('localized')
        self.embedding_collection_list = [self.embedding_planner.create_embedding_collection(gpu_id) for gpu_id in self.gpus]

    def forward(self, key, bucket_range, embedding_tables, output_buffer):
        num_output_buffer_per_gpu = len(output_buffer) // self.num_gpus

        def forward_per_gpu(gpu_id):
            current_output_buffer = [
                0 for _ in range(num_output_buffer_per_gpu)]
            self.embedding_collection_list[gpu_id].forward_per_gpu(
                key, bucket_range, embedding_tables[gpu_id], current_output_buffer)
            output_buffer[num_output_buffer_per_gpu * gpu_id: num_output_buffer_per_gpu * (gpu_id + 1)] = current_output_buffer

        threads = []
        for gpu_id in self.gpus:
            t = threading.Thread(target=forward_per_gpu, args=(gpu_id, ), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def backward(self, top_grad, grad_key, grad_key_offset, grad_ev, grad_ev_offset, grad_id_space_list, do_allreduce: bool):
        def backward_per_gpu(gpu_id):
            self.embedding_collection_list[gpu_id].backward_per_gpu(
                top_grad[gpu_id], grad_key[gpu_id], grad_key_offset[gpu_id], grad_ev[gpu_id], grad_ev_offset[gpu_id], grad_id_space_list[gpu_id], do_allreduce
            )

        threads = []
        for gpu_id in self.gpus:
            t = threading.Thread(target=backward_per_gpu, args=(gpu_id,), daemon=True)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

class RefEmbedding:
    def __init__(self, embedding_collection_param: EmbeddingCollectionParam, num_gpus) -> None:
        self.num_gpus = num_gpus
        self.gpus = [i for i in range(self.num_gpus)]
        self.embedding_collection_param = embedding_collection_param

    def forward(self, keys, bucket_range, emb_table):
        batch_size = (len(bucket_range) - 1) // self.embedding_collection_param.num_embeddings
        batch_size_per_gpu = batch_size // self.num_gpus
        embedding_params = self.embedding_collection_param.embedding_params
        output = []
        for gpu_id in range(self.num_gpus):
            for embedding_id in range(self.embedding_collection_param.num_embeddings):
                ev_size = embedding_params[embedding_id].ev_size
                combiner = embedding_params[embedding_id].combiner
                id_space = embedding_params[embedding_id].id_space
                for batch_id in range(batch_size_per_gpu):
                    start = bucket_range[embedding_id * batch_size + gpu_id * batch_size_per_gpu +  batch_id]
                    end = bucket_range[embedding_id * batch_size + gpu_id * batch_size_per_gpu +  batch_id + 1]
                    ev = [0. for _ in range(ev_size)]
                    for r in range(start, end):
                        key = keys[r]
                        for ev_id in range(ev_size):
                            ev[ev_id] += emb_table[id_space][key][ev_id]
                    if combiner == 'mean' and end > start:
                        ev = [ e / (end - start) for e in ev]
                    if combiner == 'concat':
                        raise RuntimeError('not implemented')
                    output.extend(ev)
        return output

    def backward(self, top_grad, keys, bucket_range):
        batch_size = (len(bucket_range) - 1) // self.embedding_collection_param.num_embeddings
        batch_size_per_gpu = batch_size // self.num_gpus
        embedding_params = self.embedding_collection_param.embedding_params
        ev_size_offset_list = list(accumulate([0] + [b.ev_size for b in embedding_params]))
        grad = defaultdict(dict)
        for gpu_id in range(self.num_gpus):
            for embedding_id in range(self.embedding_collection_param.num_embeddings):
                ev_size = embedding_params[embedding_id].ev_size
                ev_size_offset = batch_size_per_gpu * ev_size_offset_list[embedding_id]
                id_space = embedding_params[embedding_id].id_space

                for batch_id in range(batch_size_per_gpu):
                    start = bucket_range[embedding_id * batch_size + gpu_id * batch_size_per_gpu +  batch_id]
                    end = bucket_range[embedding_id * batch_size + gpu_id * batch_size_per_gpu +  batch_id + 1]
                    for r in range(start, end):
                        key = keys[r]
                        if key not in grad[id_space]:
                            grad[id_space][key] = [0. for _ in range(ev_size)]
                        ev = top_grad[gpu_id][ev_size_offset + batch_id * ev_size: ev_size_offset + (batch_id + 1) * ev_size]
                        for ev_id in range(ev_size):
                            grad[id_space][key][ev_id] += ev[ev_id]

        return grad


key = [
    [4],
    [6, 8, 9, 10],
    [4, 5],
    [1, 4, 5],
    [1, 2],
    [3],
    [4],
    [6, 8],
    [5],
    [6],
    [7],
    [1, 9],
    [4, 6, 8, 9],
    [1, 2, 3, 4],
    [8, 9, 1, 10],
    [1, 10],
    [4, 5],
    [3, 6],
    [1, 10],
    [11]
]
bucketcount = [len(v) for v in key]
bucket_range = [0] + list(accumulate([len(v) for v in key]))
key = list(chain.from_iterable(key))
print('key', key)
print('bucket_range', bucket_range)
num_embeddings = 5
num_gpus = 2
batch_size = len(bucket_range) // num_embeddings
assert(batch_size % num_gpus == 0)
print('batch_size', batch_size)


num_table = 4
embedding_vocabulary_size_list = [100, 100, 100, 100]
embedding_tables_ev_size = [8, 16, 8, 8]
embedding_table_params = [EmbeddingTableParam(
    id_space=id_space,
    vocabulary_size=embedding_vocabulary_size_list[id_space],
    ev_size=embedding_tables_ev_size[id_space]
) for id_space in range(num_table)]

embedding_id_space = [0, 1, 2, 1, 3]
embedding_combiner = ['mean', 'sum', 'sum', 'sum', 'sum']
embedding_hotness = [max(bucketcount[batch_size * i: batch_size * (i + 1)])
                  for i in range(num_embeddings)]

embedding_collection_param = EmbeddingCollectionParam(
    num_embeddings=num_embeddings,
    embedding_params=[EmbeddingParam(
        embedding_id=i,
        id_space=embedding_id_space[i],
        combiner=embedding_combiner[i],
        hotness=embedding_hotness[i],
        ev_size=embedding_table_params[embedding_id_space[i]].ev_size,
    ) for i in range(num_embeddings)],
    universal_batch_size=batch_size,
    key_type='int64_t',
    offset_type='uint32_t',
    emb_type='float'
)


planner = EmbeddingPlanner(num_gpus, embedding_collection_param)
# planner.generate_embedding_plan_from_json_file('./plan_file_0.json')
# planner.generate_embedding_plan_from_json_file('./plan_file_1.json')
planner.generate_embedding_plan_from_json_file('./plan_file_2.json')

emb_tables = []
for gpu_id in range(num_gpus):
    emb_tables.append(EmbeddingTableCreator.create_static_embedding_table(
        gpu_id,
        embedding_collection_param,
        embedding_table_params,
        planner.global_embedding_sharding_param_list
    ))

init_keys = [
    [key for key in range(embedding_vocabulary_size_list[i])] for i in range(num_table)
]

global_init_key(num_gpus, planner.global_embedding_sharding_param_list, emb_tables, embedding_collection_param, init_keys)

model = Model(planner, num_gpus)

num_output_elements = batch_size * \
    sum([b.ev_size if b.combiner != 'concat' else b.ev_size *
        b.hotness for b in embedding_collection_param.embedding_params])
output_buffer = [0 for _ in range(num_output_elements)]
model.forward(key, bucket_range, emb_tables, output_buffer)

top_grad = [
    [random.random() for i in range(num_output_elements // num_gpus)] for gpu_id in range(num_gpus)
]

ref_embedding_table = global_dump_emb_table(num_gpus, planner.global_embedding_sharding_param_list, emb_tables)

# print('ref_embedding_table', ref_embedding_table)
ref_embedding = RefEmbedding(embedding_collection_param, num_gpus)
ref_output = ref_embedding.forward(key, bucket_range, ref_embedding_table)
ref_grad = ref_embedding.backward(top_grad, key, bucket_range)

assert len(ref_output) == len(output_buffer), '{} != {} {}'.format(len(ref_output), len(output_buffer), num_output_elements)
for i in range(len(ref_output)):
    if abs(ref_output[i] - output_buffer[i]) > 1e-3:
        print('forward check', i, ref_output[i], output_buffer[i])
    assert abs(ref_output[i] - output_buffer[i]) < 1e-3
print('pass forward check')

grad_key = []
grad_key_offset = []
grad_ev = []
grad_ev_offset = []
grad_id_space_list = []
for gpu_id in range(num_gpus):
    grad_key.append([])
    grad_key_offset.append([0])
    grad_ev.append([])
    grad_ev_offset.append([0])
    grad_id_space_list.append([])

model.backward(top_grad, grad_key, grad_key_offset, grad_ev, grad_ev_offset, grad_id_space_list, True)

# print("grad_id_space_list", grad_id_space_list)
# print("grad_key", grad_key)
# print("grad_key_offset", grad_key_offset)
# print("grad_ev", grad_ev)
# print("grad_ev_offset", grad_ev_offset)

dp_id_space = [3]
grad = defaultdict(dict)
for gpu_id in range(num_gpus):
    local_grad_key = grad_key[gpu_id]
    local_grad_key_offset = grad_key_offset[gpu_id]
    local_grad_ev = grad_ev[gpu_id]
    local_grad_ev_offset = grad_ev_offset[gpu_id]
    local_grad_id_space_list = grad_id_space_list[gpu_id]
    for idx in range(len(local_grad_key_offset) - 1):
        start = local_grad_key_offset[idx]
        end = local_grad_key_offset[idx + 1]
        id_space = local_grad_id_space_list[idx]
        if id_space in dp_id_space and gpu_id > 0:
            continue
        ev_size = embedding_tables_ev_size[id_space]
        for i in range(start, end):
            key = local_grad_key[i]
            ev_grad = local_grad_ev[local_grad_ev_offset[idx] + (i - start) * ev_size: local_grad_ev_offset[idx] + (i - start + 1) * ev_size]
            # grad[id_space][key] = ev_grad
            if key not in grad[id_space]:
                grad[id_space][key] = [0. for _ in range(ev_size)]
            for ev_id in range(ev_size):
                grad[id_space][key][ev_id] += ev_grad[ev_id]


for id_space in ref_grad:
    for key in ref_grad[id_space]:
        ref_ev_grad = ref_grad[id_space][key]
        ev_grad = grad[id_space][key]
        for i in range(len(ref_ev_grad)):
            assert abs(ref_ev_grad[i] - ev_grad[i]) < 1e-3
print('pass backward check')
