/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"

namespace HugeCTR {

template <typename TypeHashKey>
void SparseEmbeddingFunctors::get_update_params_results(
    size_t embedding_vec_size, size_t vocabulary_size,
    const Tensors2<float> &hash_table_value_tensors,
    const std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables,
    Tensor2<TypeHashKey> &hash_table_key, Tensor2<float> &hash_table_value,
    const GPUResourceGroup &device_resources) {
  CudaDeviceContext context;

  size_t local_gpu_count = device_resources.size();

  // memory allocation
  std::unique_ptr<size_t[]> count(new size_t[local_gpu_count]);
  size_t total_count = 0;
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(device_resources[id].get_device_id());
    if ((count[id] = hash_tables[id]->get_value_head(device_resources[id].get_stream())) !=
        hash_tables[id]->get_size(device_resources[id].get_stream())) {
      std::cout << "hashtable: get_value_head()="
                << hash_tables[id]->get_value_head(device_resources[id].get_stream())
                << ", get_size()=" << hash_tables[id]->get_size(device_resources[id].get_stream())
                << std::endl;
      CK_THROW_(Error_t::WrongInput,
                "Error: hash_table get_value_head() size not equal to get_size()");
    }
    total_count += count[id];

#ifndef NDEBUG
    std::cout << "GPU[" << id << "]: number of <key,value> pairs:" << count[id] << std::endl;
#endif
  }

#ifndef NDEBUG
  std::cout << "Total number of <key,value> pairs:" << total_count << std::endl;
#endif

  if (total_count > (size_t)vocabulary_size) {
    CK_THROW_(Error_t::WrongInput,
              "Error: required download size is larger than hash table vocabulary_size");
  }

  std::unique_ptr<TypeHashKey *[]> d_hash_table_key(new TypeHashKey *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_hash_table_value_index(new size_t *[local_gpu_count]);
  std::unique_ptr<float *[]> d_hash_table_value(new float *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_dump_counter(new size_t *[local_gpu_count]);
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) {
      continue;
    }

    context.set_device(device_resources[id].get_device_id());

    cudaMalloc(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey));
    cudaMalloc(&d_hash_table_value_index[id], count[id] * sizeof(size_t));
    cudaMalloc(&d_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float));
    cudaMalloc(&d_dump_counter[id], count[id] * sizeof(size_t));
  }

  // dump hash table on GPU
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) {
      continue;
    }

    context.set_device(device_resources[id].get_device_id());

    hash_tables[id]->dump(d_hash_table_key[id], d_hash_table_value_index[id], d_dump_counter[id],
                          device_resources[id].get_stream());

    get_hash_value(count[id], embedding_vec_size, d_hash_table_value_index[id],
                   hash_table_value_tensors[id].get_ptr(), d_hash_table_value[id],
                   device_resources[id].get_stream());
  }

  // sync wait
  sync_all_gpus(device_resources);

  // memcpy from GPU to CPU memory
  size_t key_offset = 0;
  size_t value_offset = 0;
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) {
      continue;
    }

    context.set_device(device_resources[id].get_device_id());

    CK_CUDA_THROW_(cudaMemcpy(hash_table_key.get_ptr() + key_offset, d_hash_table_key[id],
                              count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost));
    key_offset += count[id];

    CK_CUDA_THROW_(cudaMemcpy(hash_table_value.get_ptr() + value_offset, d_hash_table_value[id],
                              count[id] * embedding_vec_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
    value_offset += count[id] * embedding_vec_size;
  }

  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) {
      continue;
    }

    context.set_device(device_resources[id].get_device_id());

    CK_CUDA_THROW_(cudaFree(d_hash_table_key[id]));
    CK_CUDA_THROW_(cudaFree(d_hash_table_value_index[id]));
    CK_CUDA_THROW_(cudaFree(d_hash_table_value[id]));
    CK_CUDA_THROW_(cudaFree(d_dump_counter[id]));
  }

#ifdef ENABLE_MPI
  int my_rank = 0;
  int n_ranks = 1;
  CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
  CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));

  if (n_ranks > 1) {
    std::unique_ptr<int> displs(new int(n_ranks));
    std::unique_ptr<int> recv_count(new int(n_ranks));
    MPI_Gather(&total_count, 1, MPI_INT, recv_count.get(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
      displs.get()[0] = 0;
      for (int i = 1; i < n_ranks; i++) {
        displs.get()[i] = displs.get()[i - 1] + recv_count.get()[i - 1];
      }
    }

    std::unique_ptr<int> displs_key(new int(n_ranks));
    std::unique_ptr<int> recv_count_key(new int(n_ranks));
    if (my_rank == 0) {
      for (int i = 0; i < n_ranks; i++) {
        recv_count_key.get()[i] = recv_count.get()[i] * sizeof(TypeHashKey);
        displs_key.get()[i] = displs.get()[i] * sizeof(TypeHashKey);
      }
    }

    MPI_Gatherv(hash_table_key.get_ptr(), total_count * sizeof(TypeHashKey), MPI_CHAR,
                hash_table_key.get_ptr(), recv_count_key.get(), displs_key.get(), MPI_CHAR, 0,
                MPI_COMM_WORLD);

    std::unique_ptr<int> displs_value(new int(n_ranks));
    std::unique_ptr<int> recv_count_value(new int(n_ranks));
    if (my_rank == 0) {
      for (int i = 0; i < n_ranks; i++) {
        recv_count_value.get()[i] = recv_count.get()[i] * embedding_vec_size * sizeof(float);
        displs_value.get()[i] = displs.get()[i] * embedding_vec_size * sizeof(float);
      }
    }

    MPI_Gatherv(hash_table_value.get_ptr(), total_count * embedding_vec_size * sizeof(float),
                MPI_CHAR, hash_table_value.get_ptr(), recv_count_value.get(), displs_value.get(),
                MPI_CHAR, 0, MPI_COMM_WORLD);
  }
#endif

  return;
}

template void SparseEmbeddingFunctors::get_update_params_results<unsigned int>(
    size_t embedding_vec_size, size_t vocabulary_size,
    const Tensors2<float> &hash_table_value_tensors,
    const std::vector<std::shared_ptr<HashTable<unsigned int, size_t>>> &hash_tables,
    Tensor2<unsigned int> &hash_table_key, Tensor2<float> &hash_table_value,
    const GPUResourceGroup &device_resources);

template void SparseEmbeddingFunctors::get_update_params_results<long long>(
    size_t embedding_vec_size, size_t vocabulary_size,
    const Tensors2<float> &hash_table_value_tensors,
    const std::vector<std::shared_ptr<HashTable<long long, size_t>>> &hash_tables,
    Tensor2<long long> &hash_table_key, Tensor2<float> &hash_table_value,
    const GPUResourceGroup &device_resources);

}  // namespace HugeCTR