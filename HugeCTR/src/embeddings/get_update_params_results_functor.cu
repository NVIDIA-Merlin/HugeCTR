/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <embeddings/sparse_embedding_functors.hpp>
#include <utils.hpp>

namespace HugeCTR {

template <typename TypeHashKey>
void SparseEmbeddingFunctors::get_update_params_results(
    size_t embedding_vec_size, size_t vocabulary_size,
    const Tensors2<float> &hash_table_value_tensors,
    const std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables,
    Tensor2<TypeHashKey> &hash_table_key, Tensor2<float> &hash_table_value,
    const ResourceManager &resource_manager) {
  CudaDeviceContext context;

  size_t local_gpu_count = resource_manager.get_local_gpu_count();

  // memory allocation
  std::unique_ptr<size_t[]> count(new size_t[local_gpu_count]);
  size_t total_count = 0;
  for (size_t id = 0; id < local_gpu_count; id++) {
    const auto &local_gpu = resource_manager.get_local_gpu(id);
    context.set_device(local_gpu->get_device_id());
    if ((count[id] = hash_tables[id]->get_value_head(local_gpu->get_stream())) !=
        hash_tables[id]->get_size(local_gpu->get_stream())) {
      HCTR_LOG_S(ERROR, WORLD) << "hashtable: get_value_head()="
                               << hash_tables[id]->get_value_head(local_gpu->get_stream())
                               << ", get_size()="
                               << hash_tables[id]->get_size(local_gpu->get_stream()) << std::endl;
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Error: hash_table get_value_head() size not equal to get_size()");
    }
    total_count += count[id];

#ifndef NDEBUG
    HCTR_LOG_S(DEBUG, WORLD) << "GPU[" << id << "]: number of <key,value> pairs:" << count[id]
                             << std::endl;
#endif
  }

#ifndef NDEBUG
  HCTR_LOG_S(DEBUG, WORLD) << "Total number of <key,value> pairs:" << total_count << std::endl;
#endif

  if (total_count > (size_t)vocabulary_size) {
    HCTR_OWN_THROW(Error_t::WrongInput,
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

    context.set_device(resource_manager.get_local_gpu(id)->get_device_id());

    HCTR_LIB_THROW(cudaMalloc(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey)));
    HCTR_LIB_THROW(cudaMalloc(&d_hash_table_value_index[id], count[id] * sizeof(size_t)));
    HCTR_LIB_THROW(
        cudaMalloc(&d_hash_table_value[id], count[id] * embedding_vec_size * sizeof(float)));
    HCTR_LIB_THROW(cudaMalloc(&d_dump_counter[id], count[id] * sizeof(size_t)));
  }

  // dump hash table on GPU
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) {
      continue;
    }

    const auto &local_gpu = resource_manager.get_local_gpu(id);
    context.set_device(local_gpu->get_device_id());

    hash_tables[id]->dump(d_hash_table_key[id], d_hash_table_value_index[id], d_dump_counter[id],
                          local_gpu->get_stream());

    get_hash_value(count[id], embedding_vec_size, d_hash_table_value_index[id],
                   hash_table_value_tensors[id].get_ptr(), d_hash_table_value[id],
                   local_gpu->get_stream());
  }

  // sync wait
  sync_all_gpus(resource_manager);

  // memcpy from GPU to CPU memory
  size_t key_offset = 0;
  size_t value_offset = 0;
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) {
      continue;
    }

    context.set_device(resource_manager.get_local_gpu(id)->get_device_id());

    HCTR_LIB_THROW(cudaMemcpy(hash_table_key.get_ptr() + key_offset, d_hash_table_key[id],
                              count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost));
    key_offset += count[id];

    HCTR_LIB_THROW(cudaMemcpy(hash_table_value.get_ptr() + value_offset, d_hash_table_value[id],
                              count[id] * embedding_vec_size * sizeof(float),
                              cudaMemcpyDeviceToHost));
    value_offset += count[id] * embedding_vec_size;
  }

  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) {
      continue;
    }

    context.set_device(resource_manager.get_local_gpu(id)->get_device_id());

    HCTR_LIB_THROW(cudaFree(d_hash_table_key[id]));
    HCTR_LIB_THROW(cudaFree(d_hash_table_value_index[id]));
    HCTR_LIB_THROW(cudaFree(d_hash_table_value[id]));
    HCTR_LIB_THROW(cudaFree(d_dump_counter[id]));
  }

#ifdef ENABLE_MPI

  if (resource_manager.get_num_process() > 1) {
    constexpr int32_t key_size{sizeof(TypeHashKey)};
    const int32_t val_size{static_cast<int32_t>(embedding_vec_size * sizeof(float))};

    // TODO: Allow updates larger than 2B KVs. MPI_Gather uses MPI_INT32_T to accommodate
    // MPI_Gatherv. This check ensures that the multiplications below don't overflow.
    HCTR_CHECK(total_count <= static_cast<size_t>(std::max(key_size, val_size)) *
                                  std::numeric_limits<int32_t>::max());

    // Count results.
    std::vector<int32_t> recv_cnts;
    if (resource_manager.is_master_process()) {
      recv_cnts.resize(resource_manager.get_num_process());
    }
    HCTR_MPI_THROW(MPI_Gather(&total_count, 1, MPI_INT32_T,      // src
                              recv_cnts.data(), 1, MPI_INT32_T,  // dst (root only)
                              0, MPI_COMM_WORLD));

    // Download keys.
    std::vector<int32_t> displs(recv_cnts.size());
    if (!recv_cnts.empty()) {
      int32_t displ{};
      for (size_t i{}; i != recv_cnts.size(); ++i) {
        displs[i] = displ;

        const int32_t recv_cnt{recv_cnts[i] * key_size};
        recv_cnts[i] = recv_cnt;

        displ += recv_cnt;
        // TODO: Allow updates > 2B x MPI_CHAR. Here, we get a 32 bit overflow in this case.
        HCTR_CHECK(displ >= 0);
      }
    }

    HCTR_MPI_THROW(MPI_Gatherv(hash_table_key.get_ptr(), total_count * key_size, MPI_CHAR,  // src
                               hash_table_key.get_ptr(), recv_cnts.data(), displs.data(),
                               MPI_CHAR,  // dst (root only)
                               0, MPI_COMM_WORLD));

    // Download values.
    if (!recv_cnts.empty()) {
      int32_t displ{};
      for (size_t i{}; i != recv_cnts.size(); ++i) {
        displs[i] = displ;

        const int32_t recv_cnt{recv_cnts[i] / key_size * val_size};
        recv_cnts[i] = recv_cnt;

        displ += recv_cnt;
        // TODO: Allow updates > 2B x MPI_CHAR. Here, we get a 32 bit overflow in this case.
        HCTR_CHECK(displ >= 0);
      }
    }

    HCTR_MPI_THROW(MPI_Gatherv(hash_table_value.get_ptr(), total_count * val_size,
                               MPI_CHAR,  // src
                               hash_table_value.get_ptr(), recv_cnts.data(), displs.data(),
                               MPI_CHAR,  // dst (root only)
                               0, MPI_COMM_WORLD));
  }
#endif

  return;
}

template void SparseEmbeddingFunctors::get_update_params_results<unsigned int>(
    size_t embedding_vec_size, size_t vocabulary_size,
    const Tensors2<float> &hash_table_value_tensors,
    const std::vector<std::shared_ptr<HashTable<unsigned int, size_t>>> &hash_tables,
    Tensor2<unsigned int> &hash_table_key, Tensor2<float> &hash_table_value,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::get_update_params_results<long long>(
    size_t embedding_vec_size, size_t vocabulary_size,
    const Tensors2<float> &hash_table_value_tensors,
    const std::vector<std::shared_ptr<HashTable<long long, size_t>>> &hash_tables,
    Tensor2<long long> &hash_table_key, Tensor2<float> &hash_table_value,
    const ResourceManager &resource_manager);

}  // namespace HugeCTR