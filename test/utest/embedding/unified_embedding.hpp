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
#pragma once

#include <gtest/gtest.h>

#include <cstring>
#include <data_generator.hpp>
#include <embedding.hpp>
#include <embeddings/distributed_slot_sparse_embedding_hash.hpp>
#include <embeddings/localized_slot_sparse_embedding_hash.hpp>
#include <embeddings/localized_slot_sparse_embedding_one_hot.hpp>
#include <embeddings/sparse_embedding_functors.hpp>
#include <memory>
#include <utest/embedding/cpu_hashtable.hpp>
#include <utest/test_utils.hpp>
#include <utils.hpp>

using namespace HugeCTR;

namespace unified_embedding_test {
template <typename T>
inline bool compare_array(size_t len, const T *a, const T *b, float epsilon) {
  for (size_t i = 0; i < len; i++) {
    if (fabs(a[i] - b[i]) >= epsilon) {
      HCTR_LOG(INFO, WORLD, "Error in compare_array: i=%zu, a=%.8f, b=%.8f\n", i, a[i], b[i]);
      return false;
    }
  }

  return true;
}

template <typename Type>
bool compare_host_and_device_array(Type *host_array, Type *device_array, size_t n) {
  std::unique_ptr<Type[]> h_array(new Type[n]);
  HCTR_LIB_THROW(cudaMemcpy(h_array.get(), device_array, sizeof(Type) * n, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n; ++i) {
    if (h_array[i] != host_array[i]) return false;
  }
  return true;
}

struct TestParams {
  size_t train_steps;
  size_t batch_size_per_gpu;
  size_t slot_num;
  size_t max_nnz_per_sample;
  std::vector<size_t> feature_num_per_sample;
  bool fixed_length;
  size_t vocabulary_size;
  size_t embedding_vec_size;
  float lr;
  bool reference_check;
  bool measure_performance;
  float eposilon;

  TestParams(size_t train_steps_, size_t batch_size_per_gpu_, size_t slot_num_,
             size_t max_nnz_per_sample_, bool fixed_length_, size_t vocabulary_size_,
             size_t embedding_vec_size_, float lr_, bool reference_check_,
             bool measure_performance_, float eposilon_)
      : train_steps(train_steps_),
        batch_size_per_gpu(batch_size_per_gpu_),
        slot_num(slot_num_),
        max_nnz_per_sample(max_nnz_per_sample_),
        feature_num_per_sample(slot_num_, max_nnz_per_sample_ / slot_num_),
        fixed_length(fixed_length_),
        vocabulary_size(vocabulary_size_),
        embedding_vec_size(embedding_vec_size_),
        lr(lr_),
        reference_check(reference_check_),
        measure_performance(measure_performance_),
        eposilon(eposilon_) {
    if (max_nnz_per_sample_ < slot_num_ || max_nnz_per_sample_ % slot_num_ != 0) {
      HCTR_OWN_THROW(Error_t::WrongInput, "test param is not illegal.");
    }
  }
};

template <typename KeyType>
void init_sparse_tensor(SparseTensor<KeyType> &sparse_tensor,
                        const std::vector<size_t> &feature_num_per_sample, bool on_device,
                        bool fixed_length) {
  IntUniformDataSimulator<KeyType> int_generator(0, 1);

  size_t batch_size = sparse_tensor.get_dimensions()[0];
  size_t slot_num = (sparse_tensor.rowoffset_count() - 1) / batch_size;

  std::vector<KeyType> value_vec;
  std::vector<KeyType> rowoffset_vec{0};
  KeyType value_count = 0;
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t s = 0; s < slot_num; ++s) {
      size_t feature_num_per_sample_per_slot = feature_num_per_sample[s];
      for (size_t n = 0; n < feature_num_per_sample_per_slot; ++n) {
        if (int_generator.get_num() > 0 || fixed_length) {
          value_vec.push_back(s * feature_num_per_sample_per_slot + n);
          ++value_count;
        }
      }
      rowoffset_vec.push_back(value_count);
    }
  }

  if (sparse_tensor.rowoffset_count() != rowoffset_vec.size()) {
    HCTR_OWN_THROW(Error_t::DataCheckError, "init sparse_tensor rowoffset count not match");
  }

  *sparse_tensor.get_nnz_ptr() = value_count;

  if (on_device) {
    HCTR_LIB_THROW(cudaMemcpy(sparse_tensor.get_value_ptr(), value_vec.data(),
                              sizeof(KeyType) * value_count, cudaMemcpyHostToDevice));
    HCTR_LIB_THROW(cudaMemcpy(sparse_tensor.get_rowoffset_ptr(), rowoffset_vec.data(),
                              sizeof(KeyType) * sparse_tensor.rowoffset_count(),
                              cudaMemcpyHostToDevice));
  } else {
    memcpy(sparse_tensor.get_value_ptr(), value_vec.data(), sizeof(KeyType) * value_count);
    memcpy(sparse_tensor.get_rowoffset_ptr(), rowoffset_vec.data(),
           sizeof(KeyType) * sparse_tensor.rowoffset_count());
  }
}

template <typename Type>
void all_gather_cpu(const SparseTensors<Type> &send_tensors, SparseTensors<Type> &recv_tensors) {
  const size_t rowoffset_count = send_tensors[0].rowoffset_count() - 1;

  size_t local_count = send_tensors.size();
  size_t global_count = recv_tensors.size();
  // do local all gather
  size_t local_total_nnz = 0;
  for (auto &t : send_tensors) {
    local_total_nnz += t.nnz();
  }
  std::vector<Type> local_all_gather_value;
  local_all_gather_value.reserve(local_total_nnz);
  std::vector<Type> local_all_gather_rowoffset{0};
  local_all_gather_rowoffset.reserve(rowoffset_count * local_count + 1);
  Type value_count = 0;
  for (auto &t : send_tensors) {
    for (size_t i = 1; i < t.rowoffset_count(); ++i) {
      local_all_gather_rowoffset.push_back(t.get_rowoffset_ptr()[i] + value_count);
    }
    for (size_t i = 0; i < t.nnz(); ++i) {
      local_all_gather_value.push_back(t.get_value_ptr()[i]);
    }
    value_count += t.nnz();
  }

  auto broadcast_result = [&recv_tensors](const std::vector<Type> &all_gather_value,
                                          const std::vector<Type> &all_gather_rowoffset) {
    for (size_t id = 0; id < recv_tensors.size(); ++id) {
      for (size_t i = 0; i < all_gather_value.size(); ++i) {
        recv_tensors[id].get_value_ptr()[i] = all_gather_value[i];
      }
      for (size_t i = 0; i < all_gather_rowoffset.size(); ++i) {
        recv_tensors[id].get_rowoffset_ptr()[i] = all_gather_rowoffset[i];
      }
      *recv_tensors[id].get_nnz_ptr() = all_gather_value.size();
    }
  };

#ifdef ENABLE_MPI
  int num_procs;
  HCTR_MPI_THROW(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));

  std::vector<int> global_total_nnz(num_procs);
  HCTR_MPI_THROW(MPI_Allgather(&local_total_nnz, sizeof(int), MPI_CHAR, global_total_nnz.data(),
                               num_procs * sizeof(int), MPI_CHAR, MPI_COMM_WORLD));
  std::vector<int> displs(num_procs);
  std::exclusive_scan(global_total_nnz.begin(), global_total_nnz.end(), global_total_nnz.begin(),
                      0);
  MPI_Datatype mpi_type = ToMpiType<Type>::T();

  size_t total_nnz_num = std::accumulate(global_total_nnz.begin(), global_total_nnz.end(), 0);
  std::vector<Type> global_all_gather_value(total_nnz_num);
  HCTR_MPI_THROW(MPI_Allgatherv(local_all_gather_value.data(), local_all_gather_value.size(),
                                mpi_type, global_all_gather_value.data(), global_total_nnz.data(),
                                displs.data(), mpi_type, MPI_COMM_WORLD));

  std::vector<Type> global_all_gather_rowoffset{0};
  global_all_gather_rowoffset.reserve(rowoffset_count * global_count + 1);
  HCTR_MPI_THROW(MPI_Allgather(local_all_gather_rowoffset.data() + 1,
                               local_all_gather_rowoffset.size() - 1, mpi_type,
                               global_all_gather_rowoffset.data() + 1,
                               rowoffset_count * global_count, mpi_type, MPI_COMM_WORLD));
  for (int i = 0; i < num_procs; ++i) {
    int restore_offset = global_total_nnz[i];
    for (size_t j = 0; j < rowoffset_count * local_count; ++j) {
      size_t idx = i * rowoffset_count * local_count + j;
      global_all_gather_rowoffset[1 + idx] += restore_offset;
    }
  }
  broadcast_result(global_all_gather_value, global_all_gather_rowoffset);
#else
  broadcast_result(local_all_gather_value, local_all_gather_rowoffset);
#endif
  return;
}

template <typename TypeKey, typename TypeEmbeddingComp>
class EmbeddingCpu {
 public:
  std::unique_ptr<HashTableCpu<TypeKey, size_t>> hash_table_;
  Tensor2<TypeKey> hash_table_key_tensors_;
  Tensor2<float> hash_table_value_tensors_;
  Tensor2<size_t> hash_value_index_tensors_;
  Tensor2<TypeEmbeddingComp> embedding_feature_tensors_;
  Tensor2<size_t> slot_id_;

  SparseEmbeddingHashParams params;
  size_t max_vocabulary_size;

  EmbeddingCpu(const SparseEmbeddingHashParams &embedding_params, size_t gpu_num)
      : hash_table_{std::make_unique<HashTableCpu<TypeKey, size_t>>()},
        params(embedding_params),
        max_vocabulary_size(embedding_params.max_vocabulary_size_per_gpu * gpu_num) {
    std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> blobs_buff =
        GeneralBuffer2<CudaHostAllocator>::create();
    size_t embedding_vec_size = embedding_params.embedding_vec_size;
    blobs_buff->reserve({max_vocabulary_size}, &hash_table_key_tensors_);
    blobs_buff->reserve({max_vocabulary_size, embedding_vec_size}, &hash_table_value_tensors_);
    blobs_buff->reserve({params.train_batch_size, params.max_feature_num},
                        &hash_value_index_tensors_);
    blobs_buff->reserve({params.train_batch_size, params.slot_num, params.embedding_vec_size},
                        &embedding_feature_tensors_);
    blobs_buff->reserve({max_vocabulary_size}, &slot_id_);

    blobs_buff->allocate();
  }

  void init(size_t key_count) {
    std::vector<size_t> temp_hash_value_index(max_vocabulary_size);
    for (size_t i = 0; i < key_count; ++i) {
      temp_hash_value_index[i] = i;
    }

    hash_table_->insert(hash_table_key_tensors_.get_ptr(), temp_hash_value_index.data(), key_count);
  }

  void forward(bool is_train, const SparseTensor<TypeKey> &train_keys,
               const SparseTensor<TypeKey> &evaluate_keys) {
    int batchsize = is_train ? params.train_batch_size : params.evaluate_batch_size;
    int slot_num = params.slot_num;
    int embedding_vec_size = params.embedding_vec_size;

    auto &cpu_keys = is_train ? train_keys : evaluate_keys;

    const TypeKey *hash_key_ = cpu_keys.get_value_ptr();
    const TypeKey *row_offset = cpu_keys.get_rowoffset_ptr();
    size_t *hash_value_index = hash_value_index_tensors_.get_ptr();
    const float *hash_table_value = hash_table_value_tensors_.get_ptr();
    TypeEmbeddingComp *embedding_feature = embedding_feature_tensors_.get_ptr();

    hash_table_->get(hash_key_, hash_value_index, cpu_keys.nnz());
    for (int user = 0; user < batchsize * slot_num; user++) {
      int feature_num = row_offset[user + 1] - row_offset[user];  // 1

      for (int vec = 0; vec < embedding_vec_size; vec++) {
        float mean = 0.0f;

        for (int item = 0; item < feature_num; item++) {
          size_t nFeatureIndex = hash_value_index[row_offset[user] + item];
          mean += hash_table_value[nFeatureIndex * embedding_vec_size + vec];
        }

        if (params.combiner == 1 && feature_num > 1) {
          mean /= (float)feature_num;
        }

        embedding_feature[user * embedding_vec_size + vec] = mean;
      }
    }
  }
};
}  // namespace unified_embedding_test
