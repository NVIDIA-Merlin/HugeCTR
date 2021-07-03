/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <math.h>
#include <stdlib.h>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/data_readers/check_none.hpp"
#include "HugeCTR/include/data_readers/check_sum.hpp"
#include "HugeCTR/include/data_readers/csr_chunk.hpp"
#include "HugeCTR/include/data_readers/file_list.hpp"
#include "HugeCTR/include/data_readers/file_source.hpp"
#include "utest/embedding/cpu_hashtable.hpp"

#include <math.h>
#include <stdlib.h>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;
using namespace HugeCTR;

namespace {

template <typename OUT, typename IN>
struct TypeConvertFunc;

template <>
struct TypeConvertFunc<__half, float> {
  static inline __half convert(float val) { return __float2half(val); }
};

template <>
struct TypeConvertFunc<float, __half> {
  static inline float convert(__half val) { return __half2float(val); }
};

template <>
struct TypeConvertFunc<float, float> {
  static inline float convert(float val) { return val; }
};

}  // namespace

enum class SparseEmbedding_t { Distributed, Localized };

template <typename TypeHashKey, typename TypeEmbeddingComp>
class SparseEmbeddingHashCpu {
  using TypeHashValueIndex = TypeHashKey;

 private:
  int batchsize_;
  int max_feature_num_;
  int vocabulary_size_;
  int embedding_vec_size_;
  int slot_num_;
  int label_dim_;
  int dense_dim_;
  int combiner_;
  uint64_t times_;
  OptParams opt_params_;

  std::unique_ptr<TypeHashKey[]> row_offset_;
  std::unique_ptr<TypeHashKey[]> hash_key_;
  std::unique_ptr<float[]> dense_feature_;
  std::unique_ptr<float[]> lable_;
  std::unique_ptr<TypeHashValueIndex[]> hash_value_index_;
  std::unique_ptr<TypeHashValueIndex[]> hash_value_index_undup_;
  std::unique_ptr<TypeHashValueIndex[]> hash_value_index_undup_offset_;
  std::unique_ptr<TypeHashKey[]> sample_id_;
  std::unique_ptr<TypeHashKey[]> hash_table_key_;
  std::unique_ptr<TypeHashValueIndex[]> hash_table_value_index_;
  std::unique_ptr<float[]> hash_table_value_;
  std::unique_ptr<TypeEmbeddingComp[]> embedding_feature_;
  std::unique_ptr<TypeEmbeddingComp[]> wgrad_;
  std::unique_ptr<TypeEmbeddingComp[]> opt_m_;
  std::unique_ptr<TypeEmbeddingComp[]> opt_v_;
  std::unique_ptr<uint64_t[]> opt_prev_time_;
  std::unique_ptr<TypeEmbeddingComp[]> opt_momentum_;
  std::unique_ptr<TypeEmbeddingComp[]> opt_accm_;
  std::unique_ptr<TypeEmbeddingComp[]> opt_accm_adagrad_;

  // std::ifstream &csr_stream_;
  // long long csr_stream_offset_ = 0;
  const Check_t check_sum_;
  long long num_records_;
  int current_record_index_{0};
  DataSetHeader data_set_header_;
  std::shared_ptr<Source> source_;   /**< source: can be file or network */
  std::shared_ptr<Checker> checker_; /**< checker aim to perform error check of the input data */

  int MAX_TRY = 10;
  void read_new_file() {
    for (int i = 0; i < MAX_TRY; i++) {
      checker_->next_source();

      Error_t err =
          checker_->read(reinterpret_cast<char *>(&data_set_header_), sizeof(DataSetHeader));
      current_record_index_ = 0;

      // todo: check if file match our DataReader setting.
      if (!(data_set_header_.error_check == 0 && check_sum_ == Check_t::None) &&
          !(data_set_header_.error_check == 1 && check_sum_ == Check_t::Sum)) {
        continue;
      }
      if (err == Error_t::Success) {
        return;
      }
    }
    CK_THROW_(Error_t::BrokenFile, "failed to read a file");
  }

  std::unique_ptr<HashTableCpu<TypeHashKey, TypeHashValueIndex>> hash_table_;

 public:
  SparseEmbeddingHashCpu(int batchsize, int max_feature_num, int vocabulary_size,
                         int embedding_vec_size, int slot_num, int label_dim, int dense_dim,
                         Check_t check_sum, long long num_records, int combiner,
                         OptParams opt_params, const std::string &file_list_name,
                         const std::string &hash_table_file, const SparseEmbedding_t emb_type);
  ~SparseEmbeddingHashCpu();

  void read_a_batch();
  void forward();
  void backward();
  void update_params();

  void cpu_forward_sum(int batchsize,                  // in
                       int slot_num,                   // in
                       int embedding_vec_size,         // in
                       const TypeHashKey *row_offset,  // in  the row offsets in CSR format
                       const TypeHashValueIndex *hash_value_index,  // in
                       const float *hash_table_value,               // in
                       TypeEmbeddingComp *embedding_feature         // out
  );

  void cpu_forward_mean(int batchsize,                  // in
                        int slot_num,                   // in
                        int embedding_vec_size,         // in
                        const TypeHashKey *row_offset,  // in  the row offsets in CSR format
                        const TypeHashValueIndex *hash_value_index,  // in
                        const float *hash_table_value,               // in
                        TypeEmbeddingComp *embedding_feature         // out
  );

  void cpu_backward_sum(int batchsize,                      // in
                        int slot_num,                       // in
                        int embedding_vec_size,             // in
                        const TypeEmbeddingComp *top_grad,  // in
                        TypeEmbeddingComp *wgrad            // out  size: row_offset[nBatch]
  );

  void cpu_backward_mean(int batchsize,                      // in
                         int slot_num,                       // in
                         int embedding_vec_size,             // in
                         const TypeHashKey *row_offset,      // in   the row offsets in CSR format
                         const TypeEmbeddingComp *top_grad,  // in
                         TypeEmbeddingComp *wgrad            // out  size: row_offset[nBatch]
  );

  void cpu_csr_extend(int batchsize, int slot_num, const TypeHashKey *row_offset,
                      TypeHashKey *sample_id);

  void cpu_swap(TypeHashKey &a, TypeHashKey &b);

  void cpu_csr_sort(int nnz, TypeHashKey *hash_value_index, TypeHashKey *hash_value_index_pair);

  int cpu_csr_unduplicate(int nnz, const TypeHashValueIndex *hash_value_index,
                          TypeHashValueIndex *hash_value_index_undup,
                          TypeHashValueIndex *hash_value_index_undup_offset);

  void cpu_optimizer_adam(int feature_num_undup, int embedding_vec_size,
                          const TypeHashValueIndex *hash_value_index_undup,
                          const TypeHashValueIndex *hash_value_index_undup_offset,
                          const TypeHashKey *sample_id, const TypeEmbeddingComp *wgrad,
                          float *hash_table_value, TypeEmbeddingComp *m, TypeEmbeddingComp *v,
                          uint64_t *prev_times, float lr, uint64_t times, float beta1, float beta2,
                          float epsilon, int vocabulary_size, Update_t update_type, float scaler);

  void cpu_optimizer_adagrad(int feature_num_undup, int embedding_vec_size,
                             const TypeHashValueIndex *hash_value_index_undup,
                             const TypeHashValueIndex *hash_value_index_undup_offset,
                             const TypeHashKey *sample_id, const TypeEmbeddingComp *wgrad,
                             float *hash_table_value, TypeEmbeddingComp *accm_ptr, float lr,
                             float epsilon, int vocabulary_size, Update_t update_type,
                             float scaler);

  void cpu_optimizer_momentum(int feature_num_undup, int embedding_vec_size,
                              const TypeHashValueIndex *hash_value_index_undup,
                              const TypeHashValueIndex *hash_value_index_undup_offset,
                              const TypeHashKey *sample_id, const TypeEmbeddingComp *wgrad,
                              float *hash_table_value, TypeEmbeddingComp *momentum_ptr,
                              float factor, float lr, int vocabulary_size, Update_t update_type,
                              float scaler);

  void cpu_optimizer_nesterov(int feature_num_undup, int embedding_vec_size,
                              const TypeHashValueIndex *hash_value_index_undup,
                              const TypeHashValueIndex *hash_value_index_undup_offset,
                              const TypeHashKey *sample_id, const TypeEmbeddingComp *wgrad,
                              float *hash_table_value, TypeEmbeddingComp *accm_ptr, float mu,
                              float lr, int vocabulary_size, Update_t update_type, float scaler);

  void cpu_optimizer_sgd(int feature_num_undup, int embedding_vec_size,
                         const TypeHashValueIndex *hash_value_index_undup,
                         const TypeHashValueIndex *hash_value_index_undup_offset,
                         const TypeHashKey *sample_id, const TypeEmbeddingComp *wgrad,
                         float *hash_table_value, float lr, int vocabulary_size, float scaler);

  // only used for results check
  TypeEmbeddingComp *get_forward_results() { return embedding_feature_.get(); }
  TypeEmbeddingComp *get_backward_results() { return wgrad_.get(); }
  TypeHashKey *get_hash_table_key_ptr() { return hash_table_key_.get(); }
  TypeHashValueIndex *get_hash_table_value_index_ptr() { return hash_table_value_index_.get(); }
  float *get_hash_table_value_ptr() { return hash_table_value_.get(); }

};  // end of class SparseEmbeddingHashCpu

template <typename TypeHashKey, typename TypeEmbeddingComp>
SparseEmbeddingHashCpu<TypeHashKey, TypeEmbeddingComp>::SparseEmbeddingHashCpu(
    int batchsize, int max_feature_num, int vocabulary_size, int embedding_vec_size, int slot_num,
    int label_dim, int dense_dim, const Check_t check_sum, const long long num_records,
    int combiner, OptParams opt_params, const std::string &file_list_name,
    const std::string &hash_table_file, const SparseEmbedding_t emb_type)
    : batchsize_(batchsize),
      max_feature_num_(max_feature_num),
      vocabulary_size_(vocabulary_size),
      embedding_vec_size_(embedding_vec_size),
      slot_num_(slot_num),
      label_dim_(label_dim),
      dense_dim_(dense_dim),
      check_sum_(check_sum),
      num_records_(num_records),
      combiner_(combiner),
      opt_params_(opt_params) {
#ifndef NDEBUG
  PRINT_FUNC_NAME_();
#endif

  // malloc memory
  hash_table_.reset(new HashTableCpu<TypeHashKey, TypeHashValueIndex>());
  hash_table_value_.reset(new float[vocabulary_size_ * embedding_vec_size_]);  // embedding table
  hash_table_key_.reset(new TypeHashKey[vocabulary_size_]);
  row_offset_.reset(new TypeHashKey[batchsize_ * slot_num_ + 1]);
  lable_.reset(new float[batchsize_ * label_dim_]);
  dense_feature_.reset(new float[batchsize_ * dense_dim_]);
  hash_key_.reset(new TypeHashKey[batchsize_ * max_feature_num_]);
  hash_value_index_.reset(new TypeHashValueIndex[batchsize_ * max_feature_num_]);
  hash_value_index_undup_.reset(new TypeHashValueIndex[batchsize_ * max_feature_num_]);
  hash_value_index_undup_offset_.reset(new TypeHashValueIndex[batchsize_ * max_feature_num_ + 1]);
  sample_id_.reset(new TypeHashKey[batchsize_ * max_feature_num_]);
  embedding_feature_.reset(new TypeEmbeddingComp[batchsize_ * slot_num_ * embedding_vec_size_]);
  wgrad_.reset(new TypeEmbeddingComp[batchsize_ * slot_num_ * embedding_vec_size_]);
  opt_m_.reset(new TypeEmbeddingComp[vocabulary_size_ * embedding_vec_size_]);
  memset(opt_m_.get(), 0, vocabulary_size_ * embedding_vec_size_ * sizeof(TypeEmbeddingComp));
  opt_v_.reset(new TypeEmbeddingComp[vocabulary_size_ * embedding_vec_size_]);
  memset(opt_v_.get(), 0, vocabulary_size_ * embedding_vec_size_ * sizeof(TypeEmbeddingComp));
  opt_prev_time_.reset(new uint64_t[vocabulary_size_ * embedding_vec_size_]);
  uint64_t *opt_prev_time_ptr = opt_prev_time_.get();
  for (TypeHashValueIndex i = 0; i < vocabulary_size_ * embedding_vec_size_; i++) {
    opt_prev_time_ptr[i] = 1;
  }
  opt_momentum_.reset(new TypeEmbeddingComp[vocabulary_size_ * embedding_vec_size_]);
  memset(opt_momentum_.get(), 0,
         vocabulary_size_ * embedding_vec_size_ * sizeof(TypeEmbeddingComp));
  opt_accm_.reset(new TypeEmbeddingComp[vocabulary_size_ * embedding_vec_size_]);
  memset(opt_accm_.get(), 0, vocabulary_size_ * embedding_vec_size_ * sizeof(TypeEmbeddingComp));
  
  opt_accm_adagrad_.reset(new TypeEmbeddingComp[vocabulary_size_ * embedding_vec_size_]);
  memset(opt_accm_adagrad_.get(), 0, vocabulary_size_ * embedding_vec_size_ * sizeof(TypeEmbeddingComp));
  

  hash_table_value_index_.reset(new TypeHashValueIndex[vocabulary_size_]);
  for (TypeHashValueIndex i = 0; i < vocabulary_size_; i++) {
    hash_table_value_index_[i] = i;
  }

  // read hash table
  {
    const std::string key_file(hash_table_file + "/" + hash_table_file + ".key");
    const std::string vec_file(hash_table_file + "/" + hash_table_file + ".vec");

    std::ifstream key_stream(key_file, std::ifstream::binary);
    std::ifstream vec_stream(vec_file, std::ifstream::binary);
    if (!key_stream.is_open() || !vec_stream.is_open()) {
      ERROR_MESSAGE_("Error: hash table file open failed");
      return;
    }
    auto key_file_size_in_B = fs::file_size(key_file);
    auto vec_file_size_in_B = fs::file_size(vec_file);
    const int num_key = key_file_size_in_B / sizeof(TypeHashKey);
    const int num_vec = vec_file_size_in_B / (sizeof(float) * embedding_vec_size_);

    if (num_key != num_vec || num_key > vocabulary_size_) {
      ERROR_MESSAGE_("Error: hash table file size is smaller than embedding_table_size required");
      return;
    }

    key_stream.read(reinterpret_cast<char *>(hash_table_key_.get()), key_file_size_in_B);
    vec_stream.read(reinterpret_cast<char *>(hash_table_value_.get()), vec_file_size_in_B);
  }

  // insert <key,value_index> into HashTableCpu
  hash_table_->insert(hash_table_key_.get(), hash_table_value_index_.get(), vocabulary_size_);

  // dataset filelist
  source_ = std::make_shared<FileSource>(0, 1, file_list_name, true);
  switch (check_sum_) {
    case Check_t::Sum:
      checker_ = std::make_shared<CheckSum>(*source_);
      break;
    case Check_t::None:
      checker_ = std::make_shared<CheckNone>(*source_);
      break;
    default:
      assert(!"Error: no such Check_t && should never get here!!");
  }

  // for optimizer
  times_ = 0;

  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
SparseEmbeddingHashCpu<TypeHashKey, TypeEmbeddingComp>::~SparseEmbeddingHashCpu() {
#ifndef NDEBUG
  PRINT_FUNC_NAME_();
#endif
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingHashCpu<TypeHashKey, TypeEmbeddingComp>::read_a_batch() {
  try {
    if (!checker_->is_open()) {
      read_new_file();
    }

    row_offset_[0] = 0;

    // batch loop
    for (int i = 0; i < batchsize_; i++) {
      checker_->read(reinterpret_cast<char *>(lable_.get() + i * label_dim_),
                     sizeof(float) * label_dim_);
      checker_->read(reinterpret_cast<char *>(dense_feature_.get() + i * dense_dim_),
                     sizeof(float) * dense_dim_);

      for (int k = 0; k < slot_num_; k++) {
        int nnz;
        checker_->read(reinterpret_cast<char *>(&nnz), sizeof(int));
        row_offset_[i * slot_num_ + k + 1] = row_offset_[i * slot_num_ + k] + nnz;
        checker_->read(reinterpret_cast<char *>(hash_key_.get() + row_offset_[i * slot_num_ + k]),
                       sizeof(TypeHashKey) * nnz);
      }

      current_record_index_++;

      // start a new file when finish one file read
      if (current_record_index_ >= data_set_header_.number_of_records) {
        read_new_file();
      }
    }  // batch loop
  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
  return;
}

// CPU implementation of forward computation of embedding lookup and sum reduction with sparse
// matrix as input
template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingHashCpu<TypeHashKey, TypeEmbeddingComp>::cpu_forward_sum(
    int batchsize,                               // in
    int slot_num,                                // in
    int embedding_vec_size,                      // in
    const TypeHashKey *row_offset,               // in  the row offsets in CSR format
    const TypeHashValueIndex *hash_value_index,  // in
    const float *hash_table_value,               // in
    TypeEmbeddingComp *embedding_feature         // out
) {
  for (int user = 0; user < batchsize * slot_num; user++) {
    int feature_num = row_offset[user + 1] - row_offset[user];

    for (int vec = 0; vec < embedding_vec_size; vec++) {
      float sum = 0.0f;

      for (int item = 0; item < feature_num; item++) {
        TypeHashValueIndex nFeatureIndex = hash_value_index[row_offset[user] + item];

        sum += hash_table_value[nFeatureIndex * embedding_vec_size + vec];
      }

      embedding_feature[user * embedding_vec_size + vec] = sum;
    }
  }
}

// CPU implementation of forward computation of embedding lookup and mean reduction with sparse
// matrix as input
template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingHashCpu<TypeHashKey, TypeEmbeddingComp>::cpu_forward_mean(
    int batchsize,                               // in
    int slot_num,                                // in
    int embedding_vec_size,                      // in
    const TypeHashKey *row_offset,               // in  the row offsets in CSR format
    const TypeHashValueIndex *hash_value_index,  // in
    const float *hash_table_value,               // in
    TypeEmbeddingComp *embedding_feature         // out
) {
  for (int user = 0; user < batchsize * slot_num; user++) {
    int feature_num = row_offset[user + 1] - row_offset[user];

    for (int vec = 0; vec < embedding_vec_size; vec++) {
      float mean = 0.0f;

      for (int item = 0; item < feature_num; item++) {
        TypeHashValueIndex nFeatureIndex = hash_value_index[row_offset[user] + item];

        mean += hash_table_value[nFeatureIndex * embedding_vec_size + vec];
      }

      if (feature_num > 1) {
        mean /= (float)feature_num;
      }

      embedding_feature[user * embedding_vec_size + vec] = mean;
    }
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingHashCpu<TypeHashKey, TypeEmbeddingComp>::forward() {
#ifndef NDEBUG
  PRINT_FUNC_NAME_();
#endif

  read_a_batch();

  // do hash_table get() value_index by key
  hash_table_->get(hash_key_.get(), hash_value_index_.get(), row_offset_[batchsize_ * slot_num_]);

  if (combiner_ == 0) {
    cpu_forward_sum(batchsize_, slot_num_, embedding_vec_size_, row_offset_.get(),
                    hash_value_index_.get(), hash_table_value_.get(), embedding_feature_.get());

  } else if (combiner_ == 1) {
    cpu_forward_mean(batchsize_, slot_num_, embedding_vec_size_, row_offset_.get(),
                     hash_value_index_.get(), hash_table_value_.get(), embedding_feature_.get());
  } else {
  }
}

// CPU implementation of backward computation of embedding layer with sparse matrix as input
template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingHashCpu<TypeHashKey, TypeEmbeddingComp>::cpu_backward_sum(
    int batchsize,                      // in
    int slot_num,                       // in
    int embedding_vec_size,             // in
    const TypeEmbeddingComp *top_grad,  // in
    TypeEmbeddingComp *wgrad            // out  size: row_offset[nBatch]
) {
  for (int user = 0; user < batchsize * slot_num; user++) {
    for (int vec = 0; vec < embedding_vec_size; vec++) {
      wgrad[user * embedding_vec_size + vec] = top_grad[user * embedding_vec_size + vec];
    }
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingHashCpu<TypeHashKey, TypeEmbeddingComp>::cpu_backward_mean(
    int batchsize,                      // in
    int slot_num,                       // in
    int embedding_vec_size,             // in
    const TypeHashKey *row_offset,      // in   the row offsets in CSR format
    const TypeEmbeddingComp *top_grad,  // in
    TypeEmbeddingComp *wgrad            // out  size: row_offset[nBatch]
) {
  for (int user = 0; user < batchsize * slot_num; user++) {
    int feature_num = row_offset[user + 1] - row_offset[user];
    float scaler = 1.0f;
    if (feature_num > 1) {
      scaler = 1.0f / (float)feature_num;
    }

    for (int vec = 0; vec < embedding_vec_size; vec++) {
      float grad = TypeConvertFunc<float, TypeEmbeddingComp>::convert(
          top_grad[user * embedding_vec_size + vec]);
      grad *= scaler;
      wgrad[user * embedding_vec_size + vec] =
          TypeConvertFunc<float, TypeEmbeddingComp>::convert(grad);
    }
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingHashCpu<TypeHashKey, TypeEmbeddingComp>::backward() {
#ifndef NDEBUG
  PRINT_FUNC_NAME_();
#endif

  if (combiner_ == 0) {
    cpu_backward_sum(batchsize_, slot_num_, embedding_vec_size_, embedding_feature_.get(),
                     wgrad_.get());
  } else if (combiner_ == 1) {
    cpu_backward_mean(batchsize_, slot_num_, embedding_vec_size_, row_offset_.get(),
                      embedding_feature_.get(), wgrad_.get());
  } else {
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingHashCpu<TypeHashKey, TypeEmbeddingComp>::cpu_csr_extend(
    int batchsize, int slot_num, const TypeHashKey *row_offset, TypeHashKey *sample_id) {
  for (int i = 0; i < batchsize * slot_num; i++) {  // loop of sample id
    int feature_num = row_offset[i + 1] - row_offset[i];
    for (int j = 0; j < feature_num; j++) {
      sample_id[row_offset[i] + j] = i;  // record sample id coresponding to each feature
    }
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingHashCpu<TypeHashKey, TypeEmbeddingComp>::cpu_swap(TypeHashKey &a,
                                                                      TypeHashKey &b) {
  TypeHashKey temp = a;
  a = b;
  b = temp;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingHashCpu<TypeHashKey, TypeEmbeddingComp>::cpu_csr_sort(
    int nnz, TypeHashValueIndex *hash_value_index, TypeHashValueIndex *hash_value_index_pair) {
  // odd even sort
  for (int i = 0; i < nnz; i++) {
    if (i % 2 == 0) {  // even
      for (int j = 1; j < nnz; j += 2) {
        if (hash_value_index[j] < hash_value_index[j - 1]) {
          cpu_swap(hash_value_index[j], hash_value_index[j - 1]);
          cpu_swap(hash_value_index_pair[j], hash_value_index_pair[j - 1]);
        }
      }
    } else {  // odd
      for (int j = 1; j < nnz - 1; j += 2) {
        if (hash_value_index[j] > hash_value_index[j + 1]) {
          cpu_swap(hash_value_index[j], hash_value_index[j + 1]);
          cpu_swap(hash_value_index_pair[j], hash_value_index_pair[j + 1]);
        }
      }
    }
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
int SparseEmbeddingHashCpu<TypeHashKey, TypeEmbeddingComp>::cpu_csr_unduplicate(
    int nnz, const TypeHashValueIndex *hash_value_index, TypeHashValueIndex *hash_value_index_undup,
    TypeHashValueIndex *hash_value_index_undup_offset) {
  hash_value_index_undup_offset[0] = 0;

  int counter = 0;
  for (int i = 0; i < nnz - 1; i++) {
    if (hash_value_index[i] != hash_value_index[i + 1]) {
      hash_value_index_undup[counter] = hash_value_index[i];
      hash_value_index_undup_offset[counter + 1] = i + 1;
      counter++;
    }
  }

  hash_value_index_undup[counter] = hash_value_index[nnz - 1];
  hash_value_index_undup_offset[counter + 1] = nnz;
  counter++;

  return counter;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingHashCpu<TypeHashKey, TypeEmbeddingComp>::cpu_optimizer_adam(
    int feature_num_undup, int embedding_vec_size, const TypeHashValueIndex *hash_value_index_undup,
    const TypeHashValueIndex *hash_value_index_undup_offset, const TypeHashKey *sample_id,
    const TypeEmbeddingComp *wgrad, float *hash_table_value, TypeEmbeddingComp *m,
    TypeEmbeddingComp *v, uint64_t *prev_times, float lr, uint64_t times, float beta1, float beta2,
    float epsilon, int vocabulary_size, Update_t update_type, float scaler) {
  const float alpha_t = lr * sqrt(1.0f - pow(beta2, times_)) / (1.0f - pow(beta1, times_));
  const float alpha_t_lazy_common = lr / (1.0f - beta1);

  for (int i = 0; i < feature_num_undup; i++) {
    TypeHashValueIndex cur_offset = hash_value_index_undup_offset[i];
    TypeHashValueIndex sample_num = hash_value_index_undup_offset[i + 1] - cur_offset;
    TypeHashValueIndex row_index = hash_value_index_undup[i];

    for (int j = 0; j < embedding_vec_size; j++) {
      float gi = 0.0f;
      for (int k = 0; k < sample_num; k++) {
        int sample_index = sample_id[cur_offset + k];
        gi += TypeConvertFunc<float, TypeEmbeddingComp>::convert(
            wgrad[sample_index * embedding_vec_size + j]);
      }

      gi = gi / scaler;

      TypeHashValueIndex feature_index = row_index * embedding_vec_size + j;

      switch (update_type) {
        case Update_t::Local: {
          float mi = beta1 * TypeConvertFunc<float, TypeEmbeddingComp>::convert(m[feature_index]) +
                     (1.0f - beta1) * gi;
          float vi = beta2 * TypeConvertFunc<float, TypeEmbeddingComp>::convert(v[feature_index]) +
                     (1.0f - beta2) * gi * gi;
          m[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mi);
          v[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(vi);
          float weight_diff = -alpha_t * mi / (sqrtf(vi) + epsilon);
          hash_table_value[feature_index] += weight_diff;
          break;
        }
        case Update_t::Global: {
          float mi = TypeConvertFunc<float, TypeEmbeddingComp>::convert(m[feature_index]) +
                     (1.0f - beta1) * gi / beta1;
          float vi = TypeConvertFunc<float, TypeEmbeddingComp>::convert(v[feature_index]) +
                     (1.0f - beta2) * gi * gi / beta2;
          m[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mi);
          v[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(vi);
          break;
        }
        case Update_t::LazyGlobal: {
          uint64_t prev_time = prev_times[feature_index];
          prev_times[feature_index] = times_;
          uint64_t skipped = times_ - prev_time;
          float beta1_pow_skipped = pow(beta1, skipped);
          float mi = TypeConvertFunc<float, TypeEmbeddingComp>::convert(m[feature_index]);
          float vi = TypeConvertFunc<float, TypeEmbeddingComp>::convert(v[feature_index]);
          const float alpha_t_lazy = alpha_t_lazy_common * sqrt(1.0f - pow(beta2, prev_time)) /
                                     (1.0f - pow(beta1, prev_time)) * (1.0f - beta1_pow_skipped);
          float weight_diff = -alpha_t_lazy * mi / (sqrtf(vi) + epsilon);
          hash_table_value[feature_index] += weight_diff;
          mi = beta1_pow_skipped * mi + (1.0f - beta1) * gi;
          vi = pow(beta2, skipped) * vi + (1.0f - beta2) * gi * gi;
          m[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mi);
          v[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(vi);
          break;
        }
      }
    }
  }

  if (update_type == Update_t::Global) {
    for (int i = 0; i < vocabulary_size; i++) {
      for (int j = 0; j < embedding_vec_size; j++) {
        TypeHashValueIndex feature_index = i * embedding_vec_size + j;
        float mi = TypeConvertFunc<float, TypeEmbeddingComp>::convert(m[feature_index]) * beta1;
        float vi = TypeConvertFunc<float, TypeEmbeddingComp>::convert(v[feature_index]) * beta2;
        m[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mi);
        v[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(vi);
        float weight_diff = -alpha_t * mi / (sqrtf(vi) + epsilon);
        hash_table_value[feature_index] += weight_diff;
      }
    }
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingHashCpu<TypeHashKey, TypeEmbeddingComp>::cpu_optimizer_adagrad(
    int feature_num_undup, int embedding_vec_size, const TypeHashValueIndex *hash_value_index_undup,
    const TypeHashValueIndex *hash_value_index_undup_offset, const TypeHashKey *sample_id,
    const TypeEmbeddingComp *wgrad, float *hash_table_value, TypeEmbeddingComp *accm_ptr, float lr,
    float epsilon, int vocabulary_size, Update_t update_type, float scaler) {
  for (int i = 0; i < feature_num_undup; i++) {
    TypeHashValueIndex cur_offset = hash_value_index_undup_offset[i];
    TypeHashValueIndex sample_num = hash_value_index_undup_offset[i + 1] - cur_offset;
    TypeHashValueIndex row_index = hash_value_index_undup[i];

    for (int j = 0; j < embedding_vec_size; j++) {
      float gi = 0.0f;
      for (int k = 0; k < sample_num; k++) {
        int sample_index = sample_id[cur_offset + k];
        gi += TypeConvertFunc<float, TypeEmbeddingComp>::convert(
            wgrad[sample_index * embedding_vec_size + j]);
      }

      gi = gi / scaler;

      TypeHashValueIndex feature_index = row_index * embedding_vec_size + j;
      float accm = TypeConvertFunc<float, TypeEmbeddingComp>::convert(accm_ptr[feature_index]) + gi * gi;
      accm_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(accm);

      float weight_diff = -lr * gi / (sqrtf(accm) + epsilon);
      hash_table_value[feature_index] += weight_diff;
    }
  }
}

  template <typename TypeHashKey, typename TypeEmbeddingComp>
  void SparseEmbeddingHashCpu<TypeHashKey, TypeEmbeddingComp>::cpu_optimizer_momentum(
      int feature_num_undup, int embedding_vec_size,
      const TypeHashValueIndex *hash_value_index_undup,
      const TypeHashValueIndex *hash_value_index_undup_offset, const TypeHashKey *sample_id,
      const TypeEmbeddingComp *wgrad, float *hash_table_value, TypeEmbeddingComp *momentum_ptr,
      float factor, float lr, int vocabulary_size, Update_t update_type, float scaler) {
    for (int i = 0; i < feature_num_undup; i++) {
      TypeHashValueIndex cur_offset = hash_value_index_undup_offset[i];
      TypeHashValueIndex sample_num = hash_value_index_undup_offset[i + 1] - cur_offset;
      TypeHashValueIndex row_index = hash_value_index_undup[i];

      for (int j = 0; j < embedding_vec_size; j++) {
        float gi = 0.0f;
        for (int k = 0; k < sample_num; k++) {
          int sample_index = sample_id[cur_offset + k];
          gi += TypeConvertFunc<float, TypeEmbeddingComp>::convert(
              wgrad[sample_index * embedding_vec_size + j]);
        }

        gi = gi / scaler;

        TypeHashValueIndex feature_index = row_index * embedding_vec_size + j;

        if (update_type == Update_t::Local) {  // local update
          float mo = factor * TypeConvertFunc<float, TypeEmbeddingComp>::convert(
                                  momentum_ptr[feature_index]) -
                     lr * gi;
          momentum_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mo);
          hash_table_value[feature_index] += mo;
        } else if (update_type == Update_t::Global) {  // global update
          float mo =
              TypeConvertFunc<float, TypeEmbeddingComp>::convert(momentum_ptr[feature_index]) -
              lr * gi / factor;
          momentum_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mo);
        } else {  // lazy global update
          /// TODO: implement CPU lazy momentum update
        }
      }
    }

    if (update_type == Update_t::Global) {
      for (int i = 0; i < vocabulary_size; i++) {
        for (int j = 0; j < embedding_vec_size; j++) {
          TypeHashValueIndex feature_index = i * embedding_vec_size + j;
          float mo = factor * TypeConvertFunc<float, TypeEmbeddingComp>::convert(
                                  momentum_ptr[feature_index]);
          momentum_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mo);
          hash_table_value[feature_index] += mo;
        }
      }
    }
  }

  template <typename TypeHashKey, typename TypeEmbeddingComp>
  void SparseEmbeddingHashCpu<TypeHashKey, TypeEmbeddingComp>::cpu_optimizer_nesterov(
      int feature_num_undup, int embedding_vec_size,
      const TypeHashValueIndex *hash_value_index_undup,
      const TypeHashValueIndex *hash_value_index_undup_offset, const TypeHashKey *sample_id,
      const TypeEmbeddingComp *wgrad, float *hash_table_value, TypeEmbeddingComp *accm_ptr,
      float mu, float lr, int vocabulary_size, Update_t update_type, float scaler) {
    if (update_type == Update_t::Global) {
      for (int i = 0; i < vocabulary_size; i++) {
        for (int j = 0; j < embedding_vec_size; j++) {
          TypeHashValueIndex feature_index = i * embedding_vec_size + j;
          float accm_old = accm_ptr[feature_index];
          float accm_new = mu * accm_old;
          accm_ptr[feature_index] = accm_new;
          hash_table_value[feature_index] += mu * accm_new;
        }
      }
    }

    for (int i = 0; i < feature_num_undup; i++) {
      TypeHashValueIndex cur_offset = hash_value_index_undup_offset[i];
      TypeHashValueIndex sample_num = hash_value_index_undup_offset[i + 1] - cur_offset;
      TypeHashValueIndex row_index = hash_value_index_undup[i];

      for (int j = 0; j < embedding_vec_size; j++) {
        float gi = 0.0f;
        for (int k = 0; k < sample_num; k++) {
          int sample_index = sample_id[cur_offset + k];
          gi += TypeConvertFunc<float, TypeEmbeddingComp>::convert(
              wgrad[sample_index * embedding_vec_size + j]);
        }

        gi = gi / scaler;

        TypeHashValueIndex feature_index = row_index * embedding_vec_size + j;

        if (update_type == Update_t::Local) {  // local update
          float accm_old =
              TypeConvertFunc<float, TypeEmbeddingComp>::convert(accm_ptr[feature_index]);
          float accm_new = mu * accm_old - lr * gi;
          accm_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(accm_new);
          float weight_diff = -mu * accm_old + (1.0f + mu) * accm_new;
          hash_table_value[feature_index] += weight_diff;
        } else if (update_type == Update_t::Global) {  // global update
          float accm = TypeConvertFunc<float, TypeEmbeddingComp>::convert(accm_ptr[feature_index]);
          accm -= lr * gi;
          accm_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(accm);
          hash_table_value[feature_index] -= (1.0f + mu) * lr * gi;
        } else {  // lazy global update
          /// TODO: implement CPU lazy Nesterov update
        }
      }
    }
  }

  template <typename TypeHashKey, typename TypeEmbeddingComp>
  void SparseEmbeddingHashCpu<TypeHashKey, TypeEmbeddingComp>::cpu_optimizer_sgd(
      int feature_num_undup, int embedding_vec_size,
      const TypeHashValueIndex *hash_value_index_undup,
      const TypeHashValueIndex *hash_value_index_undup_offset, const TypeHashKey *sample_id,
      const TypeEmbeddingComp *wgrad, float *hash_table_value, float lr, int vocabulary_size,
      float scaler) {
    for (int i = 0; i < feature_num_undup; i++) {
      TypeHashValueIndex cur_offset = hash_value_index_undup_offset[i];
      TypeHashValueIndex sample_num = hash_value_index_undup_offset[i + 1] - cur_offset;
      TypeHashValueIndex row_index = hash_value_index_undup[i];

      for (int j = 0; j < embedding_vec_size; j++) {
        float gi = 0.0f;
        for (int k = 0; k < sample_num; k++) {
          int sample_index = sample_id[cur_offset + k];
          gi += TypeConvertFunc<float, TypeEmbeddingComp>::convert(
              wgrad[sample_index * embedding_vec_size + j]);
        }

        gi = gi / scaler;

        TypeHashValueIndex feature_index = row_index * embedding_vec_size + j;
        hash_table_value[feature_index] -= lr * gi;
      }
    }
  }

  template <typename TypeHashKey, typename TypeEmbeddingComp>
  void SparseEmbeddingHashCpu<TypeHashKey, TypeEmbeddingComp>::update_params() {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif

    // step1: extend sample IDs
    cpu_csr_extend(batchsize_, slot_num_, row_offset_.get(), sample_id_.get());

    // step2: do hash table get() value_index by key
    int nnz = row_offset_[batchsize_ * slot_num_];
    hash_table_->get(hash_key_.get(), hash_value_index_.get(), nnz);

    // step3: sort by value_index
    cpu_csr_sort(nnz, hash_value_index_.get(), sample_id_.get());

    // step4: unduplicate by value_index
    int feature_num_undup =
        cpu_csr_unduplicate(nnz, hash_value_index_.get(), hash_value_index_undup_.get(),
                            hash_value_index_undup_offset_.get());

    // step5: sort by value_index   no need to do this for CPU
    cpu_csr_sort(feature_num_undup, hash_value_index_undup_.get(),
                 hash_value_index_undup_offset_.get());

    // step6: update params
    switch (opt_params_.optimizer) {
      case Optimizer_t::Adam: {
        times_++;
        cpu_optimizer_adam(feature_num_undup, embedding_vec_size_, hash_value_index_undup_.get(),
                           hash_value_index_undup_offset_.get(), sample_id_.get(), wgrad_.get(),
                           hash_table_value_.get(), opt_m_.get(), opt_v_.get(),
                           opt_prev_time_.get(), opt_params_.lr, times_,
                           opt_params_.hyperparams.adam.beta1, opt_params_.hyperparams.adam.beta2,
                           opt_params_.hyperparams.adam.epsilon, vocabulary_size_,
                           opt_params_.update_type, opt_params_.scaler);
        break;
      }
      case Optimizer_t::AdaGrad: {
        cpu_optimizer_adagrad(feature_num_undup, embedding_vec_size_, hash_value_index_undup_.get(),
                           hash_value_index_undup_offset_.get(), sample_id_.get(), wgrad_.get(),
                           hash_table_value_.get(), opt_accm_adagrad_.get(),
                           opt_params_.lr,
                           opt_params_.hyperparams.adagrad.epsilon, vocabulary_size_,
                           opt_params_.update_type, opt_params_.scaler);
        break;
      }
      case Optimizer_t::MomentumSGD: {
        cpu_optimizer_momentum(
            feature_num_undup, embedding_vec_size_, hash_value_index_undup_.get(),
            hash_value_index_undup_offset_.get(), sample_id_.get(), wgrad_.get(),
            hash_table_value_.get(), opt_momentum_.get(), opt_params_.hyperparams.momentum.factor,
            opt_params_.lr, vocabulary_size_, opt_params_.update_type, opt_params_.scaler);
        break;
      }
      case Optimizer_t::Nesterov: {
        cpu_optimizer_nesterov(feature_num_undup, embedding_vec_size_,
                               hash_value_index_undup_.get(), hash_value_index_undup_offset_.get(),
                               sample_id_.get(), wgrad_.get(), hash_table_value_.get(),
                               opt_accm_.get(), opt_params_.hyperparams.nesterov.mu, opt_params_.lr,
                               vocabulary_size_, opt_params_.update_type, opt_params_.scaler);
        break;
      }
      case Optimizer_t::SGD: {
        cpu_optimizer_sgd(feature_num_undup, embedding_vec_size_, hash_value_index_undup_.get(),
                          hash_value_index_undup_offset_.get(), sample_id_.get(), wgrad_.get(),
                          hash_table_value_.get(), opt_params_.lr, vocabulary_size_,
                          opt_params_.scaler);
        break;
      }
      default: {
        printf("Error: optimizer not supported in CPU version\n");
      }
    }

    return;
  }
