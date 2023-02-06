/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "HugeCTR/include/data_readers/multi_hot/split_batch.hpp"

#include "HugeCTR/include/general_buffer2.hpp"
#include "gtest/gtest.h"

using namespace HugeCTR;

constexpr size_t batch_size = 1e3;
constexpr size_t label_dim = 2;
constexpr size_t dense_dim = 13;
constexpr size_t sparse_dim = 26;

template <typename TypeParam>
class SplitBatchFixture : public ::testing::Test {
 public:
  using DenseType = typename std::tuple_element<0, TypeParam>::type;
  using SparseType = typename std::tuple_element<1, TypeParam>::type;

  virtual void SetUp() {
    auto allocator = GeneralBuffer2<CudaAllocator>::create();
    allocator->reserve({batch_size, label_dim}, &label_tensor);
    allocator->reserve({batch_size, dense_dim}, &dense_tensor);
    allocator->allocate();
  }

  void init_sparse_data(std::vector<int> nnz_per_slot) {
    size_t total_nnz = std::accumulate(nnz_per_slot.begin(), nnz_per_slot.end(), 0);

    std::vector<int> bucket_ids;
    std::vector<int> bucket_positions(total_nnz);
    int bucket = 0;
    auto bucket_begin = bucket_positions.begin();
    for (auto hotness : nnz_per_slot) {
      bucket_ids.insert(bucket_ids.end(), hotness, bucket);
      bucket++;

      std::iota(bucket_begin, bucket_begin + hotness, 0);
      std::advance(bucket_begin, hotness);
    }
    size_t sample_size_int = label_dim + dense_dim + total_nnz;
    printf("sample_size_int: %zu\n", sample_size_int);

    auto allocator = GeneralBuffer2<CudaAllocator>::create();
    allocator->reserve({batch_size, sample_size_int}, &label_dense_sparse);  // batch buffer
    allocator->reserve({sparse_dim}, &sparse_tensor_ptrs);
    for (size_t i = 0; i < sparse_dim; ++i) {
      Tensor2<SparseType> tensor;
      size_t hotness = nnz_per_slot[i];
      allocator->reserve({batch_size, hotness}, &tensor);
      sparse_tensors.push_back(tensor);
    }
    allocator->reserve({total_nnz}, &bucket_id_tensor);
    allocator->reserve({total_nnz}, &bucket_position_tensor);
    allocator->reserve({sparse_dim}, &max_hotness_tensor);
    allocator->allocate();

    for (size_t i = 0; i < sparse_dim; ++i) {
      SparseType* ptr = sparse_tensors[i].get_ptr();
      HCTR_LIB_THROW(cudaMemcpy(sparse_tensor_ptrs.get_ptr() + i, &ptr, sizeof(SparseType*),
                                cudaMemcpyHostToDevice));
    }

    HCTR_LIB_THROW(cudaMemcpy(bucket_id_tensor.get_ptr(), bucket_ids.data(),
                              total_nnz * sizeof(int), cudaMemcpyHostToDevice));
    HCTR_LIB_THROW(cudaMemcpy(bucket_position_tensor.get_ptr(), bucket_positions.data(),
                              total_nnz * sizeof(int), cudaMemcpyHostToDevice));
    HCTR_LIB_THROW(cudaMemcpy(max_hotness_tensor.get_ptr(), nnz_per_slot.data(),
                              sparse_dim * sizeof(int), cudaMemcpyHostToDevice));

    // initialize each column with unique number so we can check the batch was split correctly
    std::vector<int> batch(batch_size * sample_size_int);
    for (size_t i = 0; i < batch.size(); ++i) batch[i] = (i % sample_size_int) + 1;

    HCTR_LIB_THROW(cudaMemcpy(label_dense_sparse.get_ptr(), batch.data(),
                              batch.size() * sizeof(int), cudaMemcpyHostToDevice));
  }

  void check_label() {
    std::vector<float> expected(batch_size * label_dim);
    std::vector<float> actual(batch_size * label_dim);

    for (size_t i = 0; i < expected.size(); ++i)
      expected[i] = static_cast<float>((i % label_dim) + 1);

    HCTR_LIB_THROW(cudaMemcpy(actual.data(), label_tensor.get_ptr(), actual.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    EXPECT_EQ(actual, expected);
  }

  void check_dense() {
    std::vector<DenseType> expected(batch_size * dense_dim);
    std::vector<DenseType> actual(batch_size * dense_dim);

    for (size_t i = 0; i < expected.size(); ++i) {
      int value = (i % dense_dim) + 1 + label_dim;
      expected[i] = static_cast<DenseType>(static_cast<float>(log(value + 3.f)));
    }

    HCTR_LIB_THROW(cudaMemcpy(actual.data(), dense_tensor.get_ptr(),
                              actual.size() * sizeof(DenseType), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < expected.size(); ++i) {
      // comparing as floats results in inequality even though values are same
      EXPECT_EQ((int)actual[i], (int)expected[i]);
    }
  }

  void check_sparse(std::vector<int> nnz_per_slot) {
    size_t feat_id = 0;
    for (size_t i = 0; i < sparse_dim; ++i) {
      int hotness = nnz_per_slot[i];
      std::vector<SparseType> expected(batch_size * hotness);
      std::vector<SparseType> actual(batch_size * hotness);
      for (size_t j = 0; j < expected.size(); ++j) {
        expected[j] = label_dim + dense_dim + feat_id + (j % hotness) + 1;
      }
      HCTR_LIB_THROW(cudaMemcpy(actual.data(), sparse_tensors[i].get_ptr(),
                                actual.size() * sizeof(SparseType), cudaMemcpyDeviceToHost));
      EXPECT_EQ(actual, expected);

      feat_id += hotness;
    }
  }

  Tensor2<int> label_dense_sparse;
  std::vector<Tensor2<SparseType>> sparse_tensors;
  Tensor2<SparseType*> sparse_tensor_ptrs;
  Tensor2<float> label_tensor;
  Tensor2<DenseType> dense_tensor;
  Tensor2<int> bucket_id_tensor;
  Tensor2<int> bucket_position_tensor;
  Tensor2<int> max_hotness_tensor;
};

TYPED_TEST_CASE_P(SplitBatchFixture);

TYPED_TEST_P(SplitBatchFixture, split_feat_major_one_hot) {
  std::vector<int> nnz_per_slot(sparse_dim, 1);
  this->init_sparse_data(nnz_per_slot);

  split_3_way_feat_major(this->label_tensor, this->dense_tensor, this->sparse_tensor_ptrs,
                         this->label_dense_sparse, this->bucket_id_tensor,
                         this->bucket_position_tensor, this->max_hotness_tensor, NULL);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  this->check_label();
  this->check_dense();
  this->check_sparse(nnz_per_slot);
}

TYPED_TEST_P(SplitBatchFixture, split_feat_major_multi_hot) {
  std::vector<int> nnz_per_slot;
  for (int i = 0; i < sparse_dim; ++i) {
    nnz_per_slot.push_back((rand() % 100) + 1);
  }

  this->init_sparse_data(nnz_per_slot);

  split_3_way_feat_major(this->label_tensor, this->dense_tensor, this->sparse_tensor_ptrs,
                         this->label_dense_sparse, this->bucket_id_tensor,
                         this->bucket_position_tensor, this->max_hotness_tensor, NULL);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  this->check_label();
  this->check_dense();
  this->check_sparse(nnz_per_slot);
}

typedef ::testing::Types<std::tuple<float, unsigned int>, std::tuple<__half, unsigned int>,
                         std::tuple<float, long long>, std::tuple<__half, long long>>
    SplitTypes;

REGISTER_TYPED_TEST_CASE_P(SplitBatchFixture, split_feat_major_one_hot, split_feat_major_multi_hot);
INSTANTIATE_TYPED_TEST_CASE_P(SplitBatchTests, SplitBatchFixture, SplitTypes);
