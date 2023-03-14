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

#include <gtest/gtest.h>

#include <data_readers/multi_hot/split_batch.hpp>

using namespace HugeCTR;

constexpr int64_t batch_size = 1e3;
constexpr int64_t label_dim = 2;
constexpr int64_t dense_dim = 13;
constexpr int64_t sparse_dim = 26;

template <typename TypeParam>
class SplitBatchFixture : public ::testing::Test {
 public:
  using DenseType = typename std::tuple_element<0, TypeParam>::type;
  using SparseType = typename std::tuple_element<1, TypeParam>::type;

  virtual void SetUp() {
    label_tensor = core23::Tensor(
        core23::TensorParams().shape({batch_size, label_dim}).data_type(core23::ScalarType::Float));
    label_tensor.data();
    dense_tensor = core23::Tensor(core23::TensorParams()
                                      .shape({batch_size, dense_dim})
                                      .data_type(core23::ToScalarType<DenseType>::value));
    dense_tensor.data();
  }

  void init_sparse_data(std::vector<int> nnz_per_slot) {
    int64_t total_nnz = std::accumulate(nnz_per_slot.begin(), nnz_per_slot.end(), 0);

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
    int64_t sample_size_int = label_dim + dense_dim + total_nnz;
    printf("sample_size_int: %zu\n", sample_size_int);

    // auto allocator = GeneralBuffer2<CudaAllocator>::create();
    label_dense_sparse = core23::Tensor(core23::TensorParams()
                                            .shape({batch_size, sample_size_int})
                                            .data_type(core23::ScalarType::Int32));
    sparse_tensor_ptrs = core23::Tensor(
        core23::TensorParams().shape({sparse_dim}).data_type(core23::ScalarType::UInt64));
    // allocate eagerly
    label_dense_sparse.data();
    sparse_tensor_ptrs.data();

    for (int64_t i = 0; i < sparse_dim; ++i) {
      int64_t hotness = nnz_per_slot[i];
      sparse_tensors.emplace_back(core23::TensorParams()
                                      .shape({batch_size, hotness})
                                      .data_type(core23::ToScalarType<SparseType>::value));
    }
    bucket_id_tensor = core23::Tensor(
        core23::TensorParams().shape({total_nnz}).data_type(core23::ScalarType::Int32));
    bucket_position_tensor = core23::Tensor(
        core23::TensorParams().shape({total_nnz}).data_type(core23::ScalarType::Int32));
    max_hotness_tensor = core23::Tensor(
        core23::TensorParams().shape({sparse_dim}).data_type(core23::ScalarType::Int32));

    for (int64_t i = 0; i < sparse_dim; ++i) {
      SparseType* ptr = sparse_tensors[i].data<SparseType>();
      HCTR_LIB_THROW(cudaMemcpy(reinterpret_cast<SparseType**>(sparse_tensor_ptrs.data()) + i, &ptr,
                                sizeof(SparseType*), cudaMemcpyHostToDevice));
    }

    HCTR_LIB_THROW(cudaMemcpy(bucket_id_tensor.data(), bucket_ids.data(), total_nnz * sizeof(int),
                              cudaMemcpyHostToDevice));
    HCTR_LIB_THROW(cudaMemcpy(bucket_position_tensor.data(), bucket_positions.data(),
                              total_nnz * sizeof(int), cudaMemcpyHostToDevice));
    HCTR_LIB_THROW(cudaMemcpy(max_hotness_tensor.data(), nnz_per_slot.data(),
                              sparse_dim * sizeof(int), cudaMemcpyHostToDevice));

    // initialize each column with unique number so we can check the batch was split correctly
    std::vector<int> batch(batch_size * sample_size_int);
    for (size_t i = 0; i < batch.size(); ++i) batch[i] = (i % sample_size_int) + 1;

    HCTR_LIB_THROW(cudaMemcpy(label_dense_sparse.data(), batch.data(), batch.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
  }

  void check_label() {
    std::vector<float> expected(batch_size * label_dim);
    std::vector<float> actual(batch_size * label_dim);

    for (size_t i = 0; i < expected.size(); ++i)
      expected[i] = static_cast<float>((i % label_dim) + 1);

    HCTR_LIB_THROW(cudaMemcpy(actual.data(), label_tensor.data(), actual.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    EXPECT_EQ(actual, expected);
  }

  void check_dense() {
    std::vector<DenseType> expected(batch_size * dense_dim);
    std::vector<DenseType> actual(batch_size * dense_dim);

    for (size_t i = 0; i < expected.size(); ++i) {
      int value = (i % dense_dim) + 1 + label_dim;
      expected[i] = static_cast<DenseType>(static_cast<float>(log(value + 1.f)));
    }

    HCTR_LIB_THROW(cudaMemcpy(actual.data(), dense_tensor.data(), actual.size() * sizeof(DenseType),
                              cudaMemcpyDeviceToHost));

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
      HCTR_LIB_THROW(cudaMemcpy(actual.data(), sparse_tensors[i].data(),
                                actual.size() * sizeof(SparseType), cudaMemcpyDeviceToHost));
      EXPECT_EQ(actual, expected);

      feat_id += hotness;
    }
  }
  core23::Tensor label_dense_sparse;
  std::vector<core23::Tensor> sparse_tensors;
  core23::Tensor sparse_tensor_ptrs;
  core23::Tensor label_tensor;
  core23::Tensor dense_tensor;
  core23::Tensor bucket_id_tensor;
  core23::Tensor bucket_position_tensor;
  core23::Tensor max_hotness_tensor;
};

TYPED_TEST_CASE_P(SplitBatchFixture);

TYPED_TEST_P(SplitBatchFixture, split_feat_major_one_hot) {
  using DenseType = typename std::tuple_element<0, TypeParam>::type;
  using SparseType = typename std::tuple_element<1, TypeParam>::type;

  std::vector<int> nnz_per_slot(sparse_dim, 1);
  this->init_sparse_data(nnz_per_slot);

  split_3_way_feat_major<DenseType, SparseType>(
      this->label_tensor, this->dense_tensor, this->sparse_tensor_ptrs, this->label_dense_sparse,
      this->bucket_id_tensor, this->bucket_position_tensor, this->max_hotness_tensor, NULL);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  this->check_label();
  this->check_dense();
  this->check_sparse(nnz_per_slot);
}

TYPED_TEST_P(SplitBatchFixture, split_feat_major_multi_hot) {
  using DenseType = typename std::tuple_element<0, TypeParam>::type;
  using SparseType = typename std::tuple_element<1, TypeParam>::type;

  std::vector<int> nnz_per_slot;
  for (int i = 0; i < sparse_dim; ++i) {
    nnz_per_slot.push_back((rand() % 100) + 1);
  }

  this->init_sparse_data(nnz_per_slot);

  split_3_way_feat_major<DenseType, SparseType>(
      this->label_tensor, this->dense_tensor, this->sparse_tensor_ptrs, this->label_dense_sparse,
      this->bucket_id_tensor, this->bucket_position_tensor, this->max_hotness_tensor, NULL, false);
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
