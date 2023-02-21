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

#include <cub/cub.cuh>
#include <embedding/operators/transpose_input.hpp>
#include <utils.hpp>

namespace embedding {

namespace {

template <typename offset_t>
__global__ void convert_batch_major_to_feature_major_for_bucket_range_kernel(
    const offset_t *bucket_range, int num_lookup, int batch_size,
    offset_t *feature_major_bucket_range) {
  for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < batch_size * num_lookup;
       tid += blockDim.x * gridDim.x) {
    int lookup_id = tid % num_lookup;  // lookup major
    int batch_id = tid / num_lookup;   // lookup major

    int hotness = bucket_range[tid + 1] - bucket_range[tid];

    int feature_major_idx = lookup_id * batch_size + batch_id;
    feature_major_bucket_range[1 + feature_major_idx] = hotness;
  }
  if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
    feature_major_bucket_range[0] = 0;
  }
}

template <typename key_t, typename offset_t>
__global__ void convert_batch_major_to_feature_major_for_key_kernel(
    const key_t *batch_major_key, const offset_t *batch_major_bucket_range,
    const offset_t *feature_major_bucket_range, int num_lookup, int batch_size,
    key_t *feature_major_key) {
  for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < batch_size * num_lookup;
       tid += blockDim.x * gridDim.x) {
    offset_t start = batch_major_bucket_range[tid];
    offset_t end = batch_major_bucket_range[tid + 1];

    int lookup_id = tid % num_lookup;  // lookup major
    int batch_id = tid / num_lookup;   // lookup major
    int feature_major_idx = lookup_id * batch_size + batch_id;
    offset_t feature_major_start = feature_major_bucket_range[feature_major_idx];
    for (uint32_t i = 0; i < (end - start); ++i) {
      feature_major_key[feature_major_start + i] = batch_major_key[start + i];
    }
  }
}

}  // namespace

PreprocessInput::PreprocessInput(std::shared_ptr<CoreResourceManager> core,
                                 const EmbeddingCollectionParam &ebc_param)
    : core_(core), input_layout_(ebc_param.input_layout_), num_lookup_(ebc_param.num_lookup) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  int universal_batch_size = ebc_param.universal_batch_size;
  auto key_type = ebc_param.key_type;
  auto offset_type = ebc_param.offset_type;

  int max_hotness_sum = 0;
  for (auto &param : ebc_param.lookup_params) {
    max_hotness_sum += param.max_hotness;
  }

  feature_major_key_ =
      core23::Tensor(params.shape({universal_batch_size, max_hotness_sum}).data_type(key_type));
  feature_major_bucket_range_ = core23::Tensor(
      params.shape({universal_batch_size * ebc_param.num_lookup + 1}).data_type(offset_type));

  {
    size_t temp_bytes = 0;
    DISPATCH_INTEGRAL_FUNCTION_CORE23(offset_type.type(), offset_t, [&] {
      cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, (offset_t *)nullptr, (offset_t *)nullptr,
                                    universal_batch_size * ebc_param.num_lookup + 1);
    });
    d_temp_scan_storage_ = core23::Tensor(
        params.shape({static_cast<int64_t>(temp_bytes)}).data_type(core23::ScalarType::Char));
  }
}

void PreprocessInput::compute(const core23::Tensor &key, const core23::Tensor &bucket_range,
                              core23::Tensor *feature_major_key,
                              core23::Tensor *feature_major_bucket_range, int batch_size) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());

  if (input_layout_ == EmbeddingLayout::FeatureMajor) {  // feature major
    *feature_major_key = key;
    *feature_major_bucket_range = bucket_range;
  } else {
    *feature_major_key = feature_major_key_;
    *feature_major_bucket_range = feature_major_bucket_range_;
    DISPATCH_INTEGRAL_FUNCTION_CORE23(bucket_range.data_type().type(), offset_t, [&] {
      auto stream = core_->get_local_gpu()->get_stream();
      HCTR_LIB_THROW(cudaMemsetAsync(feature_major_bucket_range_.data<offset_t>(), 0,
                                     feature_major_bucket_range_.num_bytes(), stream));

      {
        constexpr int block_size = 256;
        int grid_size = (bucket_range.num_elements() - 1) / block_size + 1;
        convert_batch_major_to_feature_major_for_bucket_range_kernel<<<grid_size, block_size, 0,
                                                                       stream>>>(
            bucket_range.data<offset_t>(), num_lookup_, batch_size,
            feature_major_bucket_range_.data<offset_t>());
      }

      size_t d_temp_scan_storage_nbytes = d_temp_scan_storage_.num_bytes();
      cub::DeviceScan::InclusiveSum(d_temp_scan_storage_.data(), d_temp_scan_storage_nbytes,
                                    feature_major_bucket_range_.data<offset_t>(),
                                    feature_major_bucket_range_.data<offset_t>(),
                                    feature_major_bucket_range_.num_elements(), stream);
    });

    DISPATCH_INTEGRAL_FUNCTION_CORE23(key.data_type().type(), key_t, [&] {
      DISPATCH_INTEGRAL_FUNCTION_CORE23(bucket_range.data_type().type(), offset_t, [&] {
        auto stream = core_->get_local_gpu()->get_stream();
        HCTR_LIB_THROW(cudaMemsetAsync(feature_major_key_.data<key_t>(), 0,
                                       feature_major_key_.num_bytes(), stream));

        constexpr int block_size = 256;
        int grid_size = (bucket_range.num_elements() - 2) / block_size + 1;
        convert_batch_major_to_feature_major_for_key_kernel<<<grid_size, block_size, 0, stream>>>(
            key.data<key_t>(), bucket_range.data<offset_t>(),
            feature_major_bucket_range_.data<offset_t>(), num_lookup_, batch_size,
            feature_major_key_.data<key_t>());
      });
    });
  }
}

}  // namespace embedding