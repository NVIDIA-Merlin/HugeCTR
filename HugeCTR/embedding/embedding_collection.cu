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
#include <cub/cub.cuh>

#include "core/registry.hpp"
#include "embedding_collection.hpp"

namespace embedding {

namespace {

template <typename offset_t>
__global__ void transpose_and_flatten_bucket_range_kernel(
    const offset_t *bucket_range, int num_embedding, int batch_size, offset_t *t_num_key,
    const char *combiner_list, const int *embedding_offset, bool transpose) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (tid  < batch_size * num_embedding) {
    int hotness = bucket_range[tid + 1] - bucket_range[tid];

    int embedding_id = transpose ? tid % num_embedding : tid / batch_size;
    int batch_id = transpose ? tid / num_embedding : tid % batch_size;

    if (combiner_list[embedding_id] == static_cast<char>(Combiner::Concat)) {
      for (int i = 0; i < hotness; ++i) {
        int flatten_embedding_id = embedding_offset[embedding_id] + i;
        int t_idx = flatten_embedding_id * batch_size + batch_id;
        t_num_key[t_idx + 1] = 1;
      }
    } else {
      int flatten_embedding_id = embedding_offset[embedding_id];
      int t_idx = flatten_embedding_id * batch_size + batch_id;
      t_num_key[t_idx + 1] = hotness;
    }

    if (tid == 0) {
      t_num_key[0] = 0;
    }
  }
}

template <typename key_t, typename offset_t>
__global__ void transpose_key_kernel(const key_t *key, size_t num_key, const offset_t *bucket_range,
                                    int num_embedding, int batch_size,
                                     const offset_t *t_bucket_range, const char *combiner_list,
                                     const int *embedding_offset, key_t *t_key, bool transpose) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < batch_size * num_embedding) {
    offset_t start = bucket_range[tid];
    offset_t end = bucket_range[tid + 1];
    
    int embedding_id = transpose ? tid % num_embedding : tid / batch_size;
    int batch_id = transpose ? tid / num_embedding : tid % batch_size;

    if (combiner_list[embedding_id] == static_cast<char>(Combiner::Concat)) {
      for (int i = 0; i < static_cast<int>(end - start); ++i) {
        int flatten_embedding_id = embedding_offset[embedding_id] + i;
        int t_idx = flatten_embedding_id * batch_size + batch_id;
        offset_t t_start = t_bucket_range[t_idx];
        t_key[t_start] = key[start + i];
      }
    } else {
      int flatten_embedding_id = embedding_offset[embedding_id];
      int t_idx = flatten_embedding_id * batch_size + batch_id;
      offset_t t_start = t_bucket_range[t_idx];

      for (uint32_t i = 0; i < (end - start); ++i) {
        t_key[t_start + i] = key[i + start];
      }
    }
  }
}

template <typename emb_t>
__global__ void transpose_concat_embedding_forward_kernel(const emb_t *output_buffer, int num_ev,
const int *start_embedding_id_list, const int *original_embedding_id_list, const char *flatten_combiner_list, const int *flatten_ev_size_list, const int *flatten_ev_offset_list, const int *hotness_list, int batch_size, emb_t *t_output_buffer) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (tid < num_ev) {
    int embedding_id = tid / batch_size;
    int batch_id = tid % batch_size;

    int ev_size = flatten_ev_size_list[embedding_id];
    int start_idx = batch_size * flatten_ev_offset_list[embedding_id] + batch_id * ev_size;

    if (flatten_combiner_list[embedding_id] == static_cast<char>(Combiner::Concat)) {
      int start_embedding_id = start_embedding_id_list[embedding_id];
      int hotness = hotness_list[original_embedding_id_list[embedding_id]];

      int idx_hotness = embedding_id - start_embedding_id;
      int t_start_idx = batch_size * flatten_ev_offset_list[start_embedding_id] + (batch_id * hotness + idx_hotness) * ev_size;
      for (int i = 0; i < ev_size; ++i) {
        t_output_buffer[t_start_idx + i] = output_buffer[start_idx + i];
      }
    } else {
      for (int i = 0; i < ev_size; ++i) {
        t_output_buffer[start_idx + i] = output_buffer[start_idx + i];
      }
    }
  }
}


template <typename emb_t>
__global__ void transpose_concat_embedding_backward_kernel(const emb_t *output_buffer, int num_ev,
const int *start_embedding_id_list, const int *original_embedding_id_list, const char *flatten_combiner_list, const int *flatten_ev_size_list, const int *flatten_ev_offset_list, const int *hotness_list, int batch_size, emb_t *t_output_buffer) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (tid < num_ev) {
    int embedding_id = tid / batch_size;
    int batch_id = tid % batch_size;

    int ev_size = flatten_ev_size_list[embedding_id];
    int start_idx = batch_size * flatten_ev_offset_list[embedding_id] + batch_id * ev_size;

    if (flatten_combiner_list[embedding_id] == static_cast<char>(Combiner::Concat)) {
      int start_embedding_id = start_embedding_id_list[embedding_id];
      int hotness = hotness_list[original_embedding_id_list[embedding_id]];

      int bucket_id = embedding_id - start_embedding_id + batch_id * hotness + start_embedding_id * batch_size;
      int flatten_embedding_id = bucket_id / batch_size;
      int batch_id = bucket_id % batch_size;
      int t_start_idx = batch_size * flatten_ev_offset_list[flatten_embedding_id] + batch_id * ev_size;
      
      for (int i = 0; i < ev_size; ++i) {
        t_output_buffer[start_idx + i] = output_buffer[t_start_idx + i];
      }
    } else {
      for (int i = 0; i < ev_size; ++i) {
        t_output_buffer[start_idx + i] = output_buffer[start_idx + i];
      }
    }
  }
}
}  // namespace

PreprocessInput::PreprocessInput(std::shared_ptr<CoreResourceManager> core,
                                 const EmbeddingCollectionParam &ebc_param)
    : core_(core),
      num_embedding_(ebc_param.num_embedding),
      num_flatten_embedding_(0),
      is_table_first_input_(ebc_param.is_table_first_input) {
  CudaDeviceContext ctx(core_->get_device_id());
  Device device{DeviceType::GPU};
  int universal_batch_size = ebc_param.universal_batch_size;
  auto key_type = ebc_param.key_type;
  auto offset_type = ebc_param.offset_type;

  int hotness_sum = 0;
  std::vector<char> combiner_list;
  for (auto &param : ebc_param.embedding_params) {
    combiner_list.push_back(static_cast<char>(param.combiner));
    hotness_sum += param.hotness;
  }

  std::vector<int> flatten_concat_embedding_offset;
  for (int embedding_id = 0; embedding_id < num_embedding_; ++embedding_id) {
    flatten_concat_embedding_offset.push_back(num_flatten_embedding_);
    num_flatten_embedding_ +=
        (ebc_param.embedding_params[embedding_id].combiner == Combiner::Concat)
            ? ebc_param.embedding_params[embedding_id].hotness
            : 1;
  }
  flatten_concat_embedding_offset.push_back(num_flatten_embedding_);

  auto buffer = GetBuffer(core);
  t_key_ = buffer->reserve({universal_batch_size, hotness_sum}, device, key_type);
  flatten_t_bucket_range_ =
      buffer->reserve({universal_batch_size * num_flatten_embedding_ + 1}, device, offset_type);

  d_embedding_offset_ =
      buffer->reserve({num_embedding_ + 1}, DeviceType::GPU, TensorScalarType::Int32);
  d_combiner_list_ =
      buffer->reserve({combiner_list.size()}, DeviceType::GPU, TensorScalarType::Char);

  {
    size_t temp_bytes = 0;
    DISPATCH_INTEGRAL_FUNCTION(offset_type.type(), offset_t, [&] {
      cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, (offset_t *)nullptr, (offset_t *)nullptr,
                                    universal_batch_size * num_flatten_embedding_ + 1);
    });
    d_temp_scan_storage_ = buffer->reserve({temp_bytes}, device, TensorScalarType::Void);
  }
  buffer->allocate();

  d_embedding_offset_.copy_from(flatten_concat_embedding_offset);
  d_combiner_list_.copy_from(combiner_list);
}

void PreprocessInput::compute(const Tensor &key, const Tensor &bucket_range, size_t num_key,
                              Tensor *preprocessed_key, Tensor *preprocessed_bucket_range) {
  CudaDeviceContext ctx(core_->get_device_id());
  int batch_size = (bucket_range.get_num_elements() - 1) / num_embedding_;

  *preprocessed_key = key;
  *preprocessed_bucket_range = bucket_range;

  if (!is_table_first_input_ || (num_embedding_ != num_flatten_embedding_)) {
    DISPATCH_INTEGRAL_FUNCTION(bucket_range.dtype().type(), offset_t, [&] {
      auto stream = core_->get_local_gpu()->get_stream();
      HCTR_LIB_THROW(
        cudaMemsetAsync(flatten_t_bucket_range_.get<offset_t>(), 0, flatten_t_bucket_range_.nbytes(), stream));

      {
        constexpr int block_size = 256;
        int grid_size = (bucket_range.get_num_elements() - 1) / block_size + 1;
        transpose_and_flatten_bucket_range_kernel<<<grid_size, block_size, 0, stream>>>(
            bucket_range.get<offset_t>(), num_embedding_, batch_size,
            flatten_t_bucket_range_.get<offset_t>(), d_combiner_list_.get<char>(),
            d_embedding_offset_.get<int>(), !is_table_first_input_);
      }

      size_t d_temp_scan_storage_nbytes = d_temp_scan_storage_.nbytes();
      cub::DeviceScan::InclusiveSum(d_temp_scan_storage_.get(), d_temp_scan_storage_nbytes,
                                    flatten_t_bucket_range_.get<offset_t>(),
                                    flatten_t_bucket_range_.get<offset_t>(),
                                    flatten_t_bucket_range_.get_num_elements(), stream);
    });
    DISPATCH_INTEGRAL_FUNCTION(key.dtype().type(), key_t, [&] {
      DISPATCH_INTEGRAL_FUNCTION(bucket_range.dtype().type(), offset_t, [&] {
        auto stream = core_->get_local_gpu()->get_stream();
        HCTR_LIB_THROW(
          cudaMemsetAsync(t_key_.get<key_t>(), 0, t_key_.nbytes(), stream));

        constexpr int block_size = 256;
        int grid_size = (bucket_range.get_num_elements() - 2) / block_size + 1;
        transpose_key_kernel<<<grid_size, block_size, 0, stream>>>(
            key.get<key_t>(), num_key, bucket_range.get<offset_t>(),
            num_embedding_, batch_size,
            flatten_t_bucket_range_.get<offset_t>(), d_combiner_list_.get<char>(),
            d_embedding_offset_.get<int>(), t_key_.get<key_t>(), !is_table_first_input_);
      });
    });
    *preprocessed_key = t_key_;
    *preprocessed_bucket_range = flatten_t_bucket_range_;
  }
}

ProcessOutput::ProcessOutput(std::shared_ptr<CoreResourceManager> core,
                             const EmbeddingCollectionParam &ebc_param)
    : core_(core), num_flatten_embedding_(0) {
  CudaDeviceContext ctx(core_->get_device_id());

  std::vector<int> original_embedding_id_list;
  std::vector<int> start_embedding_id_list;
  std::vector<int> hotness_list;
  for (int embedding_id  = 0; embedding_id < ebc_param.num_embedding; ++embedding_id) {
    auto &emb_param = ebc_param.embedding_params[embedding_id];
    hotness_list.push_back(emb_param.hotness);
    
    if (emb_param.combiner == Combiner::Concat) {
      for (int i = 0; i < emb_param.hotness; ++i) {
        start_embedding_id_list.push_back(num_flatten_embedding_);
        original_embedding_id_list.push_back(embedding_id);
      }
      num_flatten_embedding_ += emb_param.hotness;
    } else {
      start_embedding_id_list.push_back(num_flatten_embedding_);
      original_embedding_id_list.push_back(embedding_id);
      num_flatten_embedding_ += 1;
    }
  }
  start_embedding_id_list.push_back(num_flatten_embedding_);

  int num_emb_vec = 0;
  int num_gpus = core_->get_global_gpu_count();
  for (size_t i = 0; i < ebc_param.embedding_params.size(); ++i) {
    auto &emb_param = ebc_param.embedding_params[i];
    if (emb_param.combiner == Combiner::Concat) {
      num_emb_vec += ebc_param.universal_batch_size * emb_param.ev_size * emb_param.hotness / num_gpus;
    } else {
      num_emb_vec += ebc_param.universal_batch_size * emb_param.ev_size / num_gpus;
    }
  }

  auto buffer = GetBuffer(core);
  original_embedding_id_list_ = buffer->reserve(original_embedding_id_list.size(), DeviceType::GPU, TensorScalarType::Int32);
  start_embedding_id_list_ = buffer->reserve(start_embedding_id_list.size(), DeviceType::GPU, TensorScalarType::Int32);
  hotness_list_ = buffer->reserve(hotness_list.size(), DeviceType::GPU, TensorScalarType::Int32);
  output_buffer_ = buffer->reserve(num_emb_vec, DeviceType::GPU, ebc_param.emb_type);
  buffer->allocate();

  original_embedding_id_list_.copy_from(original_embedding_id_list);
  start_embedding_id_list_.copy_from(start_embedding_id_list);
  hotness_list_.copy_from(hotness_list);
}

void ProcessOutput::compute(const Tensor &flatten_combiner_list, const Tensor &flatten_ev_size_list, const Tensor &flatten_ev_offset_list, Tensor &output_buffer, int batch_size) {
  CudaDeviceContext ctx(core_->get_device_id());

  DISPATCH_FLOAT_AND_HALF_FUNCTION(output_buffer.dtype().type(), emb_t, [&] {
    auto stream = core_->get_local_gpu()->get_stream();
    int num_gpus = core_->get_global_gpu_count();
    int batch_size_per_gpu = batch_size / num_gpus;
    
    int num_ev = batch_size_per_gpu * num_flatten_embedding_;
    int block_size = 256;
    int grid_size = (num_ev - 1) / block_size + 1;
    transpose_concat_embedding_forward_kernel<<<grid_size, block_size, 0, stream>>>(output_buffer.get<emb_t>(), num_ev, start_embedding_id_list_.get<int>(), original_embedding_id_list_.get<int>(), flatten_combiner_list.get<char>(), flatten_ev_size_list.get<int>(), flatten_ev_offset_list.get<int>(), hotness_list_.get<int>(), batch_size_per_gpu , output_buffer_.get<emb_t>());
    
    HCTR_LIB_THROW(cudaMemcpyAsync(output_buffer.get(), output_buffer_.get(), output_buffer.nbytes(), cudaMemcpyDeviceToDevice, stream));
  });
}


ProcessOutputBackward::ProcessOutputBackward(std::shared_ptr<CoreResourceManager> core,
                             const EmbeddingCollectionParam &ebc_param)
    : core_(core), num_flatten_embedding_(0) {
  CudaDeviceContext ctx(core_->get_device_id());

  std::vector<int> original_embedding_id_list;
  std::vector<int> start_embedding_id_list;
  std::vector<int> hotness_list;
  for (int embedding_id  = 0; embedding_id < ebc_param.num_embedding; ++embedding_id) {
    auto &emb_param = ebc_param.embedding_params[embedding_id];
    hotness_list.push_back(emb_param.hotness);
    
    if (emb_param.combiner == Combiner::Concat) {
      for (int i = 0; i < emb_param.hotness; ++i) {
        start_embedding_id_list.push_back(num_flatten_embedding_);
        original_embedding_id_list.push_back(embedding_id);
      }
      num_flatten_embedding_ += emb_param.hotness;
    } else {
      start_embedding_id_list.push_back(num_flatten_embedding_);
      original_embedding_id_list.push_back(embedding_id);
      num_flatten_embedding_ += 1;
    }
  }
  start_embedding_id_list.push_back(num_flatten_embedding_);

  int num_emb_vec = 0;
  int num_gpus = core_->get_global_gpu_count();
  for (size_t i = 0; i < ebc_param.embedding_params.size(); ++i) {
    auto &emb_param = ebc_param.embedding_params[i];
    if (emb_param.combiner == Combiner::Concat) {
      num_emb_vec += ebc_param.universal_batch_size * emb_param.ev_size * emb_param.hotness / num_gpus;
    } else {
      num_emb_vec += ebc_param.universal_batch_size * emb_param.ev_size / num_gpus;
    }
  }

  auto buffer = GetBuffer(core);
  original_embedding_id_list_ = buffer->reserve(original_embedding_id_list.size(), DeviceType::GPU, TensorScalarType::Int32);
  start_embedding_id_list_ = buffer->reserve(start_embedding_id_list.size(), DeviceType::GPU, TensorScalarType::Int32);
  hotness_list_ = buffer->reserve(hotness_list.size(), DeviceType::GPU, TensorScalarType::Int32);
  output_buffer_ = buffer->reserve(num_emb_vec, DeviceType::GPU, ebc_param.emb_type);
  buffer->allocate();

  original_embedding_id_list_.copy_from(original_embedding_id_list);
  start_embedding_id_list_.copy_from(start_embedding_id_list);
  hotness_list_.copy_from(hotness_list);
}

void ProcessOutputBackward::compute(const Tensor &flatten_combiner_list, const Tensor &flatten_ev_size_list, const Tensor &flatten_ev_offset_list, Tensor &output_buffer, int batch_size_per_gpu, Tensor *t_output_buffer) {
  CudaDeviceContext ctx(core_->get_device_id());

  DISPATCH_FLOAT_AND_HALF_FUNCTION(output_buffer.dtype().type(), emb_t, [&] {
    auto stream = core_->get_local_gpu()->get_stream();
    int num_gpus = core_->get_global_gpu_count();

    int num_ev = batch_size_per_gpu * num_flatten_embedding_;
    int block_size = 256;
    int grid_size = (num_ev - 1) / block_size + 1;
    transpose_concat_embedding_backward_kernel<<<grid_size, block_size, 0, stream>>>(output_buffer.get<emb_t>(), num_ev, start_embedding_id_list_.get<int>(), original_embedding_id_list_.get<int>(), flatten_combiner_list.get<char>(), flatten_ev_size_list.get<int>(), flatten_ev_offset_list.get<int>(), hotness_list_.get<int>(), batch_size_per_gpu , output_buffer_.get<emb_t>());
  });
  *t_output_buffer = output_buffer_;
}
}  // namespace embedding