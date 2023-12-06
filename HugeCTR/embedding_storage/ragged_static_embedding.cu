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

#include <curand_kernel.h>
#include <float.h>

#include <data_simulator.hpp>
#include <embedding/operators/generic_lookup.cuh>
#include <embedding/view.hpp>
#include <embedding_storage/ragged_static_embedding.hpp>
#include <numeric>
#include <utils.cuh>

namespace embedding {

namespace {
// the input keys are already unique by adding offset
// (key offset is emb_table_id_space_offset) (embedding table offset is emb_table_ev_offset)
template <typename key_t, typename offset_t, typename index_t>
__global__ void ragged_static_embedding_table_lookup_kernel(
    const key_t *keys, size_t num_keys, const offset_t *id_space_offset, size_t num_id_space_offset,
    const int *id_space_list, const int *local_id_space_list, size_t num_local_id_space_list,
    const index_t *emb_table_id_space_offset, float *emb_table, const uint64_t *emb_table_ev_offset,
    const int *local_ev_size_list, float **emb_vec) {
  CUDA_1D_KERNEL_LOOP_T(offset_t, tid, num_keys) {
    int64_t id_space_idx = bs_upper_bound_sub_one(id_space_offset, num_id_space_offset, tid);
    int id_space = id_space_list[id_space_idx];
    int64_t local_id_space_idx =
        bs_upper_bound_sub_one(local_id_space_list, num_local_id_space_list, id_space);

    uint64_t start = (uint64_t)emb_table_id_space_offset[local_id_space_idx];
    uint64_t ev_offset = emb_table_ev_offset[local_id_space_idx];
    int ev_size = local_ev_size_list[local_id_space_idx];
    // printf("lookup key is %llu, ev_offset %llu, start %llu, dst ptr %p\n", (uint64_t)keys[tid],
    //        ev_offset, start, emb_vec + tid);
    emb_vec[tid] = &emb_table[ev_offset + ((uint64_t)keys[tid] - start) * ev_size];
  }
}

template <typename key_t, typename index_t>
struct RaggedKeyToIndicesFunc {
  int *local_table_ids;
  int *local_ev_sizes;
  int64_t num_local_table_ids;

  index_t *emb_table_id_space_offset;
  uint64_t *emb_table_ev_start_indices;

  DEVICE_INLINE void operator()(const key_t &key, const int &table_id,
                                uint64_t *ev_start_indices_ptr, int *ev_size_ptr) {
    int local_id_space_idx = bs_upper_bound_sub_one(local_table_ids, num_local_table_ids, table_id);
    assert(local_id_space_idx >= 0);
    assert(local_id_space_idx < num_local_table_ids);
    index_t start = emb_table_id_space_offset[local_id_space_idx];

    uint64_t ev_offset = emb_table_ev_start_indices[local_id_space_idx];
    int ev_size = local_ev_sizes[local_id_space_idx];

    *ev_start_indices_ptr = ev_offset + static_cast<uint64_t>(key - start) * ev_size;
    *ev_size_ptr = ev_size;
  }
};

template <typename wgrad_t>
struct OptimizierInput {
  const wgrad_t *wgrad;
  uint64_t ev_start_indices;
  int ev_id;
  float lr;
  float scaler;
};

constexpr int num_load_floats = 4;
template <typename wgrad_t>
struct SGDOptimizer {
  DEVICE_INLINE void update4(const OptimizierInput<wgrad_t> &input, float *ev) {
    Vec4T<float> gi;
    gi.load(input.wgrad + input.ev_id, 4);
    Vec4T<float> ev_plus_gi;
    ev_plus_gi.load(ev + input.ev_id, 4);
    Vec4T<float> ret;
    ret.val.x = -input.lr * gi.val.x / input.scaler;
    ret.val.y = -input.lr * gi.val.y / input.scaler;
    ret.val.z = -input.lr * gi.val.z / input.scaler;
    ret.val.w = -input.lr * gi.val.w / input.scaler;
    ev_plus_gi.accumulate(ret);
    ev_plus_gi.store(ev + input.ev_id, 4);
  }

  DEVICE_INLINE void update(const OptimizierInput<wgrad_t> &input, float *ev) {
    ev[input.ev_id] =
        ev[input.ev_id] -
        input.lr * (HugeCTR::TypeConvertFunc<float, wgrad_t>::convert(input.wgrad[input.ev_id]) /
                    input.scaler);
  }
};

template <typename wgrad_t, typename acc_t>
struct AdaGradOptimizer {
  acc_t *v;
  float epsilon;

  DEVICE_INLINE void update4(const OptimizierInput<wgrad_t> &input, float *ev) {
    Vec4T<float> vi;
    vi.load(v + input.ev_start_indices + input.ev_id, 4);
    Vec4T<float> gi;
    gi.load(input.wgrad + input.ev_id, 4);
    Vec4T<float> ev_plus_gi;
    ev_plus_gi.load(ev + input.ev_id, 4);

    gi.val.x = gi.val.x / input.scaler;
    gi.val.y = gi.val.y / input.scaler;
    gi.val.z = gi.val.z / input.scaler;
    gi.val.w = gi.val.w / input.scaler;
    vi.val.x = vi.val.x + gi.val.x * gi.val.x;
    vi.val.y = vi.val.y + gi.val.y * gi.val.y;
    vi.val.z = vi.val.z + gi.val.z * gi.val.z;
    vi.val.w = vi.val.w + gi.val.w * gi.val.w;

    gi.val.x = -input.lr * gi.val.x / (sqrtf(vi.val.x) + epsilon);
    gi.val.y = -input.lr * gi.val.y / (sqrtf(vi.val.y) + epsilon);
    gi.val.z = -input.lr * gi.val.z / (sqrtf(vi.val.z) + epsilon);
    gi.val.w = -input.lr * gi.val.w / (sqrtf(vi.val.w) + epsilon);

    vi.store(v + input.ev_start_indices + input.ev_id, 4);

    ev_plus_gi.accumulate(gi);

    ev_plus_gi.store(ev + input.ev_id, 4);
  }

  DEVICE_INLINE void update(const OptimizierInput<wgrad_t> &input, float *ev) {
    float vi =
        HugeCTR::TypeConvertFunc<float, acc_t>::convert(v[input.ev_start_indices + input.ev_id]);
    float gi = HugeCTR::TypeConvertFunc<float, wgrad_t>::convert(input.wgrad[input.ev_id]);
    gi = gi / input.scaler;
    vi = vi + gi * gi;

    gi = -input.lr * gi / (sqrtf(vi) + epsilon);
    v[input.ev_start_indices + input.ev_id] = HugeCTR::TypeConvertFunc<acc_t, float>::convert(vi);
    ev[input.ev_id] += gi;
  }
};

template <typename wgrad_t, typename opt_t>
struct FtrlOptimizer {
  opt_t *z;
  opt_t *n;

  float beta;
  float lambda1;
  float lambda2;
  /**
   * FTRL
   * ----
      ```
  Update rule for one variable `w`:
  ```python
  prev_n = n
  n = n + g ** 2
  sigma = (sqrt(n) - sqrt(prev_n)) / lr
  z = z + g - sigma * w
  if abs(z) < lambda_1:
    w = 0
  else:
    w = (sgn(z) * lambda_1 - z) / ((beta + sqrt(n)) / lr + lambda_2)
  ```
   */
  DEVICE_INLINE void update4(const OptimizierInput<wgrad_t> &input, float *ev) {
    float lambda2_plus_beta_div_lr = lambda2 + beta / input.lr;
    Vec4T<float> zi;
    zi.load(z + input.ev_start_indices + input.ev_id, 4);
    Vec4T<float> ni;
    ni.load(n + input.ev_start_indices + input.ev_id, 4);
    Vec4T<float> weight;
    weight.load(ev + input.ev_id, 4);

    Vec4T<float> sigma;
    Vec4T<float> sqrt_ni;
    Vec4T<float> sqrt_ni_new;
    Vec4T<float> p;
    Vec4T<float> q;

    Vec4T<float> gi;
    gi.load(input.wgrad + input.ev_id, 4);

    gi.val.x = gi.val.x / input.scaler;
    gi.val.y = gi.val.y / input.scaler;
    gi.val.z = gi.val.z / input.scaler;
    gi.val.w = gi.val.w / input.scaler;

    sqrt_ni.val.x = sqrtf(ni.val.x + FLT_EPSILON);
    sqrt_ni.val.y = sqrtf(ni.val.y + FLT_EPSILON);
    sqrt_ni.val.z = sqrtf(ni.val.z + FLT_EPSILON);
    sqrt_ni.val.w = sqrtf(ni.val.w + FLT_EPSILON);
    ni.val.x = ni.val.x + gi.val.x * gi.val.x;
    ni.val.y = ni.val.y + gi.val.y * gi.val.y;
    ni.val.z = ni.val.z + gi.val.z * gi.val.z;
    ni.val.w = ni.val.w + gi.val.w * gi.val.w;
    sqrt_ni_new.val.x = sqrtf(ni.val.x + FLT_EPSILON);
    sqrt_ni_new.val.y = sqrtf(ni.val.y + FLT_EPSILON);
    sqrt_ni_new.val.z = sqrtf(ni.val.z + FLT_EPSILON);
    sqrt_ni_new.val.w = sqrtf(ni.val.w + FLT_EPSILON);

    sigma.val.x = (sqrt_ni_new.val.x - sqrt_ni.val.x) / input.lr;
    sigma.val.y = (sqrt_ni_new.val.y - sqrt_ni.val.y) / input.lr;
    sigma.val.z = (sqrt_ni_new.val.z - sqrt_ni.val.z) / input.lr;
    sigma.val.w = (sqrt_ni_new.val.w - sqrt_ni.val.w) / input.lr;

    zi.val.x = zi.val.x + gi.val.x - sigma.val.x * weight.val.x;
    zi.val.y = zi.val.y + gi.val.y - sigma.val.y * weight.val.y;
    zi.val.z = zi.val.z + gi.val.z - sigma.val.z * weight.val.z;
    zi.val.w = zi.val.w + gi.val.w - sigma.val.w * weight.val.w;

    p.val.x = (1.f - 2.f * signbit(zi.val.x)) * lambda1 - zi.val.x;
    p.val.y = (1.f - 2.f * signbit(zi.val.y)) * lambda1 - zi.val.y;
    p.val.z = (1.f - 2.f * signbit(zi.val.z)) * lambda1 - zi.val.z;
    p.val.w = (1.f - 2.f * signbit(zi.val.w)) * lambda1 - zi.val.w;

    q.val.x = sqrt_ni_new.val.x / input.lr + lambda2_plus_beta_div_lr;
    q.val.y = sqrt_ni_new.val.y / input.lr + lambda2_plus_beta_div_lr;
    q.val.z = sqrt_ni_new.val.z / input.lr + lambda2_plus_beta_div_lr;
    q.val.w = sqrt_ni_new.val.w / input.lr + lambda2_plus_beta_div_lr;

    //(p / q) * signbit(lambda1 - abs(zi)) - wi

    weight.val.x = p.val.x / q.val.x * signbit(lambda1 - abs(zi.val.x));
    weight.val.y = p.val.y / q.val.y * signbit(lambda1 - abs(zi.val.y));
    weight.val.z = p.val.z / q.val.z * signbit(lambda1 - abs(zi.val.z));
    weight.val.w = p.val.w / q.val.w * signbit(lambda1 - abs(zi.val.w));

    ni.store(n + input.ev_start_indices + input.ev_id, 4);
    zi.store(z + input.ev_start_indices + input.ev_id, 4);
    weight.store(ev + input.ev_id, 4);
  }

  DEVICE_INLINE void update(const OptimizierInput<wgrad_t> &input, float *ev) {
    float lambda2_plus_beta_div_lr = lambda2 + beta / input.lr;
    float ni =
        HugeCTR::TypeConvertFunc<float, opt_t>::convert(n[input.ev_start_indices + input.ev_id]);
    float zi =
        HugeCTR::TypeConvertFunc<float, opt_t>::convert(z[input.ev_start_indices + input.ev_id]);
    float gi = HugeCTR::TypeConvertFunc<float, wgrad_t>::convert(input.wgrad[input.ev_id]);

    gi = gi / input.scaler;
    float sqrt_ni = sqrtf(ni + FLT_EPSILON);
    ni = ni + gi * gi;
    float sqrt_ni_new = sqrtf(ni + FLT_EPSILON);

    float sigma = (sqrt_ni_new - sqrt_ni) / input.lr;
    zi = zi + gi - sigma * ev[input.ev_id];

    float p = (1.f - 2.f * signbit(zi)) * lambda1 - zi;
    float q = sqrt_ni_new / input.lr + lambda2_plus_beta_div_lr;

    n[input.ev_start_indices + input.ev_id] = HugeCTR::TypeConvertFunc<opt_t, float>::convert(ni);
    z[input.ev_start_indices + input.ev_id] = HugeCTR::TypeConvertFunc<opt_t, float>::convert(zi);
    ev[input.ev_id] = p / q * signbit(lambda1 - abs(zi));
  }
};

template <typename key_t, typename index_t, typename wgrad_t, typename OptimizerFunc,
          typename KeyToIndicesFunc>
__global__ void update4_kernel(const key_t *keys, const size_t *num_keys_ptr, const int *table_ids,
                               const wgrad_t *grad_ev, const uint32_t *ev_start_indices,
                               KeyToIndicesFunc key_to_indices_func, float *emb_table,
                               OptimizerFunc optimizer, float lr, float scaler) {
  if (*num_keys_ptr == 0) return;
  size_t num_steps = (*num_keys_ptr - 1) / (blockDim.x * gridDim.x) + 1;
  for (size_t step = 0; step < num_steps; step++) {
    size_t tid = step * blockDim.x * gridDim.x + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t emb_table_ev_start_indices_frag;
    int ev_size_frag = std::numeric_limits<int>::max();
    uint32_t grad_ev_offset_frag;
    if (tid < *num_keys_ptr) {
      key_t key = keys[tid];
      int table_id = table_ids[tid];
      key_to_indices_func(key, table_id, &emb_table_ev_start_indices_frag, &ev_size_frag);
      grad_ev_offset_frag = ev_start_indices[tid];
    }

    for (int lane_id = 0; lane_id < warpSize; lane_id++) {
      int ev_size = __shfl_sync(0xffffffff, ev_size_frag, lane_id);
      if (ev_size == std::numeric_limits<int>::max()) {
        break;
      }
      const wgrad_t *grad_ev_for_update =
          grad_ev + __shfl_sync(0xffffffff, grad_ev_offset_frag, lane_id);
      uint64_t ev_start_indices_v =
          __shfl_sync(0xffffffff, emb_table_ev_start_indices_frag, lane_id);
      float *ev = emb_table + ev_start_indices_v;

      for (int i = threadIdx.x % warpSize; i < ev_size / num_load_floats; i += warpSize) {
        OptimizierInput<wgrad_t> input{grad_ev_for_update, ev_start_indices_v, i * num_load_floats,
                                       lr, scaler};
        optimizer.update4(input, ev);
        // float4 gi = optimizer.update4(input, ev);
      }
    }
  }
}

template <typename key_t, typename index_t, typename emb_t, typename OptimizerFunc,
          typename KeyToIndicesFunc>
__global__ void update_kernel(const key_t *keys, const uint64_t *num_keys_ptr, const int *table_ids,
                              const emb_t *grad_ev, const uint32_t *ev_start_indices,
                              KeyToIndicesFunc key_to_indices_func, float *emb_table,
                              OptimizerFunc optimizer, float lr, float scaler) {
  if (*num_keys_ptr == 0) return;
  uint64_t num_steps = (*num_keys_ptr - 1) / (blockDim.x * gridDim.x) + 1;
  for (size_t step = 0; step < num_steps; step++) {
    uint64_t tid = step * blockDim.x * gridDim.x + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t emb_table_ev_start_indices_frag;
    int ev_size_frag = std::numeric_limits<int>::max();
    uint32_t grad_ev_offset_frag;
    if (tid < *num_keys_ptr) {
      key_t key = keys[tid];
      int table_id = table_ids[tid];
      key_to_indices_func(key, table_id, &emb_table_ev_start_indices_frag, &ev_size_frag);
      grad_ev_offset_frag = ev_start_indices[tid];
    }

    for (int lane_id = 0; lane_id < warpSize; lane_id++) {
      int ev_size = __shfl_sync(0xffffffff, ev_size_frag, lane_id);
      if (ev_size == std::numeric_limits<int>::max()) {
        break;
      }
      const emb_t *grad_ev_for_update =
          grad_ev + __shfl_sync(0xffffffff, grad_ev_offset_frag, lane_id);
      uint64_t ev_start_indices_v =
          __shfl_sync(0xffffffff, emb_table_ev_start_indices_frag, lane_id);
      float *ev = emb_table + ev_start_indices_v;

      for (int i = threadIdx.x % warpSize; i < ev_size; i += warpSize) {
        OptimizierInput<emb_t> input{grad_ev_for_update, ev_start_indices_v, i, lr, scaler};
        optimizer.update(input, ev);
      }
    }
  }
}

}  // namespace

RaggedStaticEmbeddingTable::RaggedStaticEmbeddingTable(
    const HugeCTR::GPUResource &gpu_resource, std::shared_ptr<CoreResourceManager> core,
    const std::vector<EmbeddingTableParam> &table_params, const EmbeddingCollectionParam &ebc_param,
    size_t grouped_id, const HugeCTR::OptParams &opt_param)
    : core_(core), emb_table_size_(0), use_vectorized_kernel_{true}, opt_param_(opt_param) {
  CudaDeviceContext ctx(core_->get_device_id());
  int global_gpu_id = core_->get_global_gpu_id();
  int num_gpus = core_->get_global_gpu_count();
  HCTR_CHECK_HINT(num_gpus == static_cast<int>(ebc_param.shard_matrix.size()),
                  "num_gpus is not match with shard matrix");

  auto key_type = ebc_param.key_type;
  auto index_type = ebc_param.index_type;
  auto emb_type = ebc_param.emb_type;
  const auto &grouped_table_param = ebc_param.grouped_table_params[grouped_id];
  for (const auto &table_param : table_params) {
    use_vectorized_kernel_ &= (table_param.ev_size % num_load_floats == 0);
  }

  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type.type(), key_t, [&] {
    DISPATCH_INTEGRAL_FUNCTION_CORE23(index_type.type(), index_t, [&] {
      std::vector<key_t> h_key_list;
      std::vector<index_t> h_num_key_per_table_offset{0};
      h_emb_table_ev_offset_.push_back(0);

      if (grouped_table_param.table_placement_strategy == TablePlacementStrategy::DataParallel) {
        for (int table_id : grouped_table_param.table_ids) {
          uint64_t num_key = 0;
          h_table_ids_.push_back(table_id);
          h_table_max_vocabulary_size_.push_back(table_params[table_id].max_vocabulary_size);
          for (int64_t k = 0; k < table_params[table_id].max_vocabulary_size; ++k) {
            h_key_list.push_back(k);
            num_key += 1;
          }
          h_num_key_per_table_.push_back(num_key);
          h_num_key_per_table_offset.push_back(num_key);

          uint64_t segment_emb_table_size = num_key * table_params[table_id].ev_size;
          h_size_per_table_.push_back(segment_emb_table_size);
          h_emb_table_ev_offset_.push_back(segment_emb_table_size);
          h_local_ev_sizes_.push_back(table_params[table_id].ev_size);
          emb_table_size_ += segment_emb_table_size;
        }
      } else if (grouped_table_param.table_placement_strategy ==
                 TablePlacementStrategy::ModelParallel) {
        for (int table_id : grouped_table_param.table_ids) {
          std::vector<int> shard_gpu_list;
          for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
            HCTR_CHECK_HINT(table_id < static_cast<int>(ebc_param.shard_matrix[gpu_id].size()),
                            "table_id is out of range");
            if (ebc_param.shard_matrix[gpu_id][table_id] == 1) {
              shard_gpu_list.push_back(gpu_id);
            }
          }
          int num_shards = static_cast<int>(shard_gpu_list.size());
          auto find_shard_id_iter =
              std::find(shard_gpu_list.begin(), shard_gpu_list.end(), global_gpu_id);
          if (find_shard_id_iter == shard_gpu_list.end()) {
            continue;
          }
          uint64_t num_key = 0;
          h_table_ids_.push_back(table_id);
          h_table_max_vocabulary_size_.push_back(table_params[table_id].max_vocabulary_size);
          int shard_id =
              static_cast<int>(std::distance(shard_gpu_list.begin(), find_shard_id_iter));
          for (int64_t k = 0; k < table_params[table_id].max_vocabulary_size; ++k) {
            if (k % num_shards == shard_id) {
              h_key_list.push_back(k);
              num_key += 1;
            }
          }

          h_num_key_per_table_.push_back(num_key);
          h_num_key_per_table_offset.push_back(num_key);
          uint64_t segment_emb_table_size = num_key * table_params[table_id].ev_size;
          h_size_per_table_.push_back(segment_emb_table_size);
          h_emb_table_ev_offset_.push_back(segment_emb_table_size);
          h_local_ev_sizes_.push_back(table_params[table_id].ev_size);
          emb_table_size_ += segment_emb_table_size;
        }
      }
      std::partial_sum(h_num_key_per_table_offset.begin(), h_num_key_per_table_offset.end(),
                       h_num_key_per_table_offset.begin());
      std::partial_sum(h_emb_table_ev_offset_.begin(), h_emb_table_ev_offset_.end(),
                       h_emb_table_ev_offset_.begin());
      for (auto tmp_offset : h_num_key_per_table_offset) {
        h_num_key_per_table_offset_.push_back(static_cast<size_t>(tmp_offset));
      }

      core23::Device device(core23::DeviceType::GPU, core->get_device_id());
      core23::TensorParams params = core23::TensorParams().device(device);

      table_ids_ = core23::Tensor(params.shape({static_cast<int64_t>(h_table_ids_.size())})
                                      .data_type(core23::ScalarType::Int32));
      keys_ = core23::Tensor(
          params.shape({static_cast<int64_t>(h_key_list.size())}).data_type(key_type));
      num_key_per_table_offset_ =
          core23::Tensor(params.shape({static_cast<int64_t>(h_num_key_per_table_offset.size())})
                             .data_type(index_type));
      emb_table_ = core23::Tensor(params.shape({static_cast<int64_t>(emb_table_size_)})
                                      .data_type(core23::ScalarType::Float));
      emb_table_ev_offset_ =
          core23::Tensor(params.shape({static_cast<int64_t>(h_emb_table_ev_offset_.size())})
                             .data_type(core23::ScalarType::UInt64));
      local_ev_size_list_ =
          core23::Tensor(params.shape({static_cast<int64_t>(h_local_ev_sizes_.size())})
                             .data_type(core23::ScalarType::Int32));

      core23::copy_sync(table_ids_, h_table_ids_);
      core23::copy_sync(keys_, h_key_list);
      core23::copy_sync(num_key_per_table_offset_, h_num_key_per_table_offset);
      core23::copy_sync(emb_table_ev_offset_, h_emb_table_ev_offset_);
      core23::copy_sync(local_ev_size_list_, h_local_ev_sizes_);
    });
  });

  if (opt_param.optimizer == HugeCTR::Optimizer_t::AdaGrad) {
    DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(emb_type.type(), emb_t, [&] {
      core23::Device device(core23::DeviceType::GPU, core->get_device_id());
      core23::TensorParams params = core23::TensorParams().device(device);
      auto accum_tensor = core23::Tensor(params.shape({static_cast<int64_t>(emb_table_size_)})
                                             .data_type(core23::ScalarType::Float));

      HCTR_LIB_THROW(cudaMemset(accum_tensor.data(), 0, accum_tensor.num_bytes()));
      opt_buffer_ = AdaGradOptBuffer{accum_tensor};
    });
  }
  if (opt_param.optimizer == HugeCTR::Optimizer_t::Ftrl) {
    DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(emb_type.type(), emb_t, [&] {
      core23::Device device(core23::DeviceType::GPU, core->get_device_id());
      core23::TensorParams params = core23::TensorParams().device(device);
      auto z_tensor = core23::Tensor(params.shape({static_cast<int64_t>(emb_table_size_)})
                                         .data_type(core23::ScalarType::Float));
      auto n_tensor = core23::Tensor(params.shape({static_cast<int64_t>(emb_table_size_)})
                                         .data_type(core23::ScalarType::Float));

      HCTR_LIB_THROW(cudaMemset(z_tensor.data(), 0, z_tensor.num_bytes()));
      HCTR_LIB_THROW(cudaMemset(n_tensor.data(), 0, n_tensor.num_bytes()));
      opt_buffer_ = FtrlOptBuffer{z_tensor, n_tensor};
    });
  }

  for (size_t i = 0; i < h_table_ids_.size(); i++) {
    int table_id = h_table_ids_[i];
    std::function<void(const curandGenerator_t &)> init_table_functor;

    if (table_params[table_id].init_param.initializer_type == HugeCTR::Initializer_t::Default) {
      init_table_functor = [&](const curandGenerator_t &generator) {
        float up_bound = sqrt(1.f / h_table_max_vocabulary_size_[i]);
        size_t offset = h_emb_table_ev_offset_[i];
        size_t num_elements = h_emb_table_ev_offset_[i + 1] - h_emb_table_ev_offset_[i];

        HugeCTR::UniformGenerator::fill(emb_table_.data<float>() + offset, num_elements, -up_bound,
                                        up_bound, gpu_resource.get_sm_count(), generator,
                                        gpu_resource.get_stream());
      };
    } else if (table_params[table_id].init_param.initializer_type ==
               HugeCTR::Initializer_t::Uniform) {
      init_table_functor = [&](const curandGenerator_t &generator) {
        float up_bound = table_params[table_id].init_param.uniform_params.up_bound;
        size_t offset = h_emb_table_ev_offset_[i];
        size_t num_elements = h_emb_table_ev_offset_[i + 1] - h_emb_table_ev_offset_[i];

        HugeCTR::UniformGenerator::fill(emb_table_.data<float>() + offset, num_elements, -up_bound,
                                        up_bound, gpu_resource.get_sm_count(), generator,
                                        gpu_resource.get_stream());
      };
    } else if (table_params[table_id].init_param.initializer_type ==
               HugeCTR::Initializer_t::Sinusoidal) {
      init_table_functor = [&](const curandGenerator_t &) {
        const SinusoidalParams &sinus_params = table_params[table_id].init_param.sinusoidal_params;
        int max_sequence_len = sinus_params.max_sequence_len;
        int ev_size = sinus_params.ev_size;
        size_t offset = h_emb_table_ev_offset_[i];
        size_t num_elements = h_emb_table_ev_offset_[i + 1] - h_emb_table_ev_offset_[i];

        HCTR_CHECK_HINT(max_sequence_len * ev_size == static_cast<int>(num_elements),
                        "max_sequent_len * ev_size ", max_sequence_len * ev_size,
                        " should equal to num_elements ", num_elements);
        HugeCTR::SinusoidalGenerator::fill(emb_table_.data<float>() + offset, num_elements, ev_size,
                                           max_sequence_len, gpu_resource.get_sm_count(),
                                           gpu_resource.get_stream());
      };
    } else {
      HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "initializer not implemented");
    }

    // data parallel table should use same curand seed across all gpus
    if (grouped_table_param.table_placement_strategy == TablePlacementStrategy::DataParallel) {
      init_table_functor(gpu_resource.get_replica_uniform_curand_generator());
    } else {
      init_table_functor(gpu_resource.get_replica_variant_curand_generator());
    }
  }
}

void RaggedStaticEmbeddingTable::lookup(const core23::Tensor &keys, size_t num_keys,
                                        const core23::Tensor &id_space_offset,
                                        size_t num_id_space_offset,
                                        const core23::Tensor &id_space_list,
                                        core23::Tensor &emb_vec) {
  CudaDeviceContext ctx(core_->get_device_id());
  cudaStream_t stream = core_->get_local_gpu()->get_stream();
  if (num_keys == 0) return;
  DISPATCH_INTEGRAL_FUNCTION_CORE23(keys.data_type().type(), key_t, [&] {
    DISPATCH_INTEGRAL_FUNCTION_CORE23(id_space_offset.data_type().type(), offset_t, [&] {
      DISPATCH_INTEGRAL_FUNCTION_CORE23(num_key_per_table_offset_.data_type().type(), index_t, [&] {
        constexpr int block_size = 256;
        int grid_size = (num_keys - 1) / block_size + 1;
        ragged_static_embedding_table_lookup_kernel<<<grid_size, block_size, 0, stream>>>(
            keys.data<key_t>(), num_keys, id_space_offset.data<offset_t>(), num_id_space_offset,
            id_space_list.data<int>(), table_ids_.data<int>(), table_ids_.num_elements(),
            num_key_per_table_offset_.data<index_t>(), emb_table_.data<float>(),
            emb_table_ev_offset_.data<uint64_t>(), local_ev_size_list_.data<int>(),
            static_cast<float **>(emb_vec.data()));

        HCTR_LIB_THROW(cudaPeekAtLastError());
      });
    });
  });
}

void RaggedStaticEmbeddingTable::update(const core23::Tensor &unique_keys,
                                        const core23::Tensor &num_unique_keys,
                                        const core23::Tensor &table_ids,
                                        const core23::Tensor &ev_start_indices,
                                        const core23::Tensor &wgrad) {
  CudaDeviceContext context(core_->get_device_id());
  auto stream = core_->get_local_gpu()->get_stream();
  if (h_table_max_vocabulary_size_.empty()) return;

  HCTR_CHECK_HINT(opt_param_.optimizer != HugeCTR::Optimizer_t::NOT_INITIALIZED,
                  "optimizer not initialized");
  HCTR_CHECK(num_unique_keys.data_type() == core23::ScalarType::UInt64);
  HCTR_CHECK(table_ids.data_type() == core23::ScalarType::Int32);
  HCTR_CHECK(ev_start_indices.data_type() == core23::ScalarType::UInt32);

  if (opt_param_.optimizer == HugeCTR::Optimizer_t::SGD) {
    DISPATCH_INTEGRAL_FUNCTION_CORE23(unique_keys.data_type().type(), key_t, [&] {
      DISPATCH_INTEGRAL_FUNCTION_CORE23(num_key_per_table_offset_.data_type().type(), index_t, [&] {
        DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(wgrad.data_type().type(), wgrad_t, [&] {
          RaggedKeyToIndicesFunc<key_t, index_t> key_to_indices_func{
              table_ids_.data<int>(),
              local_ev_size_list_.data<int>(),
              table_ids_.num_elements(),
              num_key_per_table_offset_.data<index_t>(),
              emb_table_ev_offset_.data<uint64_t>(),
          };
          SGDOptimizer<wgrad_t> optimizer;

          constexpr int block_size = 256;
          const auto &kernel_param = core_->get_kernel_param();
          const int grid_size =
              HugeCTR::ceildiv(kernel_param.num_sms * kernel_param.max_thread_per_sm, block_size);
          auto kernel = use_vectorized_kernel_
                            ? update4_kernel<key_t, index_t, wgrad_t, decltype(optimizer),
                                             decltype(key_to_indices_func)>
                            : update_kernel<key_t, index_t, wgrad_t, decltype(optimizer),
                                            decltype(key_to_indices_func)>;
          kernel<<<grid_size, block_size, 0, stream>>>(
              unique_keys.data<key_t>(), num_unique_keys.data<size_t>(), table_ids.data<int>(),
              wgrad.data<wgrad_t>(), ev_start_indices.data<uint32_t>(), key_to_indices_func,
              emb_table_.data<float>(), optimizer, opt_param_.lr, opt_param_.scaler);
        });
      });
    });
  } else if (opt_param_.optimizer == HugeCTR::Optimizer_t::AdaGrad) {
    DISPATCH_INTEGRAL_FUNCTION_CORE23(unique_keys.data_type().type(), key_t, [&] {
      DISPATCH_INTEGRAL_FUNCTION_CORE23(num_key_per_table_offset_.data_type().type(), index_t, [&] {
        DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(wgrad.data_type().type(), wgrad_t, [&] {
          auto adagrad_opt_buffer = std::get_if<AdaGradOptBuffer>(&opt_buffer_);
          HCTR_CHECK_HINT(adagrad_opt_buffer != nullptr, "Adagrad Opt Buffer not initialized.");
          DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(
              adagrad_opt_buffer->opt_accum_tensor.data_type().type(), acc_t, [&] {
                RaggedKeyToIndicesFunc<key_t, index_t> key_to_indices_func{
                    table_ids_.data<int>(),
                    local_ev_size_list_.data<int>(),
                    table_ids_.num_elements(),
                    num_key_per_table_offset_.data<index_t>(),
                    emb_table_ev_offset_.data<uint64_t>(),
                };
                AdaGradOptimizer<wgrad_t, acc_t> optimizer{
                    adagrad_opt_buffer->opt_accum_tensor.data<acc_t>(),
                    opt_param_.hyperparams.adagrad.epsilon};

                constexpr int block_size = 256;
                const auto &kernel_param = core_->get_kernel_param();
                const int grid_size = HugeCTR::ceildiv(
                    kernel_param.num_sms * kernel_param.max_thread_per_sm, block_size);
                auto kernel = use_vectorized_kernel_
                                  ? update4_kernel<key_t, index_t, wgrad_t, decltype(optimizer),
                                                   decltype(key_to_indices_func)>
                                  : update_kernel<key_t, index_t, wgrad_t, decltype(optimizer),
                                                  decltype(key_to_indices_func)>;
                kernel<<<grid_size, block_size, 0, stream>>>(
                    unique_keys.data<key_t>(), num_unique_keys.data<size_t>(),
                    table_ids.data<int>(), wgrad.data<wgrad_t>(), ev_start_indices.data<uint32_t>(),
                    key_to_indices_func, emb_table_.data<float>(), optimizer, opt_param_.lr,
                    opt_param_.scaler);
              });
        });
      });
    });
  } else if (opt_param_.optimizer == HugeCTR::Optimizer_t::Ftrl) {
    DISPATCH_INTEGRAL_FUNCTION_CORE23(unique_keys.data_type().type(), key_t, [&] {
      DISPATCH_INTEGRAL_FUNCTION_CORE23(num_key_per_table_offset_.data_type().type(), index_t, [&] {
        DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(wgrad.data_type().type(), wgrad_t, [&] {
          auto ftrl_opt_buffer = std::get_if<FtrlOptBuffer>(&opt_buffer_);
          HCTR_CHECK_HINT(ftrl_opt_buffer != nullptr, "Ftrl Opt Buffer not initialized.");
          DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(
              ftrl_opt_buffer->opt_z_tensor.data_type().type(), opt_t, [&] {
                RaggedKeyToIndicesFunc<key_t, index_t> key_to_indices_func{
                    table_ids_.data<int>(),
                    local_ev_size_list_.data<int>(),
                    table_ids_.num_elements(),
                    num_key_per_table_offset_.data<index_t>(),
                    emb_table_ev_offset_.data<uint64_t>(),
                };
                FtrlOptimizer<wgrad_t, opt_t> optimizer{
                    ftrl_opt_buffer->opt_z_tensor.data<opt_t>(),
                    ftrl_opt_buffer->opt_n_tensor.data<opt_t>(), opt_param_.hyperparams.ftrl.beta,
                    opt_param_.hyperparams.ftrl.lambda1, opt_param_.hyperparams.ftrl.lambda2};

                constexpr int block_size = 256;
                const auto &kernel_param = core_->get_kernel_param();
                const int grid_size = HugeCTR::ceildiv(
                    kernel_param.num_sms * kernel_param.max_thread_per_sm, block_size);
                auto kernel = use_vectorized_kernel_
                                  ? update4_kernel<key_t, index_t, wgrad_t, decltype(optimizer),
                                                   decltype(key_to_indices_func)>
                                  : update_kernel<key_t, index_t, wgrad_t, decltype(optimizer),
                                                  decltype(key_to_indices_func)>;
                kernel<<<grid_size, block_size, 0, stream>>>(
                    unique_keys.data<key_t>(), num_unique_keys.data<size_t>(),
                    table_ids.data<int>(), wgrad.data<wgrad_t>(), ev_start_indices.data<uint32_t>(),
                    key_to_indices_func, emb_table_.data<float>(), optimizer, opt_param_.lr,
                    opt_param_.scaler);
              });
        });
      });
    });
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "optimizer not implemented");
  }
}
}  // namespace embedding
