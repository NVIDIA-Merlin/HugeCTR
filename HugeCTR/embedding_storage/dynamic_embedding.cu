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

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include <core/registry.hpp>
#include <cub/cub.cuh>
#include <dynamic_embedding_table/dynamic_embedding_table.hpp>
#include <embedding_storage/dynamic_embedding.hpp>
#include <embedding_storage/optimizers.cuh>
#include <utils.cuh>

namespace embedding {

namespace {

template <typename KeyT, typename ValueT>
constexpr det::DynamicEmbeddingTable<KeyT, ValueT> *cast_table(void *t) noexcept {
  return reinterpret_cast<det::DynamicEmbeddingTable<KeyT, ValueT> *>(t);
}

}  // namespace

DynamicEmbeddingTable::DynamicEmbeddingTable(const HugeCTR::GPUResource &gpu_resource,
                                             std::shared_ptr<CoreResourceManager> core,
                                             const std::vector<EmbeddingTableParam> &table_params,
                                             const EmbeddingCollectionParam &ebc_param,
                                             size_t grouped_id, const HugeCTR::OptParams &opt_param)
    : core_(core),
      key_type_(data_type_core23_to_core[ebc_param.key_type.type()]),
      opt_param_(opt_param) {
  CudaDeviceContext ctx(core_->get_device_id());
  cudaStream_t stream = core_->get_local_gpu()->get_stream();

  const auto &grouped_emb_params = ebc_param.grouped_emb_params[grouped_id];
  const auto &table_ids = grouped_emb_params.table_ids;

  h_table_ids_.assign(table_ids.begin(), table_ids.end());
  DISPATCH_INTEGRAL_FUNCTION(key_type_.type(), key_t, [&] {
    std::vector<size_t> dim_per_class;
    dim_per_class.reserve(table_ids.size());

    // Create DET for model parameters.
    dim_per_class.clear();
    for (auto table_id : table_ids) {
      auto &emb_table_param = table_params[table_id];
      dim_per_class.push_back(emb_table_param.ev_size);
      size_t local_id_space = global_to_local_id_space_map_.size();
      global_to_local_id_space_map_[table_id] = local_id_space;
    }

    dim_per_class_ = dim_per_class;

    table_ =
        new det::DynamicEmbeddingTable<key_t, float>(dim_per_class.size(), dim_per_class.data());
    cast_table<key_t, float>(table_)->initialize(stream);

    // Some optimizers contain state, which will be contained in `table_opt_states_`.
    dim_per_class.clear();
    for (auto table_id : table_ids) {
      auto &emb_table_param = table_params[table_id];
      dim_per_class.push_back(emb_table_param.ev_size * opt_param.num_parameters_per_weight());
    }
    table_opt_states_ = new det::DynamicEmbeddingTable<key_t, float>(dim_per_class.size(),
                                                                     dim_per_class.data(), "zeros");
    cast_table<key_t, float>(table_opt_states_)->initialize(stream);

    // Allocate tensor lists to grab information as we run advanced optimzers.
    size_t max_total_hotness = 0;
    for (const LookupParam &lookup_params : ebc_param.lookup_params) {
      max_total_hotness += lookup_params.max_hotness;
    }
    Device device{DeviceType::GPU, core_->get_device_id()};
    opt_state_view_ = std::make_unique<TensorList>(
        core_.get(), ebc_param.universal_batch_size * max_total_hotness, device,
        TensorScalarType::Float32);

    weight_view_ = std::make_unique<TensorList>(core_.get(),
                                                ebc_param.universal_batch_size * max_total_hotness,
                                                device, TensorScalarType::Float32);
  });

  // Await GPU.
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
}

std::vector<size_t> DynamicEmbeddingTable::remap_id_space(const Tensor &id_space_list,
                                                          cudaStream_t stream) {
  std::vector<size_t> idsl_cpu;
  DISPATCH_INTEGRAL_FUNCTION(id_space_list.dtype().type(), index_type, [&] {
    auto idsl_cpu_tensor = id_space_list.to(core_, DeviceType::CPU, stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    for (int i = 0; i < idsl_cpu_tensor.get_num_elements(); ++i) {
      size_t id_space = static_cast<size_t>(idsl_cpu_tensor.get<index_type>()[i]);
      HCTR_CHECK_HINT(
          global_to_local_id_space_map_.find(id_space) != global_to_local_id_space_map_.end(),
          "DynamicEmbeddingTable remap id space failed.");
      size_t local_id_space = global_to_local_id_space_map_.at(id_space);
      idsl_cpu.push_back(local_id_space);
    }
  });
  return idsl_cpu;
}

std::vector<size_t> DynamicEmbeddingTable::remap_id_space(const std::vector<int> &idsl_cpu) {
  std::vector<size_t> local_idsl_cpu;
  for (size_t i = 0; i < idsl_cpu.size(); ++i) {
    size_t id_space = static_cast<size_t>(idsl_cpu[i]);
    HCTR_CHECK_HINT(
        global_to_local_id_space_map_.find(id_space) != global_to_local_id_space_map_.end(),
        "DynamicEmbeddingTable remap id space failed.");
    size_t local_id_space = global_to_local_id_space_map_.at(id_space);
    local_idsl_cpu.push_back(local_id_space);
  }
  return local_idsl_cpu;
}

void DynamicEmbeddingTable::lookup(const core23::Tensor &core23_keys, size_t num_keys,
                                   const core23::Tensor &core23_id_space_offset,
                                   size_t num_id_space_offset,
                                   const core23::Tensor &core23_id_space_list,
                                   core23::Tensor &core23_emb_vec) {
  CudaDeviceContext ctx(core_->get_device_id());
  cudaStream_t stream = core_->get_local_gpu()->get_stream();
  auto keys = convert_core23_tensor_to_core_tensor(core23_keys);
  auto id_space_offset = convert_core23_tensor_to_core_tensor(core23_id_space_offset);
  auto id_space_list = convert_core23_tensor_to_core_tensor(core23_id_space_list);
  HCTR_CHECK(keys.dtype() == key_type_);

  const auto mapped_id_space_list = remap_id_space(id_space_list, stream);
  std::vector<size_t> id_space_offset_cpu;
  DISPATCH_INTEGRAL_FUNCTION(id_space_offset.dtype().type(), index_t, [&] {
    const auto id_space_offset_cpu_tensor = id_space_offset.to(core_, DeviceType::CPU, stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));

    for (int i = 0; i < id_space_offset_cpu_tensor.get_num_elements(); ++i) {
      size_t offset = static_cast<size_t>(id_space_offset_cpu_tensor.get<index_t>()[i]);
      id_space_offset_cpu.push_back(offset);
    }
  });
  if (num_keys > 0) {
    DISPATCH_INTEGRAL_FUNCTION(keys.dtype().type(), key_t, [&] {
      auto table = cast_table<key_t, float>(table_);

      table->lookup_unsafe(keys.get<key_t>(), (float **)core23_emb_vec.data(), num_keys,
                           mapped_id_space_list.data(), id_space_offset_cpu.data(),
                           num_id_space_offset - 1, stream);
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    });
  }
}

void DynamicEmbeddingTable::update(const Tensor &unique_keys, const Tensor &num_unique_keys,
                                   const Tensor &table_ids, const Tensor &ev_start_indices,
                                   const Tensor &wgrad) {
  CudaDeviceContext context(core_->get_device_id());

  cudaStream_t stream = core_->get_local_gpu()->get_stream();

  HCTR_CHECK(unique_keys.dtype() == key_type_);
  HCTR_CHECK(num_unique_keys.dtype().type() == TensorScalarType::UInt64);
  HCTR_CHECK(table_ids.dtype().type() == TensorScalarType::Int32);
  HCTR_CHECK(wgrad.dtype().type() == TensorScalarType::Float32);
  HCTR_CHECK(num_unique_keys.nbytes() == sizeof(size_t));

  size_t num_unique_keys_cpu;
  HCTR_LIB_THROW(cudaMemcpyAsync(&num_unique_keys_cpu, num_unique_keys.get(),
                                 num_unique_keys.nbytes(), cudaMemcpyDeviceToHost, stream));
  std::vector<int> table_ids_cpu = table_ids.to_vector<int>(stream);
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  if (num_unique_keys_cpu > 0ul) {
    std::vector<int> unique_table_ids_cpu;
    std::vector<size_t> table_range_cpu;

    for (size_t i = 0; i < num_unique_keys_cpu; ++i) {
      if (i == 0 || table_ids_cpu[i] != table_ids_cpu[i - 1]) {
        unique_table_ids_cpu.push_back(table_ids_cpu[i]);
        table_range_cpu.push_back(i);
      }
    }
    table_range_cpu.push_back(num_unique_keys_cpu);

    const auto mapped_unique_table_ids = remap_id_space(unique_table_ids_cpu);
    size_t num_table = mapped_unique_table_ids.size();
    // Request exclusive access to avoid update race.
    const std::lock_guard lock(write_mutex_);

    // FIXME: use another buffer
    float *wgrad_ptr = const_cast<float *>(wgrad.get<float>());
    DISPATCH_INTEGRAL_FUNCTION(unique_keys.dtype().type(), key_t, [&] {
      switch (opt_param_.optimizer) {
        case HugeCTR::Optimizer_t::Ftrl: {
          auto table_opt_states = cast_table<key_t, float>(table_opt_states_);
          table_opt_states->lookup_unsafe(unique_keys.get<key_t>(), opt_state_view_->get<float>(),
                                          num_unique_keys_cpu, mapped_unique_table_ids.data(),
                                          table_range_cpu.data(), num_table, stream);

          auto table = cast_table<key_t, float>(table_);
          table->lookup_unsafe(unique_keys.get<key_t>(), weight_view_->get<float>(),
                               num_unique_keys_cpu, mapped_unique_table_ids.data(),
                               table_range_cpu.data(), num_table, stream);

          const float lambda2_plus_beta_div_lr = opt_param_.hyperparams.ftrl.lambda2 +
                                                 opt_param_.hyperparams.ftrl.beta / opt_param_.lr;

          constexpr int block_size = 256;
          const int grid_size = (static_cast<int64_t>(num_unique_keys_cpu) - 1) / block_size + 1;

          ftrl_update_grad_kernel<<<grid_size, block_size, 0, stream>>>(
              ev_start_indices.get<uint32_t>(), num_unique_keys_cpu, opt_param_.lr,
              opt_param_.hyperparams.ftrl.lambda1, lambda2_plus_beta_div_lr,
              opt_state_view_->get<float>(), weight_view_->get<float>(), opt_param_.scaler,
              wgrad_ptr);
        } break;

        case HugeCTR::Optimizer_t::Adam: {
          auto table_opt_states = cast_table<key_t, float>(table_opt_states_);
          table_opt_states->lookup_unsafe(unique_keys.get<key_t>(), opt_state_view_->get<float>(),
                                          num_unique_keys_cpu, mapped_unique_table_ids.data(),
                                          table_range_cpu.data(), num_table, stream);

          ++opt_param_.hyperparams.adam.times;
          const float lr_scaled_bias = opt_param_.lr * opt_param_.hyperparams.adam.bias();

          constexpr int block_size = 256;
          const int grid_size = (static_cast<int64_t>(num_unique_keys_cpu) - 1) / block_size + 1;

          adam_update_grad_kernel<<<grid_size, block_size, 0, stream>>>(
              ev_start_indices.get<uint32_t>(), num_unique_keys_cpu, lr_scaled_bias,
              opt_param_.hyperparams.adam.beta1, opt_param_.hyperparams.adam.beta2,
              opt_state_view_->get<float>(), opt_param_.hyperparams.adam.epsilon, opt_param_.scaler,
              wgrad_ptr);
        } break;

        case HugeCTR::Optimizer_t::RMSProp: {
          auto table_opt_states = cast_table<key_t, float>(table_opt_states_);
          table_opt_states->lookup_unsafe(unique_keys.get<key_t>(), opt_state_view_->get<float>(),
                                          num_unique_keys_cpu, mapped_unique_table_ids.data(),
                                          table_range_cpu.data(), num_table, stream);

          constexpr int block_size = 256;
          const int grid_size = (static_cast<int64_t>(num_unique_keys_cpu) - 1) / block_size + 1;

          rms_prop_update_grad_kernel<<<grid_size, block_size, 0, stream>>>(
              ev_start_indices.get<uint32_t>(), num_unique_keys_cpu, opt_param_.lr,
              opt_param_.hyperparams.rmsprop.beta, opt_state_view_->get<float>(),
              opt_param_.hyperparams.rmsprop.epsilon, opt_param_.scaler, wgrad_ptr);
        } break;

        case HugeCTR::Optimizer_t::AdaGrad: {
          auto table_opt_states = cast_table<key_t, float>(table_opt_states_);
          table_opt_states->lookup_unsafe(unique_keys.get<key_t>(), opt_state_view_->get<float>(),
                                          num_unique_keys_cpu, mapped_unique_table_ids.data(),
                                          table_range_cpu.data(), num_table, stream);

          constexpr int block_size = 256;
          const int grid_size = (static_cast<int64_t>(num_unique_keys_cpu) - 1) / block_size + 1;

          ada_grad_update_grad_kernel<<<grid_size, block_size, 0, stream>>>(
              ev_start_indices.get<uint32_t>(), num_unique_keys_cpu, opt_param_.lr,
              opt_state_view_->get<float>(), opt_param_.hyperparams.adagrad.epsilon,
              opt_param_.scaler, wgrad_ptr);
        } break;

        case HugeCTR::Optimizer_t::MomentumSGD: {
          auto table_opt_states = cast_table<key_t, float>(table_opt_states_);
          table_opt_states->lookup_unsafe(unique_keys.get<key_t>(), opt_state_view_->get<float>(),
                                          num_unique_keys_cpu, mapped_unique_table_ids.data(),
                                          table_range_cpu.data(), num_table, stream);

          constexpr int block_size = 256;
          const int grid_size = (static_cast<int64_t>(num_unique_keys_cpu) - 1) / block_size + 1;

          momentum_update_grad_kernel<<<grid_size, block_size, 0, stream>>>(
              ev_start_indices.get<uint32_t>(), num_unique_keys_cpu, opt_param_.lr,
              opt_param_.hyperparams.momentum.factor, opt_state_view_->get<float>(),
              opt_param_.scaler, wgrad_ptr);
        } break;

        case HugeCTR::Optimizer_t::Nesterov: {
          auto table_opt_states = cast_table<key_t, float>(table_opt_states_);
          table_opt_states->lookup_unsafe(unique_keys.get<key_t>(), opt_state_view_->get<float>(),
                                          num_unique_keys_cpu, mapped_unique_table_ids.data(),
                                          table_range_cpu.data(), num_table, stream);

          constexpr int block_size = 256;
          const int grid_size = (static_cast<int64_t>(num_unique_keys_cpu) - 1) / block_size + 1;

          nesterov_update_grad_kernel<<<grid_size, block_size, 0, stream>>>(
              ev_start_indices.get<uint32_t>(), num_unique_keys_cpu, opt_param_.lr,
              opt_param_.hyperparams.nesterov.mu, opt_state_view_->get<float>(), opt_param_.scaler,
              wgrad_ptr);
        } break;

        case HugeCTR::Optimizer_t::SGD: {
          constexpr int block_size = 256;
          const int grid_size = (static_cast<int64_t>(num_unique_keys_cpu) - 1) / block_size + 1;

          sgd_update_grad_kernel<<<grid_size, block_size, 0, stream>>>(
              ev_start_indices.get<uint32_t>(), num_unique_keys_cpu, opt_param_.lr,
              opt_param_.scaler, wgrad_ptr);
        } break;

        default:
          HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "optimizer not implemented");
          break;
      }

      // `scatter_add` automatically handles the offsets in `grad_ev_offset` using
      // the embedding vector dimensions given at construction.
      auto table = cast_table<key_t, float>(table_);
      table->scatter_add(unique_keys.get<key_t>(), wgrad.get<float>(), num_unique_keys_cpu,
                         mapped_unique_table_ids.data(), table_range_cpu.data(), num_table, stream);
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    });
  }
}

void DynamicEmbeddingTable::assign(const Tensor &keys, size_t num_keys,
                                   const Tensor &num_unique_key_per_table_offset,
                                   size_t num_table_offset, const Tensor &table_id_list,
                                   Tensor &embeding_vector, const Tensor &embedding_vector_offset) {
  CudaDeviceContext context(core_->get_device_id());
  cudaStream_t stream = core_->get_local_gpu()->get_stream();

  HCTR_CHECK(keys.dtype() == key_type_);
  HCTR_CHECK(embeding_vector.dtype().type() == TensorScalarType::Float32);

  const auto mapped_id_space_list = remap_id_space(table_id_list, stream);
  std::vector<size_t> id_space_offset_cpu;
  DISPATCH_INTEGRAL_FUNCTION(num_unique_key_per_table_offset.dtype().type(), index_t, [&] {
    const auto id_space_offset_cpu_tensor =
        num_unique_key_per_table_offset.to(core_, DeviceType::CPU, stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    for (int i = 0; i < id_space_offset_cpu_tensor.get_num_elements(); ++i) {
      size_t offset = static_cast<size_t>(id_space_offset_cpu_tensor.get<index_t>()[i]);
      id_space_offset_cpu.push_back(offset);
    }
  });

  if (num_keys > 0) {
    DISPATCH_INTEGRAL_FUNCTION(keys.dtype().type(), key_t, [&] {
      // `scatter_add` automatically handles the offsets in `grad_ev_offset` using
      // the embedding vector dimensions given at construction.

      auto table = cast_table<key_t, float>(table_);
      table->scatter_update(keys.get<key_t>(), embeding_vector.get<float>(), num_keys,
                            mapped_id_space_list.data(), id_space_offset_cpu.data(),
                            num_table_offset - 1, stream);
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    });
  }
}

void DynamicEmbeddingTable::load(Tensor &keys, Tensor &id_space_offset, Tensor &embedding_table,
                                 Tensor &ev_size_list, Tensor &id_space) {
  throw std::runtime_error("Not implemented yet!");
}

void DynamicEmbeddingTable::dump(Tensor *keys, Tensor *id_space_offset, Tensor *embedding_table,
                                 Tensor *ev_size_list, Tensor *id_space) {
  throw std::runtime_error("Not implemented yet!");
}

void DynamicEmbeddingTable::dump_by_id(Tensor *h_keys_tensor, Tensor *h_embedding_table,
                                       int table_id) {
  CudaDeviceContext ctx(core_->get_device_id());
  cudaStream_t stream = core_->get_local_gpu()->get_stream();

  auto it = find(h_table_ids_.begin(), h_table_ids_.end(), table_id);
  int table_index = 0;
  if (it != h_table_ids_.end()) {
    table_index = it - h_table_ids_.begin();
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput, "Error: Wrong table id");
  }

  auto key_type = h_keys_tensor->dtype();
  DISPATCH_INTEGRAL_FUNCTION(key_type.type(), key_t, [&] {
    auto table = cast_table<key_t, float>(table_);
    key_t *d_keys;
    float *d_values;
    auto values_sizes = table->size_per_class();
    auto key_nums = this->key_num_per_table();

    auto key_num = key_nums[table_index];
    auto values_size = values_sizes[table_index];

    HCTR_LIB_THROW(cudaMalloc(&d_keys, sizeof(key_t) * key_num));
    HCTR_LIB_THROW(cudaMalloc(&d_values, sizeof(float) * values_size));

    table->eXport(table_index, d_keys, d_values, key_num, stream);

    key_t *h_keys = (key_t *)h_keys_tensor->get();
    float *h_values = (float *)h_embedding_table->get();
    HCTR_LIB_THROW(
        cudaMemcpyAsync(h_keys, d_keys, sizeof(key_t) * key_num, cudaMemcpyDeviceToHost, stream));

    HCTR_LIB_THROW(cudaMemcpyAsync(h_values, d_values, sizeof(float) * values_size,
                                   cudaMemcpyDeviceToHost, stream));
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    HCTR_LIB_THROW(cudaFree(d_keys));
    HCTR_LIB_THROW(cudaFree(d_values));
  });
}

void DynamicEmbeddingTable::load_by_id(Tensor *h_keys_tensor, Tensor *h_embedding_table,
                                       int table_id) {
  CudaDeviceContext ctx(core_->get_device_id());
  cudaStream_t stream = core_->get_local_gpu()->get_stream();
  auto key_type = h_keys_tensor->dtype();
  HCTR_CHECK(h_keys_tensor->dtype() == key_type_);

  auto it = find(h_table_ids_.begin(), h_table_ids_.end(), table_id);
  int table_index = 0;
  if (it != h_table_ids_.end()) {
    table_index = it - h_table_ids_.begin();
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput, "Error: Wrong table id");
  }

  DISPATCH_INTEGRAL_FUNCTION(key_type.type(), key_t, [&] {
    key_t *d_keys;
    float *d_values;
    auto values_size = h_embedding_table->get_num_elements();
    auto key_num = h_keys_tensor->get_num_elements();

    HCTR_LIB_THROW(cudaMalloc(&d_keys, sizeof(key_t) * key_num));
    HCTR_LIB_THROW(cudaMalloc(&d_values, sizeof(float) * values_size));

    key_t *h_keys = (key_t *)h_keys_tensor->get();
    HCTR_LIB_THROW(
        cudaMemcpyAsync(d_keys, h_keys, sizeof(key_t) * key_num, cudaMemcpyHostToDevice, stream));

    auto table = cast_table<key_t, float>(table_);
    table->lookup_by_index(table_index, d_keys, d_values, key_num, stream);

    float *h_values = (float *)h_embedding_table->get();

    HCTR_LIB_THROW(cudaMemcpyAsync(d_values, h_values, sizeof(float) * values_size,
                                   cudaMemcpyHostToDevice, stream));

    table->scatter_update_by_index(table_index, d_keys, d_values, key_num, stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    HCTR_LIB_THROW(cudaFree(d_keys));
    HCTR_LIB_THROW(cudaFree(d_values));
  });
}

size_t DynamicEmbeddingTable::size() const {
  size_t sz = 0;
  DISPATCH_INTEGRAL_FUNCTION(key_type_.type(), key_t, [&] {
    auto table = cast_table<key_t, float>(table_);
    sz = table->size();
  });
  return sz;
}

size_t DynamicEmbeddingTable::capacity() const {
  size_t cap = 0;
  DISPATCH_INTEGRAL_FUNCTION(key_type_.type(), key_t, [&] {
    auto table = cast_table<key_t, float>(table_);
    cap = table->capacity();
  });
  return cap;
}

size_t DynamicEmbeddingTable::key_num() const {
  size_t kn = 0;
  DISPATCH_INTEGRAL_FUNCTION(key_type_.type(), key_t, [&] {
    auto table = cast_table<key_t, float>(table_);
    std::vector<size_t> sizes = table->size_per_class();
    int table_nums = sizes.size();
    for (int i = 0; i < table_nums; ++i) {
      kn += sizes[i] / dim_per_class_[i];
    }
  });
  return kn;
}

std::vector<size_t> DynamicEmbeddingTable::size_per_table() const {
  std::vector<size_t> sizes;
  DISPATCH_INTEGRAL_FUNCTION(key_type_.type(), key_t, [&] {
    auto table = cast_table<key_t, float>(table_);
    sizes = table->size_per_class();
  });
  return sizes;
}

std::vector<size_t> DynamicEmbeddingTable::capacity_per_table() const {
  std::vector<size_t> capacities;
  DISPATCH_INTEGRAL_FUNCTION(key_type_.type(), key_t, [&] {
    auto table = cast_table<key_t, float>(table_);
    capacities = table->capacity_per_class();
  });
  return capacities;
}

std::vector<size_t> DynamicEmbeddingTable::key_num_per_table() const {
  std::vector<size_t> key_nums;
  DISPATCH_INTEGRAL_FUNCTION(key_type_.type(), key_t, [&] {
    auto table = cast_table<key_t, float>(table_);
    std::vector<size_t> sizes = table->size_per_class();
    int table_nums = sizes.size();
    for (int i = 0; i < table_nums; ++i) {
      key_nums.push_back(sizes[i] / dim_per_class_[i]);
    }
  });
  return key_nums;
}

std::vector<int> DynamicEmbeddingTable::table_ids() const { return h_table_ids_; }

std::vector<int> DynamicEmbeddingTable::table_evsize() const {
  std::vector<int> ev_vector;
  for (auto dim : dim_per_class_) {
    ev_vector.push_back(static_cast<int>(dim));
  }
  return ev_vector;
}

void DynamicEmbeddingTable::clear() {
  CudaDeviceContext ctx(core_->get_device_id());
  cudaStream_t stream = core_->get_local_gpu()->get_stream();

  DISPATCH_INTEGRAL_FUNCTION(key_type_.type(), key_t, [&] {
    auto table = cast_table<key_t, float>(table_);
    table->clear(stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  });
}

void DynamicEmbeddingTable::evict(const Tensor &keys, size_t num_keys,
                                  const Tensor &id_space_offset, size_t num_id_space_offset,
                                  const Tensor &id_space_list) {
  CudaDeviceContext ctx(core_->get_device_id());
  cudaStream_t stream = core_->get_local_gpu()->get_stream();

  HCTR_CHECK(keys.dtype() == key_type_);
  HCTR_CHECK(id_space_offset.dtype().type() == TensorScalarType::Size_t);
  HCTR_CHECK(id_space_list.dtype().type() == TensorScalarType::Size_t);

  const auto mapped_id_space_list = remap_id_space(id_space_list, stream);
  std::vector<size_t> id_space_offset_cpu;
  DISPATCH_INTEGRAL_FUNCTION(id_space_offset.dtype().type(), index_t, [&] {
    const auto id_space_offset_cpu_tensor = id_space_offset.to(core_, DeviceType::CPU, stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    for (int i = 0; i < id_space_offset_cpu_tensor.get_num_elements(); ++i) {
      size_t offset = static_cast<size_t>(id_space_offset_cpu_tensor.get<index_t>()[i]);
      id_space_offset_cpu.push_back(offset);
    }
  });

  DISPATCH_INTEGRAL_FUNCTION(key_type_.type(), key_t, [&] {
    auto table = cast_table<key_t, float>(table_);
    table->remove(keys.get<key_t>(), num_keys, mapped_id_space_list.data(),
                  id_space_offset_cpu.data(), num_id_space_offset, stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  });
}

}  // namespace embedding
