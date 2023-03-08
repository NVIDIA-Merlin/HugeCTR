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
    : core_(core), key_type_(ebc_param.key_type), opt_param_(opt_param) {
  CudaDeviceContext ctx(core_->get_device_id());
  cudaStream_t stream = core_->get_local_gpu()->get_stream();

  const auto &grouped_emb_params = ebc_param.grouped_emb_params[grouped_id];
  const auto &table_ids = grouped_emb_params.table_ids;

  h_table_ids_.assign(table_ids.begin(), table_ids.end());
  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type_.type(), key_t, [&] {
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

    opt_state_view_ = std::make_unique<core23::Tensor>(core23::init_tensor_list<float>(
        ebc_param.universal_batch_size * max_total_hotness, core_->get_device_id()));

    weight_view_ = std::make_unique<core23::Tensor>(core23::init_tensor_list<float>(
        ebc_param.universal_batch_size * max_total_hotness, core_->get_device_id()));
  });

  // Await GPU.
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
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

std::vector<size_t> DynamicEmbeddingTable::remap_id_space(const core23::Tensor &id_space_list,
                                                          cudaStream_t stream) {
  std::vector<size_t> idsl_cpu;
  DISPATCH_INTEGRAL_FUNCTION_CORE23(id_space_list.data_type().type(), index_type, [&] {
    std::vector<index_type> idsl_cpu_tensor(id_space_list.num_elements());
    core23::copy_async(idsl_cpu_tensor, id_space_list, stream);

    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < idsl_cpu_tensor.size(); ++i) {
      size_t id_space = static_cast<size_t>(idsl_cpu_tensor[i]);
      HCTR_CHECK_HINT(
          global_to_local_id_space_map_.find(id_space) != global_to_local_id_space_map_.end(),
          "DynamicEmbeddingTable remap id space failed.");
      size_t local_id_space = global_to_local_id_space_map_.at(id_space);
      idsl_cpu.push_back(local_id_space);
    }
  });
  return idsl_cpu;
}

void DynamicEmbeddingTable::lookup(const core23::Tensor &keys, size_t num_keys,
                                   const core23::Tensor &id_space_offset,
                                   size_t num_id_space_offset, const core23::Tensor &id_space_list,
                                   core23::Tensor &emb_vec) {
  CudaDeviceContext ctx(core_->get_device_id());
  cudaStream_t stream = core_->get_local_gpu()->get_stream();
  HCTR_CHECK(keys.data_type() == key_type_);

  const auto mapped_id_space_list = remap_id_space(id_space_list, stream);
  std::vector<size_t> id_space_offset_cpu;
  DISPATCH_INTEGRAL_FUNCTION_CORE23(id_space_offset.data_type().type(), index_t, [&] {
    std::vector<index_t> id_space_offset_cpu_tensor(id_space_offset.num_elements());
    core23::copy_async(id_space_offset_cpu_tensor, id_space_offset, stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));

    for (size_t i = 0; i < id_space_offset_cpu_tensor.size(); ++i) {
      size_t offset = static_cast<size_t>(id_space_offset_cpu_tensor[i]);
      id_space_offset_cpu.push_back(offset);
    }
  });
  if (num_keys > 0) {
    DISPATCH_INTEGRAL_FUNCTION_CORE23(keys.data_type().type(), key_t, [&] {
      auto table = cast_table<key_t, float>(table_);

      table->lookup_unsafe(keys.data<key_t>(), (float **)emb_vec.data(), num_keys,
                           mapped_id_space_list.data(), id_space_offset_cpu.data(),
                           num_id_space_offset - 1, stream);
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    });
  }
}

void DynamicEmbeddingTable::update(const core23::Tensor &unique_keys,
                                   const core23::Tensor &num_unique_keys,
                                   const core23::Tensor &table_ids,
                                   const core23::Tensor &ev_start_indices,
                                   const core23::Tensor &wgrad) {
  CudaDeviceContext context(core_->get_device_id());

  cudaStream_t stream = core_->get_local_gpu()->get_stream();

  HCTR_CHECK(unique_keys.data_type() == key_type_);
  HCTR_CHECK(num_unique_keys.data_type().type() == core23::ScalarType::UInt64);
  HCTR_CHECK(table_ids.data_type().type() == core23::ScalarType::Int32);
  HCTR_CHECK(wgrad.data_type().type() == core23::ScalarType::Float);
  HCTR_CHECK(num_unique_keys.num_bytes() == sizeof(size_t));

  size_t num_unique_keys_cpu;
  HCTR_LIB_THROW(cudaMemcpyAsync(&num_unique_keys_cpu, num_unique_keys.data(),
                                 num_unique_keys.num_bytes(), cudaMemcpyDeviceToHost, stream));
  std::vector<int> table_ids_cpu(table_ids.num_elements());
  core23::copy_async(table_ids_cpu, table_ids, stream);
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
    float *wgrad_ptr = const_cast<float *>(wgrad.data<float>());
    DISPATCH_INTEGRAL_FUNCTION_CORE23(unique_keys.data_type().type(), key_t, [&] {
      switch (opt_param_.optimizer) {
        case HugeCTR::Optimizer_t::Ftrl: {
          auto table_opt_states = cast_table<key_t, float>(table_opt_states_);
          table_opt_states->lookup_unsafe(
              unique_keys.data<key_t>(), (float **)opt_state_view_->data(), num_unique_keys_cpu,
              mapped_unique_table_ids.data(), table_range_cpu.data(), num_table, stream);

          auto table = cast_table<key_t, float>(table_);
          table->lookup_unsafe(unique_keys.data<key_t>(), (float **)weight_view_->data(),
                               num_unique_keys_cpu, mapped_unique_table_ids.data(),
                               table_range_cpu.data(), num_table, stream);

          const float lambda2_plus_beta_div_lr = opt_param_.hyperparams.ftrl.lambda2 +
                                                 opt_param_.hyperparams.ftrl.beta / opt_param_.lr;

          constexpr int block_size = 256;
          const int grid_size = (static_cast<int64_t>(num_unique_keys_cpu) - 1) / block_size + 1;

          ftrl_update_grad_kernel<<<grid_size, block_size, 0, stream>>>(
              ev_start_indices.data<uint32_t>(), num_unique_keys_cpu, opt_param_.lr,
              opt_param_.hyperparams.ftrl.lambda1, lambda2_plus_beta_div_lr,
              (float **)opt_state_view_->data(), (float **)weight_view_->data(), opt_param_.scaler,
              wgrad_ptr);
        } break;

        case HugeCTR::Optimizer_t::Adam: {
          auto table_opt_states = cast_table<key_t, float>(table_opt_states_);
          table_opt_states->lookup_unsafe(
              unique_keys.data<key_t>(), (float **)opt_state_view_->data(), num_unique_keys_cpu,
              mapped_unique_table_ids.data(), table_range_cpu.data(), num_table, stream);

          ++opt_param_.hyperparams.adam.times;
          const float lr_scaled_bias = opt_param_.lr * opt_param_.hyperparams.adam.bias();

          constexpr int block_size = 256;
          const int grid_size = (static_cast<int64_t>(num_unique_keys_cpu) - 1) / block_size + 1;

          adam_update_grad_kernel<<<grid_size, block_size, 0, stream>>>(
              ev_start_indices.data<uint32_t>(), num_unique_keys_cpu, lr_scaled_bias,
              opt_param_.hyperparams.adam.beta1, opt_param_.hyperparams.adam.beta2,
              (float **)opt_state_view_->data(), opt_param_.hyperparams.adam.epsilon,
              opt_param_.scaler, wgrad_ptr);
        } break;

        case HugeCTR::Optimizer_t::RMSProp: {
          auto table_opt_states = cast_table<key_t, float>(table_opt_states_);
          table_opt_states->lookup_unsafe(
              unique_keys.data<key_t>(), (float **)opt_state_view_->data(), num_unique_keys_cpu,
              mapped_unique_table_ids.data(), table_range_cpu.data(), num_table, stream);

          constexpr int block_size = 256;
          const int grid_size = (static_cast<int64_t>(num_unique_keys_cpu) - 1) / block_size + 1;

          rms_prop_update_grad_kernel<<<grid_size, block_size, 0, stream>>>(
              ev_start_indices.data<uint32_t>(), num_unique_keys_cpu, opt_param_.lr,
              opt_param_.hyperparams.rmsprop.beta, (float **)opt_state_view_->data(),
              opt_param_.hyperparams.rmsprop.epsilon, opt_param_.scaler, wgrad_ptr);
        } break;

        case HugeCTR::Optimizer_t::AdaGrad: {
          auto table_opt_states = cast_table<key_t, float>(table_opt_states_);
          table_opt_states->lookup_unsafe(
              unique_keys.data<key_t>(), (float **)opt_state_view_->data(), num_unique_keys_cpu,
              mapped_unique_table_ids.data(), table_range_cpu.data(), num_table, stream);

          constexpr int block_size = 256;
          const int grid_size = (static_cast<int64_t>(num_unique_keys_cpu) - 1) / block_size + 1;

          ada_grad_update_grad_kernel<<<grid_size, block_size, 0, stream>>>(
              ev_start_indices.data<uint32_t>(), num_unique_keys_cpu, opt_param_.lr,
              (float **)opt_state_view_->data(), opt_param_.hyperparams.adagrad.epsilon,
              opt_param_.scaler, wgrad_ptr);
        } break;

        case HugeCTR::Optimizer_t::MomentumSGD: {
          auto table_opt_states = cast_table<key_t, float>(table_opt_states_);
          table_opt_states->lookup_unsafe(
              unique_keys.data<key_t>(), (float **)opt_state_view_->data(), num_unique_keys_cpu,
              mapped_unique_table_ids.data(), table_range_cpu.data(), num_table, stream);

          constexpr int block_size = 256;
          const int grid_size = (static_cast<int64_t>(num_unique_keys_cpu) - 1) / block_size + 1;

          momentum_update_grad_kernel<<<grid_size, block_size, 0, stream>>>(
              ev_start_indices.data<uint32_t>(), num_unique_keys_cpu, opt_param_.lr,
              opt_param_.hyperparams.momentum.factor, (float **)opt_state_view_->data(),
              opt_param_.scaler, wgrad_ptr);
        } break;

        case HugeCTR::Optimizer_t::Nesterov: {
          auto table_opt_states = cast_table<key_t, float>(table_opt_states_);
          table_opt_states->lookup_unsafe(
              unique_keys.data<key_t>(), (float **)opt_state_view_->data(), num_unique_keys_cpu,
              mapped_unique_table_ids.data(), table_range_cpu.data(), num_table, stream);

          constexpr int block_size = 256;
          const int grid_size = (static_cast<int64_t>(num_unique_keys_cpu) - 1) / block_size + 1;

          nesterov_update_grad_kernel<<<grid_size, block_size, 0, stream>>>(
              ev_start_indices.data<uint32_t>(), num_unique_keys_cpu, opt_param_.lr,
              opt_param_.hyperparams.nesterov.mu, (float **)opt_state_view_->data(),
              opt_param_.scaler, wgrad_ptr);
        } break;

        case HugeCTR::Optimizer_t::SGD: {
          constexpr int block_size = 256;
          const int grid_size = (static_cast<int64_t>(num_unique_keys_cpu) - 1) / block_size + 1;

          sgd_update_grad_kernel<<<grid_size, block_size, 0, stream>>>(
              ev_start_indices.data<uint32_t>(), num_unique_keys_cpu, opt_param_.lr,
              opt_param_.scaler, wgrad_ptr);
        } break;

        default:
          HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "optimizer not implemented");
          break;
      }

      // `scatter_add` automatically handles the offsets in `grad_ev_offset` using
      // the embedding vector dimensions given at construction.
      auto table = cast_table<key_t, float>(table_);
      table->scatter_add(unique_keys.data<key_t>(), wgrad.data<float>(), num_unique_keys_cpu,
                         mapped_unique_table_ids.data(), table_range_cpu.data(), num_table, stream);
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    });
  }
}

void DynamicEmbeddingTable::assign(const core23::Tensor &keys, size_t num_keys,
                                   const core23::Tensor &num_unique_key_per_table_offset,
                                   size_t num_table_offset, const core23::Tensor &table_id_list,
                                   core23::Tensor &embeding_vector,
                                   const core23::Tensor &embedding_vector_offset) {
  CudaDeviceContext context(core_->get_device_id());
  cudaStream_t stream = core_->get_local_gpu()->get_stream();

  HCTR_CHECK(keys.data_type() == key_type_);
  HCTR_CHECK(embeding_vector.data_type().type() == core23::ScalarType::Float);

  const auto mapped_id_space_list = remap_id_space(table_id_list, stream);
  std::vector<size_t> id_space_offset_cpu;
  DISPATCH_INTEGRAL_FUNCTION_CORE23(
      num_unique_key_per_table_offset.data_type().type(), index_t, [&] {
        std::vector<int> id_space_offset_cpu_tensor(num_unique_key_per_table_offset.num_elements());
        core23::copy_async(id_space_offset_cpu_tensor, num_unique_key_per_table_offset, stream);
        HCTR_LIB_THROW(cudaStreamSynchronize(stream));

        for (size_t i = 0; i < id_space_offset_cpu_tensor.size(); ++i) {
          size_t offset = static_cast<size_t>(id_space_offset_cpu_tensor[i]);
          id_space_offset_cpu.push_back(offset);
        }
      });

  if (num_keys > 0) {
    DISPATCH_INTEGRAL_FUNCTION_CORE23(keys.data_type().type(), key_t, [&] {
      // `scatter_add` automatically handles the offsets in `grad_ev_offset` using
      // the embedding vector dimensions given at construction.

      auto table = cast_table<key_t, float>(table_);
      table->scatter_update(keys.data<key_t>(), embeding_vector.data<float>(), num_keys,
                            mapped_id_space_list.data(), id_space_offset_cpu.data(),
                            num_table_offset - 1, stream);
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    });
  }
}

void DynamicEmbeddingTable::load(core23::Tensor &keys, core23::Tensor &id_space_offset,
                                 core23::Tensor &embedding_table, core23::Tensor &ev_size_list,
                                 core23::Tensor &id_space) {
  throw std::runtime_error("Not implemented yet!");
}

void DynamicEmbeddingTable::dump(core23::Tensor *keys, core23::Tensor *id_space_offset,
                                 core23::Tensor *embedding_table, core23::Tensor *ev_size_list,
                                 core23::Tensor *id_space) {
  throw std::runtime_error("Not implemented yet!");
}

void DynamicEmbeddingTable::dump_by_id(core23::Tensor *h_keys_tensor,
                                       core23::Tensor *h_embedding_table, int table_id) {
  CudaDeviceContext ctx(core_->get_device_id());
  cudaStream_t stream = core_->get_local_gpu()->get_stream();

  auto it = find(h_table_ids_.begin(), h_table_ids_.end(), table_id);
  int table_index = 0;
  if (it != h_table_ids_.end()) {
    table_index = it - h_table_ids_.begin();
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput, "Error: Wrong table id");
  }

  auto key_type = h_keys_tensor->data_type();
  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type.type(), key_t, [&] {
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

    key_t *h_keys = (key_t *)h_keys_tensor->data();
    float *h_values = (float *)h_embedding_table->data();
    HCTR_LIB_THROW(
        cudaMemcpyAsync(h_keys, d_keys, sizeof(key_t) * key_num, cudaMemcpyDeviceToHost, stream));

    HCTR_LIB_THROW(cudaMemcpyAsync(h_values, d_values, sizeof(float) * values_size,
                                   cudaMemcpyDeviceToHost, stream));
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    HCTR_LIB_THROW(cudaFree(d_keys));
    HCTR_LIB_THROW(cudaFree(d_values));
  });
}

void DynamicEmbeddingTable::load_by_id(core23::Tensor *h_keys_tensor,
                                       core23::Tensor *h_embedding_table, int table_id) {
  CudaDeviceContext ctx(core_->get_device_id());
  cudaStream_t stream = core_->get_local_gpu()->get_stream();
  auto key_type = h_keys_tensor->data_type();
  HCTR_CHECK(h_keys_tensor->data_type() == key_type_);

  auto it = find(h_table_ids_.begin(), h_table_ids_.end(), table_id);
  int table_index = 0;
  if (it != h_table_ids_.end()) {
    table_index = it - h_table_ids_.begin();
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput, "Error: Wrong table id");
  }

  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type.type(), key_t, [&] {
    key_t *d_keys;
    float *d_values;
    auto values_size = h_embedding_table->num_elements();
    auto key_num = h_keys_tensor->num_elements();

    HCTR_LIB_THROW(cudaMalloc(&d_keys, sizeof(key_t) * key_num));
    HCTR_LIB_THROW(cudaMalloc(&d_values, sizeof(float) * values_size));

    key_t *h_keys = (key_t *)h_keys_tensor->data();
    HCTR_LIB_THROW(
        cudaMemcpyAsync(d_keys, h_keys, sizeof(key_t) * key_num, cudaMemcpyHostToDevice, stream));

    auto table = cast_table<key_t, float>(table_);
    table->lookup_by_index(table_index, d_keys, d_values, key_num, stream);

    float *h_values = (float *)h_embedding_table->data();

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
  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type_.type(), key_t, [&] {
    auto table = cast_table<key_t, float>(table_);
    sz = table->size();
  });
  return sz;
}

size_t DynamicEmbeddingTable::capacity() const {
  size_t cap = 0;
  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type_.type(), key_t, [&] {
    auto table = cast_table<key_t, float>(table_);
    cap = table->capacity();
  });
  return cap;
}

size_t DynamicEmbeddingTable::key_num() const {
  size_t kn = 0;
  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type_.type(), key_t, [&] {
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
  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type_.type(), key_t, [&] {
    auto table = cast_table<key_t, float>(table_);
    sizes = table->size_per_class();
  });
  return sizes;
}

std::vector<size_t> DynamicEmbeddingTable::capacity_per_table() const {
  std::vector<size_t> capacities;
  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type_.type(), key_t, [&] {
    auto table = cast_table<key_t, float>(table_);
    capacities = table->capacity_per_class();
  });
  return capacities;
}

std::vector<size_t> DynamicEmbeddingTable::key_num_per_table() const {
  std::vector<size_t> key_nums;
  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type_.type(), key_t, [&] {
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

  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type_.type(), key_t, [&] {
    auto table = cast_table<key_t, float>(table_);
    table->clear(stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  });
}

void DynamicEmbeddingTable::evict(const core23::Tensor &keys, size_t num_keys,
                                  const core23::Tensor &id_space_offset, size_t num_id_space_offset,
                                  const core23::Tensor &id_space_list) {
  CudaDeviceContext ctx(core_->get_device_id());
  cudaStream_t stream = core_->get_local_gpu()->get_stream();

  HCTR_CHECK(keys.data_type() == key_type_);
  HCTR_CHECK(id_space_offset.data_type().type() == core23::ScalarType::UInt64);
  HCTR_CHECK(id_space_list.data_type().type() == core23::ScalarType::UInt64);

  const auto mapped_id_space_list = remap_id_space(id_space_list, stream);
  std::vector<size_t> id_space_offset_cpu;
  DISPATCH_INTEGRAL_FUNCTION_CORE23(id_space_offset.data_type().type(), index_t, [&] {
    std::vector<uint64_t> id_space_offset_cpu_tensor(id_space_offset.num_elements());
    core23::copy_async(id_space_offset_cpu_tensor, id_space_offset, stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < id_space_offset_cpu_tensor.size(); ++i) {
      size_t offset = static_cast<size_t>(id_space_offset_cpu_tensor[i]);
      id_space_offset_cpu.push_back(offset);
    }
  });

  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type_.type(), key_t, [&] {
    auto table = cast_table<key_t, float>(table_);
    table->remove(keys.data<key_t>(), num_keys, mapped_id_space_list.data(),
                  id_space_offset_cpu.data(), num_id_space_offset, stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  });
}

}  // namespace embedding
