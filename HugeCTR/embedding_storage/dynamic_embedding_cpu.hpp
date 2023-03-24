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

#include <core23/logger.hpp>
#include <embedding_storage/optimizers.hpp>
#include <map>
#include <mutex>
#include <random>
#include <unordered_map>
#include <vector>

namespace embedding {

/**
 * This is a CPU mock implementation that mimics the behavior DynamicEmbeddingTable for debugging
 * purposes.
 */
template <class Key>
class DynamicEmbeddingTableCPU final : public IDynamicEmbeddingTable {
 private:
  class IDSpace {
   public:
    IDSpace(const size_t value_size) : value_size{value_size} {}

    void clear() { pairs_.clear(); }

    size_t size() const { return pairs_.size(); }

    void put(const Key& key, const float* values) {
      auto it = pairs_.try_emplace(key).first;
      it->second.assign(values, values + value_size);
    }

    void drop(const Key& key) {
      auto it = pairs_.find(key);
      if (it != pairs_.end()) {
        pairs_.erase(it);
      }
    }

   public:
    const size_t value_size;

   protected:
    std::unordered_map<Key, std::vector<float>> pairs_;
  };

  class WeightIDSpace : public IDSpace {
   public:
    WeightIDSpace(const size_t value_size) : IDSpace(value_size) {}

    std::vector<float>* get(const Key& key) {
      auto it = this->pairs_.try_emplace(key).first;
      if (it->second.empty()) {
        it->second.resize(this->value_size);
        for (size_t i = 0; i < it->second.size(); ++i) {
          it->second[i] = weight_distribution_(weight_generator_);
        }
      }
      return &it->second;
    }

   protected:
    std::random_device weight_rd_;
    std::mt19937 weight_generator_{weight_rd_()};
    std::normal_distribution<float> weight_distribution_{0.f, 1.f};
  };

  class OptStateIDSpace : public IDSpace {
   public:
    OptStateIDSpace(const size_t value_size) : IDSpace(value_size) {}

    std::vector<float>* get(const Key& key) {
      auto it = this->pairs_.try_emplace(key).first;
      if (it->second.empty()) {
        it->second.resize(this->value_size, 0);
      }
      return &it->second;
    }
  };

  mutable std::mutex write_mutex_;

  std::map<size_t, size_t> global_to_local_id_space_;
  std::vector<std::unique_ptr<WeightIDSpace>> weights_;
  std::vector<std::unique_ptr<OptStateIDSpace>> opt_states_;

  HugeCTR::OptParams opt_param_;

 public:
  DynamicEmbeddingTableCPU(const std::vector<EmbeddingTableParam>& table_params,
                           const EmbeddingCollectionParam& ebc_param, size_t group_id,
                           const HugeCTR::OptParams& opt_param)
      : opt_param_{opt_param} {
    const auto& grouped_emb_params = ebc_param.grouped_emb_params[group_id];
    const auto& table_ids = grouped_emb_params.table_ids;

    // Build id_spaces.
    weights_.reserve(table_ids.size());
    opt_states_.reserve(table_ids.size());

    for (auto table_id : table_ids) {
      global_to_local_id_space_[table_id] = weights_.size();

      const size_t ev_size = table_params.at(table_id).ev_size;
      weights_.emplace_back(std::make_unique<WeightIDSpace>(ev_size));
      opt_states_.emplace_back(
          std::make_unique<OptStateIDSpace>(ev_size * opt_param.num_parameters_per_weight()));
    }
  }

  void remap_id_space(std::vector<int32_t>& id_spaces) {
    for (int i = 0; i < id_spaces.size(); ++i) {
      auto it = global_to_local_id_space_.find(id_spaces[i]);
      HCTR_CHECK_HINT(it != global_to_local_id_space_.end(), "ID space remapping failed!");
      id_spaces[i] = static_cast<int32_t>(it->second);
    }
  }

  std::vector<float*> gather_opt_states(const std::vector<Key>& keys,
                                        const std::vector<uint32_t>& id_space_offsets,
                                        const std::vector<int32_t>& id_spaces) {
    std::vector<float*> s;
    s.reserve(keys.size());

    for (size_t i = 0; i < id_spaces.size(); ++i) {
      auto& id_space = opt_states_.at(id_spaces[i]);

      const uint32_t next_off = id_space_offsets[i + 1];
      for (uint32_t off = id_space_offsets[i]; off < next_off; ++off) {
        s.emplace_back(id_space->get(keys[off])->data());
      }
    }

    return s;
  }

  std::vector<float*> gather_weights(const std::vector<Key>& keys,
                                     const std::vector<uint32_t>& id_space_offsets,
                                     const std::vector<int32_t>& id_spaces) {
    std::vector<float*> w;
    w.reserve(keys.size());

    for (size_t i = 0; i < id_spaces.size(); ++i) {
      auto& id_space = weights_.at(id_spaces[i]);

      const uint32_t next_off = id_space_offsets[i + 1];
      for (uint32_t off = id_space_offsets[i]; off < next_off; ++off) {
        w.emplace_back(id_space->get(keys[off])->data());
      }
    }

    return w;
  }

  void lookup(const core23::Tensor& keys, size_t num_keys,
              const core23::Tensor& num_keys_per_table_offset, size_t num_table_offset,
              const core23::Tensor& table_id_list, core23::Tensor& embedding_vec) override {
    // Move to CPU.
    std::vector<Key> k(keys.num_elements());
    core23::copy_sync(k, keys);
    //    auto k = keys.to_vector<Key>();
    HCTR_CHECK(num_keys <= k.size());
    k.resize(num_keys);

    std::vector<uint32_t> is_off(num_keys_per_table_offset.num_elements());
    core23::copy_sync(is_off, num_keys_per_table_offset);
    //    auto is_off = id_space_offsets.to_vector<uint32_t>();
    HCTR_CHECK(num_table_offset <= is_off.size());
    is_off.resize(num_table_offset);

    std::vector<int32_t> is(table_id_list.num_elements());
    core23::copy_sync(is, table_id_list);
    //    auto is = id_spaces.to_vector<int32_t>();
    HCTR_CHECK(is.size() + 1 == is_off.size());
    remap_id_space(is);

    std::vector<float*> w_dev;
    w_dev.resize(k.size());
    HCTR_LIB_THROW(cudaMemcpy(w_dev.data(), embedding_vec.data<float*>(),
                              w_dev.size() * sizeof(float*), cudaMemcpyDeviceToHost));

    // Perform actual lookup.
    for (size_t i = 0; i < is.size(); ++i) {
      auto& id_space = weights_.at(is[i]);

      const uint32_t next_off = is_off[i + 1];
      for (uint32_t off = is_off[i]; off < next_off; ++off) {
        std::vector<float>* w = id_space->get(k[off]);
        HCTR_LIB_THROW(
            cudaMemcpy(w_dev[off], w->data(), w->size() * sizeof(float), cudaMemcpyHostToDevice));
      }
    }
  }
  void assign(const core23::Tensor& unique_key, size_t num_unique_key,
              const core23::Tensor& num_unique_key_per_table_offset, size_t num_table_offset,
              const core23::Tensor& table_id_list, core23::Tensor& embeding_vector,
              const core23::Tensor& embedding_vector_offset) override {
    throw std::runtime_error("Not implemented yet!");
  }

  void compress_table_ids(const std::vector<int>& table_ids, size_t num_table_ids,
                          std::vector<int>* unique_table_ids, std::vector<uint32_t>* table_range) {
    uint32_t cnt = 0;
    for (size_t i = 0; i < num_table_ids; ++i) {
      if (i == 0 || table_ids[i] != table_ids[i - 1]) {
        unique_table_ids->push_back(table_ids[i]);
        table_range->push_back(cnt);
      }
      cnt += 1;
    }
    table_range->push_back(cnt);
  }

  void update(const core23::Tensor& unique_keys, const core23::Tensor& num_unique_keys,
              const core23::Tensor& table_ids, const core23::Tensor& ev_start_indices,
              const core23::Tensor& wgrad) override {
    // Move to CPU.
    std::vector<Key> k(unique_keys.num_elements());
    core23::copy_sync(k, unique_keys);
    std::vector<uint64_t> num_keys_vec(num_unique_keys.num_elements());
    core23::copy_sync(num_keys_vec, num_unique_keys);
    HCTR_CHECK(num_keys_vec.size() == 1);
    auto num_keys = num_keys_vec[0];
    HCTR_CHECK(num_keys <= k.size());
    k.resize(num_keys);
    std::vector<int> table_ids_vec(table_ids.num_elements());
    core23::copy_sync(table_ids_vec, table_ids);
    HCTR_CHECK(num_keys <= table_ids_vec.size());

    std::vector<int> is;
    std::vector<uint32_t> is_off;
    this->compress_table_ids(table_ids_vec, num_keys, &is, &is_off);
    remap_id_space(is);

    std::vector<float> g(wgrad.num_elements());
    core23::copy_sync(g, wgrad);

    std::vector<uint32_t> g_off(ev_start_indices.num_elements());
    core23::copy_sync(g_off, ev_start_indices);

    HCTR_CHECK(k.size() + 1 == g_off.size());

    // Request exclusive access to avoid update race.
    const std::lock_guard lock(write_mutex_);

    // Apply optimizers.
    {
      const float lr = opt_param_.lr;
      const float scaler = opt_param_.scaler;

      switch (opt_param_.optimizer) {
        case HugeCTR::Optimizer_t::Ftrl: {
          std::vector<float*> s = gather_opt_states(k, is_off, is);
          std::vector<float*> w = gather_weights(k, is_off, is);

          const float lambda1 = opt_param_.hyperparams.ftrl.lambda1;
          const float lambda2_plus_beta_div_lr = opt_param_.hyperparams.ftrl.lambda2 +
                                                 opt_param_.hyperparams.ftrl.beta / opt_param_.lr;
          for (uint32_t i = 0; i < k.size(); ++i) {
            ftrl_update_grad(i, g_off.data(), lr, lambda1, lambda2_plus_beta_div_lr, s.data(),
                             w.data(), scaler, g.data());
          }
        } break;

        case HugeCTR::Optimizer_t::Adam: {
          std::vector<float*> s = gather_opt_states(k, is_off, is);

          ++opt_param_.hyperparams.adam.times;
          const float lr_scaled_bias = opt_param_.lr * opt_param_.hyperparams.adam.bias();
          const float beta1 = opt_param_.hyperparams.adam.beta1;
          const float beta2 = opt_param_.hyperparams.adam.beta2;
          const float epsilon = opt_param_.hyperparams.adam.epsilon;
          for (uint32_t i = 0; i < k.size(); ++i) {
            adam_update_grad(i, g_off.data(), lr_scaled_bias, beta1, beta2, s.data(), epsilon,
                             scaler, g.data());
          }
        } break;

        case HugeCTR::Optimizer_t::RMSProp: {
          std::vector<float*> s = gather_opt_states(k, is_off, is);

          const float beta = opt_param_.hyperparams.rmsprop.beta;
          const float epsilon = opt_param_.hyperparams.rmsprop.epsilon;
          for (uint32_t i = 0; i < k.size(); ++i) {
            rms_prop_update_grad(i, g_off.data(), lr, beta, s.data(), epsilon, scaler, g.data());
          }
        } break;

        case HugeCTR::Optimizer_t::AdaGrad: {
          std::vector<float*> s = gather_opt_states(k, is_off, is);

          const float epsilon = opt_param_.hyperparams.adagrad.epsilon;
          for (uint32_t i = 0; i < k.size(); ++i) {
            ada_grad_update_grad(i, g_off.data(), lr, s.data(), epsilon, scaler, g.data());
          }
        } break;

        case HugeCTR::Optimizer_t::MomentumSGD: {
          std::vector<float*> s = gather_opt_states(k, is_off, is);

          const float momentum_decay = opt_param_.hyperparams.momentum.factor;
          for (uint32_t i = 0; i < k.size(); ++i) {
            momentum_update_grad(i, g_off.data(), lr, momentum_decay, s.data(), scaler, g.data());
          }
        } break;

        case HugeCTR::Optimizer_t::Nesterov: {
          std::vector<float*> s = gather_opt_states(k, is_off, is);

          const float momentum_decay = opt_param_.hyperparams.nesterov.mu;
          for (uint32_t i = 0; i < k.size(); ++i) {
            nesterov_update_grad(i, g_off.data(), lr, momentum_decay, s.data(), scaler, g.data());
          }
        } break;

        case HugeCTR::Optimizer_t::SGD: {
          for (uint32_t i = 0; i < k.size(); ++i) {
            sgd_update_grad(i, g_off.data(), lr, scaler, g.data());
          }
        } break;

        default: {
          HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall, "optimizer not implemented");
          break;
        }
      }
    }

    // Scatter-add scaled gradients.
    for (size_t i = 0; i < is.size(); ++i) {
      auto& w_id_space = weights_.at(is[i]);

      const uint32_t next_off = is_off[i + 1];
      for (uint32_t off = is_off[i]; off < next_off; ++off) {
        std::vector<float>& w = *w_id_space->get(k[off]);

        uint32_t w_i = 0;
        uint32_t g_i = g_off[off];
        while (w_i < w.size()) {
          w[w_i++] += g[g_i++];
        }
      }
    }
  }

  void load(core23::Tensor& keys, core23::Tensor& id_space_offsets, core23::Tensor& embeddings,
            core23::Tensor& embedding_sizes, core23::Tensor& id_spaces) override {
    // Move to CPU.
    std::vector<Key> k(keys.num_elements());
    core23::copy_sync(k, keys);

    std::vector<uint32_t> is_off(id_space_offsets.num_elements());
    core23::copy_sync(is_off, id_space_offsets);
    HCTR_CHECK(is_off.back() <= k.size());

    std::vector<int32_t> is(id_spaces.num_elements());
    core23::copy_sync(is, id_spaces);
    HCTR_CHECK(is.size() + 1 == is_off.size());
    remap_id_space(is);

    std::vector<float> v(embeddings.num_elements());
    core23::copy_sync(v, embeddings);

    std::vector<uint32_t> v_sizes(embedding_sizes.num_elements());
    core23::copy_sync(v_sizes, embedding_sizes);
    HCTR_CHECK(v_sizes.size() == k.size());

    // Insert embeddings.
    size_t v_idx = 0;
    for (size_t i = 0; i < is.size(); ++i) {
      auto& id_space = weights_.at(is[i]);

      const uint32_t next_off = is_off[i + 1];
      for (uint32_t off = is_off[i]; off < next_off; ++off) {
        HCTR_CHECK(v_sizes[off] == id_space->value_size);
        id_space->put(k[off], &v[v_idx]);
        v_idx += id_space->value_size;
      }
    }
  }

  void dump_by_id(core23::Tensor* h_keys_tensor, core23::Tensor* h_embedding_table,
                  int table_id) override {
    throw std::runtime_error("Not implemented yet!");
  }

  void load_by_id(core23::Tensor* h_keys_tensor, core23::Tensor* h_embedding_table,
                  int table_id) override {
    throw std::runtime_error("Not implemented yet!");
  }

  void dump(core23::Tensor* keys, core23::Tensor* id_space_offset, core23::Tensor* embedding_table,
            core23::Tensor* ev_size_list, core23::Tensor* id_space) override {
    throw std::runtime_error("Not implemented yet!");
  }

  size_t size() const override {
    size_t n = 0;
    for (const auto& w : weights_) {
      n += w->size();
    }
    return n;
  }

  size_t capacity() const override { return std::numeric_limits<size_t>::max(); }

  size_t key_num() const override { throw std::runtime_error("Not implemented yet!"); }

  std::vector<size_t> size_per_table() const override {
    throw std::runtime_error("Not implemented yet!");
  }

  std::vector<size_t> capacity_per_table() const override {
    throw std::runtime_error("Not implemented yet!");
  }

  std::vector<size_t> key_num_per_table() const override {
    throw std::runtime_error("Not implemented yet!");
  }

  std::vector<int> table_ids() const override { throw std::runtime_error("Not implemented yet!"); }

  std::vector<int> table_evsize() const override {
    throw std::runtime_error("Not implemented yet!");
  }

  void clear() override {
    for (auto& w : weights_) {
      w->clear();
    }
    for (auto& s : opt_states_) {
      s->clear();
    }
  }

  void evict(const core23::Tensor& keys, size_t num_keys, const core23::Tensor& id_space_offsets,
             size_t num_id_space_offsets, const core23::Tensor& id_spaces) override {
    // Move to CPU.
    std::vector<Key> k(keys.num_elements());
    core23::copy_sync(k, keys);
    HCTR_CHECK(num_keys <= k.size());
    k.resize(num_keys);

    std::vector<uint32_t> is_off(id_space_offsets.num_elements());
    core23::copy_sync(is_off, id_space_offsets);
    HCTR_CHECK(num_id_space_offsets <= is_off.size());
    is_off.resize(num_id_space_offsets);

    std::vector<int32_t> is(id_spaces.num_elements());
    core23::copy_sync(is, id_spaces);
    HCTR_CHECK(is.size() + 1 == is_off.size());
    remap_id_space(is);

    // Perform actual eviction.
    for (size_t i = 0; i < is.size(); ++i) {
      auto& w_id_space = weights_.at(is[i]);
      auto& s_id_space = opt_states_.at(is[i]);

      const uint32_t next_off = is_off[i + 1];
      for (uint32_t off = is_off[i]; off < next_off; ++off) {
        w_id_space->drop(k[off]);
        s_id_space->drop(k[off]);
      }
    }
  }

  void set_learning_rate(float lr) override { opt_param_.lr = lr; }
};

}  // namespace embedding
