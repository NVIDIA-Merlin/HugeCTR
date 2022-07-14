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

#include "hybrid_embedding_cpu.hpp"

#include <algorithm>
#include <random>
#include <utility>

using namespace HugeCTR;
using namespace HugeCTR::hybrid_embedding;

namespace utils {
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

template <typename T>
static bool lesser_by_first(const std::pair<T, T>& a, const std::pair<T, T>& b) {
  return (a.first < b.first);
}

}  // namespace utils

template <typename dtype, typename emtype>
void HybridEmbeddingCpu<dtype, emtype>::calculate_infrequent_model_indices() {
  model_indices.resize(num_instances);
  model_indices_offsets.resize(num_instances);

  for (uint32_t model_id = 0; model_id < num_instances; model_id++) {
    model_indices[model_id].resize(batch_size * num_tables);
    model_indices_offsets[model_id].resize(num_instances + 1);

    // Prefix sum
    uint32_t sum = 0;
    for (uint32_t j = 0; j < batch_size; j++) {
      if (j % local_batch_size == 0) {
        model_indices_offsets[model_id][j / local_batch_size] = sum;
      }
      for (uint32_t i = 0; i < num_tables; i++) {
        uint32_t idx = j * num_tables + i;

        dtype category = samples[idx];
        bool mask = category_location[2 * category] == model_id;

        sum += static_cast<uint32_t>(mask);

        if (mask) model_indices[model_id][sum - 1] = idx;
      }
    }
    // Total size stored at the end of the offsets vector
    model_indices_offsets[model_id][num_instances] = sum;
    model_indices[model_id].resize(sum);
  }
}

template <typename dtype, typename emtype>
void HybridEmbeddingCpu<dtype, emtype>::calculate_infrequent_network_indices() {
  network_indices.resize(num_instances);
  network_indices_offsets.resize(num_instances);

  for (uint32_t network_id = 0; network_id < num_instances; network_id++) {
    network_indices[network_id].resize(local_batch_size * num_tables);
    network_indices_offsets[network_id].resize(num_instances + 1);

    std::vector<std::pair<uint32_t, uint32_t>> network_sources_indices =
        std::vector<std::pair<uint32_t, uint32_t>>(local_batch_size * num_tables);

    // Prefix sum only of this GPU's sub-batch
    uint32_t sum = 0;
    for (uint32_t j = local_batch_size * network_id;
         j < std::min(batch_size, local_batch_size * (network_id + 1)); j++) {
      for (uint32_t i = 0; i < num_tables; i++) {
        uint32_t idx = j * num_tables + i;
        dtype category = samples[idx];
        dtype model_id = category_location[2 * category];
        bool mask = model_id < num_instances;
        sum += static_cast<uint32_t>(mask);
        uint32_t local_mlp_index = (j - local_batch_size * network_id) * num_tables + i;
        if (mask)
          network_sources_indices[sum - 1] =
              std::make_pair(static_cast<uint32_t>(model_id), local_mlp_index);
      }
    }
    // Sort by source only, otherwise stable
    std::stable_sort(network_sources_indices.begin(), network_sources_indices.begin() + sum,
                     utils::lesser_by_first<dtype>);

    // Retrieve indices
    for (uint32_t idx = 0; idx < sum; idx++) {
      network_indices[network_id][idx] = network_sources_indices[idx].second;
    }
    // Compute offsets
    for (uint32_t i = 0; i < num_instances; i++) {
      network_indices_offsets[network_id][i] =
          std::lower_bound(network_sources_indices.begin(), network_sources_indices.begin() + sum,
                           std::make_pair(i, (uint32_t)0), utils::lesser_by_first<uint32_t>) -
          network_sources_indices.begin();
    }
    // Total size stored at the end of the offsets vector
    network_indices_offsets[network_id][num_instances] = sum;
    network_indices[network_id].resize(sum);
  }
}

template <typename dtype, typename emtype>
void HybridEmbeddingCpu<dtype, emtype>::calculate_frequent_sample_indices() {
  frequent_sample_indices.resize(num_instances);

  for (uint32_t network_id = 0; network_id < num_instances; network_id++) {
    frequent_sample_indices[network_id].resize(local_batch_size * num_tables);

    // Prefix sum only of this GPU's sub-batch
    uint32_t sum = 0;
    for (uint32_t j = local_batch_size * network_id;
         j < std::min(batch_size, local_batch_size * (network_id + 1)); j++) {
      for (uint32_t i = 0; i < num_tables; i++) {
        uint32_t idx = j * num_tables + i;

        dtype category = samples[idx];
        dtype model_id = category_location[2 * category];
        bool mask = model_id == num_instances;

        sum += static_cast<uint32_t>(mask);

        uint32_t local_mlp_index = (j - local_batch_size * network_id) * num_tables + i;

        if (mask) frequent_sample_indices[network_id][sum - 1] = local_mlp_index;
      }
    }

    frequent_sample_indices[network_id].resize(sum);
  }
}

template <typename dtype, typename emtype>
void HybridEmbeddingCpu<dtype, emtype>::calculate_frequent_model_cache_indices() {
  const uint32_t num_frequent_per_model = num_frequent / num_instances;

  model_cache_indices.resize(num_instances);
  model_cache_indices_offsets.resize(num_instances);

  for (uint32_t model_id = 0; model_id < num_instances; model_id++) {
    model_cache_indices[model_id].resize(num_frequent);
    model_cache_indices_offsets[model_id].resize(num_instances + 1);

    /* Compute the mask (for each network, frequent categories that belong to my model id) */
    std::vector<bool> network_frequent_mask = std::vector<bool>(num_frequent, false);
    for (uint32_t i = 0; i < num_instances; i++) {
      for (uint32_t j = 0; j < local_batch_size * num_tables; j++) {
        uint32_t global_j = local_batch_size * num_tables * i + j;

        dtype category = samples[global_j];
        dtype frequent_index = category_location[2 * category + 1];

        if (category_location[2 * category] == num_instances &&
            frequent_index / num_frequent_per_model == model_id) {
          network_frequent_mask[i * num_frequent_per_model +
                                frequent_index % num_frequent_per_model] = true;
        }
      }
    }

    /* Select categories according to the mask */
    uint32_t sum = 0;
    for (uint32_t idx = 0; idx < num_frequent; idx++) {
      bool mask = network_frequent_mask[idx];
      sum += static_cast<uint32_t>(mask);
      if (mask) model_cache_indices[model_id][sum - 1] = idx;
    }

    /* Compute offsets */
    for (uint32_t i = 0; i < num_instances; i++) {
      model_cache_indices_offsets[model_id][i] =
          std::lower_bound(model_cache_indices[model_id].begin(),
                           model_cache_indices[model_id].begin() + sum,
                           i * num_frequent_per_model) -
          model_cache_indices[model_id].begin();
    }
    model_cache_indices_offsets[model_id][num_instances] = sum;

    /* Convert to buffer indices */
    for (uint32_t idx = 0; idx < sum; idx++) {
      model_cache_indices[model_id][idx] =
          model_cache_indices[model_id][idx] % num_frequent_per_model +
          num_frequent_per_model * model_id;
    }

    model_cache_indices[model_id].resize(sum);
  }
}

template <typename dtype, typename emtype>
void HybridEmbeddingCpu<dtype, emtype>::calculate_frequent_network_cache_indices() {
  const uint32_t num_frequent_per_model = num_frequent / num_instances;

  if (network_cache_mask.size() == 0) calculate_frequent_network_cache_mask();

  network_cache_indices.resize(num_instances);
  network_cache_indices_offsets.resize(num_instances);

  for (uint32_t network_id = 0; network_id < num_instances; network_id++) {
    network_cache_indices[network_id].resize(num_frequent);
    network_cache_indices_offsets[network_id].resize(num_instances + 1);

    uint32_t sum = 0;
    for (uint32_t i = 0; i < num_frequent; ++i) {
      if (network_cache_mask[network_id][i]) {
        network_cache_indices[network_id][sum] = i;
        sum++;
      }
    }

    /* Compute offsets */
    for (uint32_t i = 0; i < num_instances; i++) {
      network_cache_indices_offsets[network_id][i] =
          std::lower_bound(network_cache_indices[network_id].begin(),
                           network_cache_indices[network_id].begin() + sum,
                           i * num_frequent_per_model) -
          network_cache_indices[network_id].begin();
    }
    network_cache_indices_offsets[network_id][num_instances] = sum;

    network_cache_indices[network_id].resize(sum);
  }
}

template <typename dtype, typename emtype>
void HybridEmbeddingCpu<dtype, emtype>::calculate_frequent_network_cache_mask() {
  network_cache_mask.resize(num_instances);

  for (uint32_t network_id = 0; network_id < num_instances; network_id++) {
    network_cache_mask[network_id].resize(num_frequent);

    for (uint32_t j = local_batch_size * network_id;
         j < std::min(batch_size, local_batch_size * (network_id + 1)); j++) {
      for (uint32_t i = 0; i < num_tables; i++) {
        uint32_t idx = j * num_tables + i;
        dtype category = samples[idx];
        if (category_location[2 * category] == num_instances) {
          dtype frequent_index = category_location[2 * category + 1];
          network_cache_mask[network_id][frequent_index] = 1;
        }
      }
    }
  }
}

template <typename dtype, typename emtype>
void HybridEmbeddingCpu<dtype, emtype>::generate_embedding_vectors() {
  frequent_embedding_vectors.resize(num_instances);
  infrequent_embedding_vectors.resize(num_instances);

  // Fixed seed for reproducibility
  std::default_random_engine generator(1234UL);
  std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);

  for (size_t i = 0; i < num_instances; i++) {
    frequent_embedding_vectors[i].resize(num_frequent * embedding_vec_size);
    infrequent_embedding_vectors[i].resize(
        utils::ceildiv<dtype>(num_categories - num_frequent, num_instances) * embedding_vec_size);
  }
  for (dtype category = 0; category < num_categories; category++) {
    dtype model_id = category_location[2 * category];
    dtype location = category_location[2 * category + 1];
    if (model_id == num_instances) {
      dtype freq_index = location;
      HCTR_CHECK(freq_index < num_frequent);
      for (uint32_t k = 0; k < embedding_vec_size; k++) {
        float value = distribution(generator);
        for (uint32_t i = 0; i < num_instances; i++)
          frequent_embedding_vectors[i][freq_index * embedding_vec_size + k] = value;
      }
    } else {
      for (uint32_t k = 0; k < embedding_vec_size; k++)
        infrequent_embedding_vectors[model_id][location * embedding_vec_size + k] =
            distribution(generator);
    }
  }
}

template <typename dtype, typename emtype>
void HybridEmbeddingCpu<dtype, emtype>::generate_gradients() {
  gradients.resize(num_instances);

  // Fixed seed for reproducibility
  std::default_random_engine generator(1234UL);
  std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);

  for (size_t i = 0; i < num_instances; i++)
    gradients[i].resize(local_samples_size * embedding_vec_size);
  for (size_t i = 0; i < num_instances; i++) {
    for (size_t j = 0; j < local_samples_size; j++) {
      for (size_t k = 0; k < embedding_vec_size; k++) {
        gradients[i][j * embedding_vec_size + k] =
            utils::TypeConvertFunc<emtype, float>::convert(distribution(generator));
      }
    }
  }
}

template <typename dtype, typename emtype>
void HybridEmbeddingCpu<dtype, emtype>::forward_a2a_messages() {
  forward_sent_messages.resize(num_instances);
  forward_received_messages.resize(num_instances);

  for (uint32_t i = 0; i < num_instances; i++) {
    for (uint32_t j = 0; j < num_instances; j++) {
      uint32_t k0 = model_indices_offsets[i][j];
      uint32_t k1 = model_indices_offsets[i][j + 1];
      for (uint32_t k = k0; k < k1; ++k) {
        uint32_t model_indices_to_dst = model_indices[i][k];
        dtype category_to_dst = samples[model_indices_to_dst];
        uint32_t embedding_vec_indices = category_location[2 * category_to_dst + 1];
        for (uint32_t m = 0; m < embedding_vec_size; ++m) {
          emtype value = utils::TypeConvertFunc<emtype, float>::convert(
              infrequent_embedding_vectors[i][embedding_vec_indices * embedding_vec_size + m]);
          forward_received_messages[j].push_back(value);
          forward_sent_messages[i].push_back(value);
        }
      }
    }
  }
}

template <typename dtype, typename emtype>
void HybridEmbeddingCpu<dtype, emtype>::forward_a2a_messages_hier() {
  forward_sent_messages.resize(num_instances);
  forward_received_messages.resize(num_instances);
  for (uint32_t i = 0; i < num_instances; i++) {
    forward_received_messages[i].resize(num_instances * local_samples_size * embedding_vec_size);
    forward_sent_messages[i].resize(num_instances * local_samples_size * embedding_vec_size);
  }

  uint32_t instances_per_node = num_instances / num_nodes;

  for (uint32_t model_id = 0; model_id < num_instances; model_id++) {
    for (uint32_t network_id = 0; network_id < num_instances; network_id++) {
      uint32_t k0 = model_indices_offsets[model_id][network_id];
      uint32_t k1 = model_indices_offsets[model_id][network_id + 1];
      for (uint32_t k = k0; k < k1; ++k) {
        uint32_t index = model_indices[model_id][k];
        dtype category = samples[index];
        uint32_t location = category_location[2 * category + 1];
        for (uint32_t m = 0; m < embedding_vec_size; ++m) {
          emtype value = utils::TypeConvertFunc<emtype, float>::convert(
              infrequent_embedding_vectors[model_id][location * embedding_vec_size + m]);
          forward_received_messages[network_id]
                                   [(model_id * local_samples_size + k - k0) * embedding_vec_size +
                                    m] = value;
          forward_sent_messages
              [model_id - model_id % instances_per_node + network_id % instances_per_node]
              [((network_id - network_id % instances_per_node + model_id % instances_per_node) *
                    local_samples_size +
                k - k0) *
                   embedding_vec_size +
               m] = value;
        }
      }
    }
  }
}

template <typename dtype, typename emtype>
void HybridEmbeddingCpu<dtype, emtype>::backward_a2a_messages() {
  backward_sent_messages.resize(num_instances);
  backward_received_messages.resize(num_instances);

  for (size_t i = 0; i < num_instances; i++) {
    for (size_t j = 0; j < num_instances; j++) {
      uint32_t k0 = model_indices_offsets[i][j];
      uint32_t k1 = model_indices_offsets[i][j + 1];
      for (size_t k = k0; k < k1; ++k) {
        uint32_t index = model_indices[i][k];
        uint32_t local_index = index % local_samples_size;
        for (uint32_t m = 0; m < embedding_vec_size; ++m) {
          backward_sent_messages[j].push_back(gradients[j][local_index * embedding_vec_size + m]);
          backward_received_messages[i].push_back(
              gradients[j][local_index * embedding_vec_size + m]);
        }
      }
    }
  }
}

template <typename dtype, typename emtype>
void HybridEmbeddingCpu<dtype, emtype>::backward_a2a_messages_hier() {
  backward_sent_messages.resize(num_instances);
  backward_received_messages.resize(num_instances);
  for (uint32_t i = 0; i < num_instances; i++) {
    backward_received_messages[i].resize(num_instances * local_samples_size * embedding_vec_size);
    backward_sent_messages[i].resize(num_instances * local_samples_size * embedding_vec_size);
  }

  uint32_t instances_per_node = num_instances / num_nodes;

  for (size_t model_id = 0; model_id < num_instances; model_id++) {
    for (size_t network_id = 0; network_id < num_instances; network_id++) {
      uint32_t k0 = model_indices_offsets[model_id][network_id];
      uint32_t k1 = model_indices_offsets[model_id][network_id + 1];
      for (size_t k = k0; k < k1; ++k) {
        uint32_t index = model_indices[model_id][k];
        uint32_t local_index = index % local_samples_size;
        for (uint32_t m = 0; m < embedding_vec_size; ++m) {
          emtype value = gradients[network_id][local_index * embedding_vec_size + m];
          backward_received_messages[model_id][(network_id * local_samples_size + k - k0) *
                                                   embedding_vec_size +
                                               m] = value;
          backward_sent_messages
              [network_id - network_id % instances_per_node + model_id % instances_per_node]
              [((model_id - model_id % instances_per_node + network_id % instances_per_node) *
                    local_samples_size +
                k - k0) *
                   embedding_vec_size +
               m] = value;
        }
      }
    }
  }
}

template <typename dtype, typename emtype>
void HybridEmbeddingCpu<dtype, emtype>::infrequent_update() {
  for (size_t network_id = 0; network_id < num_instances; network_id++) {
    for (size_t j = 0; j < local_samples_size; j++) {
      dtype category = samples[network_id * local_samples_size + j];
      dtype model_id = category_location[2 * category];
      dtype location = category_location[2 * category + 1];
      if (model_id < num_instances) {
        {
          for (uint32_t k = 0; k < embedding_vec_size; k++)
            infrequent_embedding_vectors[model_id][location * embedding_vec_size + k] -=
                lr * utils::TypeConvertFunc<float, emtype>::convert(
                         gradients[network_id][j * embedding_vec_size + k]);
        }
      }
    }
  }
}

template <typename dtype, typename emtype>
void HybridEmbeddingCpu<dtype, emtype>::frequent_reduce_gradients() {
  // Reduce to a float32 array
  std::vector<float> reduced_gradients_f32(num_frequent * embedding_vec_size, 0.0f);
  for (size_t network_id = 0; network_id < num_instances; network_id++) {
    for (size_t j = 0; j < local_samples_size; j++) {
      dtype category = samples[network_id * local_samples_size + j];
      dtype model_id = category_location[2 * category];
      if (model_id == num_instances) {
        dtype freq_index = category_location[2 * category + 1];
        HCTR_CHECK(freq_index < num_frequent);
        for (uint32_t k = 0; k < embedding_vec_size; k++) {
          reduced_gradients_f32[freq_index * embedding_vec_size + k] +=
              utils::TypeConvertFunc<float, emtype>::convert(
                  gradients[network_id][j * embedding_vec_size + k]);
        }
      }
    }
  }

  // Copy to the emtype array
  reduced_gradients.resize(num_frequent * embedding_vec_size);
  for (size_t i = 0; i < num_frequent * embedding_vec_size; i++) {
    reduced_gradients[i] = utils::TypeConvertFunc<emtype, float>::convert(reduced_gradients_f32[i]);
  }
}

template <typename dtype, typename emtype>
void HybridEmbeddingCpu<dtype, emtype>::frequent_update() {
  for (size_t model_id = 0; model_id < num_instances; model_id++) {
    for (size_t i = 0; i < num_frequent * embedding_vec_size; i++) {
      frequent_embedding_vectors[model_id][i] -=
          lr * utils::TypeConvertFunc<float, emtype>::convert(reduced_gradients[i]);
    }
  }
}

template <typename dtype, typename emtype>
void HybridEmbeddingCpu<dtype, emtype>::frequent_update_single_node() {
  uint32_t num_frequent_per_model = num_frequent / num_instances;
  for (size_t network_id = 0; network_id < num_instances; network_id++) {
    for (size_t j = 0; j < local_samples_size; j++) {
      dtype category = samples[network_id * local_samples_size + j];
      dtype model_id = category_location[2 * category];
      if (model_id == num_instances) {
        dtype freq_index = category_location[2 * category + 1];
        HCTR_CHECK(freq_index < num_frequent);
        uint32_t frequent_model_id = freq_index / num_frequent_per_model;
        for (uint32_t k = 0; k < embedding_vec_size; k++)
          frequent_embedding_vectors[frequent_model_id][freq_index * embedding_vec_size + k] -=
              lr * utils::TypeConvertFunc<float, emtype>::convert(
                       gradients[network_id][j * embedding_vec_size + k]);
      }
    }
  }
}

template <typename dtype, typename emtype>
void HybridEmbeddingCpu<dtype, emtype>::forward_network() {
  interaction_layer_input.resize(num_instances);

  for (uint32_t network_id = 0; network_id < num_instances; network_id++) {
    interaction_layer_input[network_id].resize(local_samples_size * embedding_vec_size);

    for (uint32_t i = 0; i < local_samples_size; i++) {
      dtype category = samples[local_samples_size * network_id + i];
      dtype model_id = category_location[2 * category];
      dtype location = category_location[2 * category + 1];
      if (model_id == num_instances) {
        dtype freq_index = location;
        HCTR_CHECK(freq_index < num_frequent);
        for (uint32_t k = 0; k < embedding_vec_size; k++) {
          interaction_layer_input[network_id][embedding_vec_size * i + k] =
              frequent_embedding_vectors[network_id][embedding_vec_size * freq_index + k];
        }
      } else {
        for (uint32_t k = 0; k < embedding_vec_size; k++) {
          interaction_layer_input[network_id][embedding_vec_size * i + k] =
              infrequent_embedding_vectors[model_id][embedding_vec_size * location + k];
        }
      }
    }
  }
}

template <typename dtype, typename emtype>
void HybridEmbeddingCpu<dtype, emtype>::frequent_forward_model() {
  frequent_embedding_vectors_cache.resize(num_instances);

  for (uint32_t network_id = 0; network_id < num_instances; network_id++) {
    if (sizeof(emtype) != sizeof(float)) {
      // Separate buffers, initialize with zeros
      frequent_embedding_vectors_cache[network_id].resize(num_frequent * embedding_vec_size,
                                                          (emtype)0.0);
    } else {
      // Same buffers, copy previous values
      frequent_embedding_vectors_cache[network_id].resize(num_frequent * embedding_vec_size);
      for (size_t i = 0; i < num_frequent * embedding_vec_size; i++) {
        frequent_embedding_vectors_cache[network_id][i] =
            utils::TypeConvertFunc<emtype, float>::convert(
                frequent_embedding_vectors[network_id][i]);
      }
    }
  }

  for (uint32_t network_id = 0; network_id < num_instances; network_id++) {
    for (uint32_t model_id = 0; model_id < num_instances; model_id++) {
      uint32_t i0 = network_cache_indices_offsets[network_id][model_id];
      uint32_t i1 = network_cache_indices_offsets[network_id][model_id + 1];
      for (uint32_t i = i0; i < i1; i++) {
        uint32_t freq_index = network_cache_indices[network_id][i];
        for (uint32_t k = 0; k < embedding_vec_size; k++) {
          frequent_embedding_vectors_cache[network_id][embedding_vec_size * freq_index + k] =
              utils::TypeConvertFunc<emtype, float>::convert(
                  frequent_embedding_vectors[model_id][embedding_vec_size * freq_index + k]);
        }
      }
    }
  }
}

template class HybridEmbeddingCpu<uint32_t, __half>;
template class HybridEmbeddingCpu<uint32_t, float>;
template class HybridEmbeddingCpu<long long, __half>;
template class HybridEmbeddingCpu<long long, float>;