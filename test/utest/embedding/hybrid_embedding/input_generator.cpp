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

#include "input_generator.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <set>

#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

template <typename dtype>
std::vector<size_t> HybridEmbeddingInputGenerator<dtype>::generate_rand_table_sizes(
    size_t num_tables, size_t vec_size, double max_mem) {
  std::vector<size_t> table_sizes(num_tables);

  // mem = sizeof(float) * vec_size * num_tables * max_table_size;
  // =>
  const size_t max_table_size = (size_t)(max_mem / (sizeof(float) * vec_size * num_tables));
  const double max_exp = log(max_table_size) / log(10.);

  for (size_t embedding = 0; embedding < num_tables; ++embedding) {
    double r = rand() / (double)RAND_MAX;
    table_sizes[embedding] = std::max((size_t)2, (size_t)floor(pow(10., 1. + r * (max_exp - 1))));
  }

  return table_sizes;
}

template <typename dtype>
void HybridEmbeddingInputGenerator<dtype>::generate_uniform_rand_table_sizes(size_t num_categories,
                                                                             size_t num_tables) {
  if (num_categories > 0) config_.num_categories = num_categories;
  if (num_tables > 0) config_.num_tables = num_tables;

  std::set<size_t> separators;
  separators.insert(0);
  separators.insert(config_.num_categories);
  std::uniform_int_distribution<size_t> dist(1, config_.num_categories - 1);

  for (size_t i = 0; i < config_.num_tables - 1; i++) {
    size_t sep;
    do {
      sep = dist(gen_);
    } while (separators.find(sep) != separators.end());
    separators.insert(sep);
  }

  for (auto it = std::next(separators.begin()); it != separators.end(); it++) {
    table_sizes_.push_back(*it - *(std::prev(it)));
  }
}

template <typename dtype>
void HybridEmbeddingInputGenerator<dtype>::create_probability_distribution() {
  const size_t num_embeddings = table_sizes_.size();
  std::uniform_real_distribution<double> distr(0.3, 0.8);

  embedding_prob_distribution_.resize(num_embeddings);
  for (size_t embedding = 0; embedding < num_embeddings; ++embedding) {
    embedding_prob_distribution_[embedding].resize(table_sizes_[embedding]);
  }

  for (size_t embedding = 0; embedding < num_embeddings; ++embedding) {
    const size_t embedding_size = table_sizes_[embedding];
    std::vector<size_t> embedding_shuffle_arg(table_sizes_[embedding]);
    std::iota(embedding_shuffle_arg.begin(), embedding_shuffle_arg.end(), (size_t)0);
    std::shuffle(embedding_shuffle_arg.begin(), embedding_shuffle_arg.end(), gen_);
    embedding_shuffle_args.push_back(embedding_shuffle_arg);
    if (embedding_size < 30) {
      // choose uniform distribution
      for (size_t c_e = 0; c_e < table_sizes_[embedding]; ++c_e)
        embedding_prob_distribution_[embedding][c_e] = 1. / (double)embedding_size;
    } else {
      size_t size_first = std::max((size_t)1, size_t(4. * log10((double)embedding_size)));
      size_first = std::min((size_t)embedding_size, (size_t)size_first);
      double acc_prob_first = distr(gen_);
      // a * (1 - r^n) / (1 - r) = acc_p
      // Let a * r^{n} = 0.02 * acc_prob_first
      // a - 0.02 * acc_prob_first = acc_prob_first * (1-r)

      // (1 + 0.02) * acc_prob_first - a = r * acc_prob_first
      double r = 0.9;
      double a = acc_prob_first * (1. - r) / (1. - pow(r, (double)size_first));
      for (size_t c_e = 0; c_e < size_first; ++c_e)
        embedding_prob_distribution_[embedding][c_e] = a * pow(r, (double)c_e);

      // the following is approximate, will be normalized..
      //
      // now apply power law to the remaining elements:
      //
      //   p = a * n^{-2}
      // => 1 - acc_prob_first = a / N - a / n
      // => a ( 1/n - 1/N ) = 1 - acc_prob_first
      // => a (N-n) / (nN) = 1 - acc_prob_first
      // => a = n * N / (N-n) * (1 - acc_prob_first)

      a = size_first * embedding_size / (embedding_size - size_first) * (1. - acc_prob_first);
      for (size_t c_e = size_first; c_e < table_sizes_[embedding]; ++c_e)
        embedding_prob_distribution_[embedding][c_e] = a * pow((double)c_e, -2.);

      // normalize probability distribution
      // calculate norm
      double sum_p = 0.;
      for (size_t c_e = 0; c_e < table_sizes_[embedding]; ++c_e)
        sum_p += embedding_prob_distribution_[embedding][c_e];
      // correct
      for (size_t c_e = 0; c_e < table_sizes_[embedding]; ++c_e)
        embedding_prob_distribution_[embedding][c_e] /= sum_p;
    }
  }
}

template <typename dtype>
void HybridEmbeddingInputGenerator<dtype>::generate_categories(dtype* data, size_t batch_size,
                                                               bool normalized) {
  const size_t num_embeddings = table_sizes_.size();
  std::uniform_real_distribution<double> distr(0, 1);
  std::vector<dtype> embedding_offsets;
  HugeCTR::hybrid_embedding::EmbeddingTableFunctors<dtype>::get_embedding_offsets(embedding_offsets,
                                                                                  table_sizes_);
  // create samples
  for (size_t embedding = 0; embedding < num_embeddings; ++embedding) {
    std::vector<size_t>& embedding_shuffle_arg = embedding_shuffle_args[embedding];
    std::vector<double>& f_prob_e = embedding_prob_distribution_[embedding];
    std::vector<double> acc_prob(f_prob_e.size() + 1, 0.0);
    double acc = 0.;
    for (size_t c_e = 0; c_e < f_prob_e.size(); ++c_e) {
      acc_prob[c_e] = acc;
      acc += f_prob_e[c_e];
    }

    acc_prob.front() = -42.0;
    acc_prob.back() = 42.0;

    for (size_t sample = 0; sample < batch_size; ++sample) {
      double r = distr(gen_);
      size_t category =
          (size_t)(std::lower_bound(acc_prob.begin(), acc_prob.end(), r) - acc_prob.begin()) - 1;

      // category index within table
      size_t category_shuffled = embedding_shuffle_arg[category];
      data[sample * num_embeddings + embedding] = category_shuffled;

      if (normalized) {
        data[sample * num_embeddings + embedding] += (size_t)embedding_offsets[embedding];
      }
    }
  }
}

template <typename dtype>
void HybridEmbeddingInputGenerator<dtype>::generate_category_location() {
  std::uniform_int_distribution<dtype> distr(0, config_.num_instances - 1);

  std::vector<double> all_probabilities;
  for (auto& v : embedding_prob_distribution_) {
    all_probabilities.insert(all_probabilities.end(), v.begin(), v.end());
  }
  std::vector<dtype> original_index(config_.num_categories);
  std::iota(original_index.begin(), original_index.end(), (dtype)0);

  std::sort(original_index.begin(), original_index.end(), [&all_probabilities](dtype i1, dtype i2) {
    return all_probabilities[i1] < all_probabilities[i2];
  });

  // First num_frequent categories are frequent
  category_location_.resize(2 * config_.num_categories);
  category_frequent_index_.resize(config_.num_categories, config_.num_frequent);
  for (dtype i = 0; i < config_.num_frequent; i++) {
    dtype cat = original_index[i];
    category_location_[2 * cat + 0] = config_.num_categories;
    category_location_[2 * cat + 1] = config_.num_categories;
    category_frequent_index_[cat] = i;
  }

  dtype max_size_per_instance =
      (config_.num_categories - config_.num_frequent + config_.num_instances - 1) /
      config_.num_instances;
  std::vector<dtype> sizes_per_instance(config_.num_instances, 0);
  for (dtype i = config_.num_frequent; i < config_.num_categories; i++) {
    dtype cat = original_index[i];
    dtype instance;
    do {
      instance = distr(gen_);
      // If the selected instance is already full, pick another one
    } while (sizes_per_instance[instance] == max_size_per_instance);
    category_location_[2 * cat + 0] = instance;
    category_location_[2 * cat + 1] = sizes_per_instance[instance]++;
  }
}

template <typename dtype>
HybridEmbeddingInputGenerator<dtype>::HybridEmbeddingInputGenerator(
    HybridEmbeddingConfig<dtype> config, size_t seed)
    : config_(config), seed_(seed), gen_(seed) {
  generate_uniform_rand_table_sizes(config_.num_categories, config_.num_tables);
  create_probability_distribution();
}

template <typename dtype>
HybridEmbeddingInputGenerator<dtype>::HybridEmbeddingInputGenerator(
    HybridEmbeddingConfig<dtype> config, const std::vector<size_t>& table_sizes, size_t seed)
    : config_(config), table_sizes_(table_sizes), seed_(seed), gen_(seed) {
  config_.num_tables = table_sizes.size();
  config_.num_categories = std::accumulate(table_sizes.begin(), table_sizes.end(), 0);
  create_probability_distribution();
}

template <typename dtype>
std::vector<dtype> HybridEmbeddingInputGenerator<dtype>::generate_categorical_input(
    size_t batch_size, size_t num_tables) {
  table_sizes_ = generate_rand_table_sizes(num_tables);
  config_.num_tables = table_sizes_.size();
  config_.num_categories = std::accumulate(table_sizes_.begin(), table_sizes_.end(), 0);
  create_probability_distribution();

  std::vector<dtype> data(batch_size * config_.num_tables);
  generate_categories(data.data(), batch_size, false);
  return data;
}

template <typename dtype>
std::vector<dtype> HybridEmbeddingInputGenerator<dtype>::generate_flattened_categorical_input(
    size_t batch_size, size_t num_tables) {
  table_sizes_ = generate_rand_table_sizes(num_tables);
  config_.num_tables = table_sizes_.size();
  config_.num_categories = std::accumulate(table_sizes_.begin(), table_sizes_.end(), 0);
  create_probability_distribution();

  std::vector<dtype> data(batch_size * config_.num_tables);
  generate_categories(data.data(), batch_size, true);
  return data;
}

template <typename dtype>
std::vector<dtype> HybridEmbeddingInputGenerator<dtype>::generate_categorical_input(
    size_t batch_size) {
  std::vector<dtype> data(batch_size * config_.num_tables);
  generate_categories(data.data(), batch_size, false);
  return data;
}

template <typename dtype>
std::vector<dtype> HybridEmbeddingInputGenerator<dtype>::generate_flattened_categorical_input(
    size_t batch_size) {
  std::vector<dtype> data(batch_size * config_.num_tables);
  generate_categories(data.data(), batch_size, true);
  return data;
}

template <typename dtype>
void HybridEmbeddingInputGenerator<dtype>::generate_categorical_input(dtype* batch,
                                                                      size_t batch_size) {
  generate_categories(batch, batch_size, false);
}

template <typename dtype>
void HybridEmbeddingInputGenerator<dtype>::generate_flattened_categorical_input(dtype* batch,
                                                                                size_t batch_size) {
  generate_categories(batch, batch_size, true);
}

template <typename dtype>
std::vector<dtype>& HybridEmbeddingInputGenerator<dtype>::get_category_location() {
  return category_location_;
}

template <typename dtype>
std::vector<dtype>& HybridEmbeddingInputGenerator<dtype>::get_category_frequent_index() {
  return category_frequent_index_;
}

template <typename dtype>
std::vector<size_t>& HybridEmbeddingInputGenerator<dtype>::get_table_sizes() {
  return table_sizes_;
}

template class HybridEmbeddingInputGenerator<uint32_t>;
template class HybridEmbeddingInputGenerator<long long>;
}  // namespace hybrid_embedding

}  // namespace HugeCTR