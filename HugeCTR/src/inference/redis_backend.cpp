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

#include <inference/redis_backend.hpp>
#include <iostream>
#include <optional>
#include <unordered_set>

#define DEBUG                                                  \
  std::cout << "[DEBUG]" << __FILE__ << "|" << __func__ << "|" \
            << "LINE" << __LINE__ << " "

namespace HugeCTR {

std::string make_hash_key(const std::string& table_name, const size_t partition) {
  return table_name + "/p" + std::to_string(partition);
}

template <typename TKey>
RedisClusterBackend<TKey>::RedisClusterBackend(const std::string& address,
                                               const std::string& password,
                                               const size_t num_partitions,
                                               const size_t max_get_batch_size,
                                               const size_t max_set_batch_size)
    : TBase(),
      num_partitions_(num_partitions),
      max_get_batch_size_(max_get_batch_size),
      max_set_batch_size_(max_set_batch_size) {
  sw::redis::ConnectionOptions options;

  const std::string::size_type comma_pos = address.find(',');
  const std::string host = comma_pos == std::string::npos ? address : address.substr(0, comma_pos);

  const std::string::size_type colon_pos = host.find(':');
  if (colon_pos == std::string::npos) {
    options.host = host;
  } else {
    options.host = host.substr(0, colon_pos);
    options.port = std::stoi(host.substr(colon_pos + 1));
  }
  options.password = password;
  options.keep_alive = true;

  sw::redis::ConnectionPoolOptions pool_options;
  pool_options.size = 1;

  DEBUG << "Connecting to Redis cluster via " << options.host << ":" << options.port << " ..."
        << std::endl;

  redis_ = std::make_unique<sw::redis::RedisCluster>(options, pool_options);

  DEBUG << "Connected to Redis database!" << std::endl;
}

template <typename TKey>
RedisClusterBackend<TKey>::~RedisClusterBackend() {
  DEBUG << "Disconnecting from Redis database..." << std::endl;

  redis_.reset();

  DEBUG << "Disconnected from Redis database!" << std::endl;
}

template <typename TKey>
const char* RedisClusterBackend<TKey>::get_name() const {
  return "RedisCluster";
}

template <typename TKey>
size_t RedisClusterBackend<TKey>::contains(const std::string& table_name, const size_t num_keys,
                                           const TKey* const keys) const {
  size_t hit_count = 0;

  switch (num_keys) {
    case 0: {
      break;
    }
    case 1: {
      const size_t partition = *keys % num_partitions_;
      const std::string& hkey = make_hash_key(table_name, partition);
      const std::string k(reinterpret_cast<const char*>(keys), sizeof(TKey));

      if (redis_->hexists(hkey, k)) {
        hit_count++;
      }
      break;
    }
    default: {
      auto fn = [&, this](const size_t partition) -> size_t {
        const std::string& hkey = make_hash_key(table_name, partition);

        // Form query.
        std::unordered_set<TKey> query;
        const TKey* const keys_end = &keys[num_keys];
        for (const TKey* k = keys; k != keys_end; k++) {
          if (*k % num_partitions_ == partition) {
            query.emplace(*k);
          }
        }

        // Enumerate keys.
        std::vector<std::string> existing_keys;
        redis_->hkeys(hkey, std::back_inserter(existing_keys));

        size_t hit_count = 0;

        // Iterate over keys and check 1 by 1.
        for (const auto& k : existing_keys) {
          assert(k.size() == sizeof(TKey));
          if (query.find(*reinterpret_cast<const TKey*>(k.data())) != query.end()) {
            hit_count++;
          }
        }

        return hit_count;
      };

      // Process partitions.
      for (size_t partition = 0; partition < num_partitions_; partition++) {
        hit_count += fn(partition);
      }
      break;
    }
  }

  DEBUG << "Redis table: " << table_name << ". Contains " << hit_count << " / " << num_keys
        << " keys." << std::endl;
  return hit_count;
}

template <typename TKey>
bool RedisClusterBackend<TKey>::insert(const std::string& table_name, const size_t num_pairs,
                                       const TKey* const keys, const char* const values,
                                       const size_t value_size) {
  size_t num_inserts = 0;

  auto fn = [&, this](const size_t partition) -> void {
    const std::string& hkey = make_hash_key(table_name, partition);

    std::vector<std::pair<std::string, std::string>> batch_pairs;

    size_t num_queries = 0;

    const TKey* const keys_end = &keys[num_pairs];
    for (const TKey* k = keys; k != keys_end; num_queries++) {
      batch_pairs.clear();
      for (; k != keys_end; k++) {
        if (batch_pairs.size() >= max_set_batch_size_) {
          break;
        }
        if (*k % num_partitions_ == partition) {
          batch_pairs.emplace_back(
              std::piecewise_construct,
              std::forward_as_tuple(reinterpret_cast<const char*>(k), sizeof(TKey)),
              std::forward_as_tuple(&values[(k - keys) * value_size], value_size));
        }
      }

      redis_->hmset(hkey, batch_pairs.begin(), batch_pairs.end());
      num_inserts += batch_pairs.size();
      DEBUG << "Redis partition " << hkey << ", query " << num_queries << ": Inserted "
            << batch_pairs.size() << " keys." << std::endl;
    }
  };

  // Process partitions.
  for (size_t partition = 0; partition < num_partitions_; partition++) {
    fn(partition);
  }
  DEBUG << "Redis table " << table_name << ": Inserted " << num_inserts << " / " << num_pairs
        << " keys." << std::endl;
  return true;
}

template <typename TKey>
size_t RedisClusterBackend<TKey>::fetch(const std::string& table_name, const size_t num_keys,
                                        const TKey* const keys, char* const values,
                                        const size_t value_size,
                                        MissingKeyCallback& missing_callback) const {
  auto fn = [&, this](const size_t partition) -> size_t {
    const std::string& hkey = make_hash_key(table_name, partition);

    std::vector<size_t> batch_indices;
    std::vector<std::string> batch_keys;
    std::vector<std::optional<std::string>> batch_values;

    size_t hit_count = 0;
    size_t num_queries = 0;

    const TKey* const keys_end = &keys[num_keys];
    for (const TKey* k = keys; k != keys_end; num_queries++) {
      // Prepare and launch query.
      batch_indices.clear();
      batch_keys.clear();
      for (; k != keys_end; k++) {
        if (batch_keys.size() >= max_set_batch_size_) {
          break;
        }
        if (*k % num_partitions_ == partition) {
          batch_indices.push_back(k - keys);
          batch_keys.emplace_back(reinterpret_cast<const char*>(k), sizeof(TKey));
        }
      }

      batch_values.clear();
      redis_->hmget(hkey, batch_keys.begin(), batch_keys.end(), std::back_inserter(batch_values));

      // Process results.
      auto batch_indices_it = batch_indices.begin();
      for (const auto& value : batch_values) {
        if (value) {
          if (value->size() != value_size) {
            DEBUG << "Redis return value embedding size mismatch!" << std::endl;
            exit(EXIT_FAILURE);
          }
          memcpy(&values[*batch_indices_it * value_size], value->data(), value_size);
          hit_count++;
        } else {
          missing_callback(*batch_indices_it);
        }
        batch_indices_it++;
      }

      DEBUG << "Redis partition " << hkey << " | query " << num_queries << ": Fetched "
            << batch_values.size() << " keys. Hits: " << hit_count << std::endl;
    }

    return hit_count;
  };

  // Process partitions.
  size_t hit_count = 0;
  for (size_t partition = 0; partition < num_partitions_; partition++) {
    hit_count += fn(partition);
  }
  DEBUG << "Redis table: " << table_name << ". Fetched " << hit_count << " / " << num_keys
        << " keys." << std::endl;
  return hit_count;
}

template <typename TKey>
size_t RedisClusterBackend<TKey>::fetch(const std::string& table_name, const size_t num_indices,
                                        const size_t* const indices, const TKey* const keys,
                                        char* const values, const size_t value_size,
                                        MissingKeyCallback& missing_callback) const {
  auto fn = [&, this](const size_t partition) -> size_t {
    const std::string& hkey = make_hash_key(table_name, partition);

    std::vector<std::string> batch_keys;
    std::vector<std::optional<std::string>> batch_values;

    size_t hit_count = 0;
    size_t num_queries = 0;

    const size_t* const indices_end = &indices[num_indices];
    for (const size_t* i = indices; i != indices_end; num_queries++) {
      // Create and launch query.
      batch_keys.clear();
      for (const size_t* tmp_i = indices; tmp_i != indices_end; tmp_i++) {
        if (batch_keys.size() >= max_get_batch_size_) {
          break;
        }
        const TKey& k = keys[*tmp_i];
        if (k % num_partitions_ == partition) {
          batch_keys.emplace_back(reinterpret_cast<const char*>(&k), sizeof(TKey));
        }
      }

      batch_values.clear();
      redis_->hmget(hkey, batch_keys.begin(), batch_keys.end(), std::back_inserter(batch_values));

      // Process results.
      for (const auto& value : batch_values) {
        if (value) {
          assert(value->size() == value_size);
          memcpy(&values[*i * value_size], value->data(), value_size);
          hit_count++;
        } else {
          missing_callback(*i);
        }
        i++;
      }

      DEBUG << "Redis partition " << hkey << " | query " << num_queries << ": Fetched "
            << batch_values.size() << " keys. Hits: " << hit_count << std::endl;
    }

    return hit_count;
  };

  // Process partitions.
  size_t hit_count = 0;
  for (size_t partition = 0; partition < num_partitions_; partition++) {
    hit_count += fn(partition);
  }
  DEBUG << "Redis table " << table_name << ": Fetched " << hit_count << " / " << num_indices
        << " keys." << std::endl;
  return hit_count;
}

template <typename TKey>
size_t RedisClusterBackend<TKey>::evict(const std::string& table_name) {
  auto fn = [&](const size_t partition) -> size_t {
    const std::string& hkey = make_hash_key(table_name, partition);

    // Enumerate keys.
    std::vector<std::string> keys;
    redis_->hkeys(hkey, std::back_inserter(keys));

    // Delete the keys.
    return redis_->hdel(hkey, keys.begin(), keys.end());
  };

  // Process partitions.
  size_t hit_count = 0;
  for (size_t partition = 0; partition < num_partitions_; partition++) {
    hit_count += fn(partition);
  }
  DEBUG << "Redis table " << table_name << ": Deleted " << hit_count << " keys." << std::endl;
  return hit_count;
}

template <typename TKey>
size_t RedisClusterBackend<TKey>::evict(const std::string& table_name, const size_t num_keys,
                                        const TKey* const keys) {
  auto fn = [&, this](const size_t partition) -> size_t {
    const std::string& hkey = make_hash_key(table_name, partition);

    std::vector<std::string> batch;

    size_t hit_count = 0;
    size_t num_queries = 0;

    const TKey* const keys_end = &keys[num_keys];
    for (const TKey* k = keys; k != keys_end; num_queries++) {
      batch.clear();
      for (; k != keys_end; k++) {
        if (batch.size() >= max_set_batch_size_) {
          break;
        }
        if (*k % num_partitions_ == partition) {
          batch.emplace_back(reinterpret_cast<const char*>(k), sizeof(TKey));
        }
      }

      hit_count += redis_->hdel(hkey, batch.begin(), batch.end());
      DEBUG << "Redis partition " << hkey << " | query " << num_queries << ": Deleted "
            << batch.size() << " keys. Hits: " << hit_count << std::endl;
    }

    return hit_count;
  };

  // Process partitions.
  size_t hit_count = 0;
  for (size_t partition = 0; partition < num_partitions_; partition++) {
    hit_count += fn(partition);
  }
  DEBUG << "Redis table " << table_name << ": Deleted " << hit_count << " / " << num_keys
        << " keys." << std::endl;
  return hit_count;
}

template class RedisClusterBackend<unsigned int>;
template class RedisClusterBackend<long long>;

}  // namespace HugeCTR