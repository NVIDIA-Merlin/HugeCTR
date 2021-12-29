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
#pragma once

#include <functional>
#include <inference/inference_utils.hpp>
#include <string>

namespace HugeCTR {

/**
 * Format of callback that is invoked by \p fetch methods of the \p DatabaseBackend if a key does
 * not exist.
 */
using MissingKeyCallback = std::function<void(const size_t)>;

/**
 * Base class for database backends. Implementations that inherit from this should override all
 * public members.
 *
 * @tparam TKey The data-type that is used for keys in this database.
 */
template <typename TKey>
class DatabaseBackend {
 public:
  DatabaseBackend() = default;

  DatabaseBackend(const DatabaseBackend&) = delete;

  virtual ~DatabaseBackend() = default;

  DatabaseBackend& operator=(const DatabaseBackend&) = delete;

  /**
   * Returns that allows identifying the backend implementation.
   *
   * @return Pointer to the immutable name of this implementation.
   */
  virtual const char* get_name() const = 0;

  /**
   * Flag indicating whether the backing database is shared among all backend instances.
   *
   * @return Is this is a distributed database?
   */
  virtual bool is_shared() const = 0;

  virtual size_t max_capacity(const std::string& table_name) const = 0;

  /**
   * Check whether a set of keys exists in the database.
   *
   * @param table_name The name of the table to be queried (see also
   * paramter_server_base::make_tag_name).
   * @param num_keys Number of keys in \p keys .
   * @param keys Pointer to the keys.
   *
   * @return The number of keys that are actually present in the database. Will throw if an
   * recoverable error is encountered.
   */
  virtual size_t contains(const std::string& table_name, size_t num_keys, const TKey* keys) const;

  /**
   * Insert key/value pairs into the underlying database. For existing keys, the respective value is
   * overriden.
   *
   * @param table_name The name of the table to be queried (see also
   * paramter_server_base::make_tag_name).
   * @param num_pairs Number of \p keys and \p values .
   * @param keys Pointer to the keys.
   * @param values Pointer to the values.
   * @param value_size The size of each value in bytes.
   *
   * @return True if operation was successful.
   */
  virtual bool insert(const std::string& table_name, size_t num_pairs, const TKey* keys,
                      const char* values, size_t value_size) = 0;

  /**
   * Attempt to retrieve the stored value for a set of keys in the backing database (direct
   * indexing).
   *
   * @param table_name The name of the table to be queried (see also
   * paramter_server_base::make_tag_name).
   * @param num_keys Number of \p keys .
   * @param keys Pointer to the keys.
   * @param values Pointer to a preallocated memory area where the values will be stored.
   * @param value_size The size of each value in bytes.
   * @param missing_callback A function that is called for every key that was not present in this
   * database.
   *
   * @return The number of keys that were successfully retrieved from this database. Will throw if
   * an recoverable error is encountered.
   */
  virtual size_t fetch(const std::string& table_name, size_t num_keys, const TKey* keys,
                       char* values, size_t value_size, MissingKeyCallback& missing_callback) const;

  /**
   * Attempt to retrieve the stored value for a set of keys in the backing database. This variant
   * supports indirect indexing, to allow sparse lookup.
   *
   * @param table_name The name of the table to be queried (see also
   * paramter_server_base::make_tag_name).
   * @param num_indices Number of \p indices .
   * @param indices Pointer of indices in \p keys that need to be fetched.
   * @param keys Pointer to the key.
   * @param values Pointer to a preallocated memory area where the values will be stored. This
   * function operates sparse and will only update values that correspond to \p keys referenced by
   * \p indices .
   * @param value_size The size of each value in bytes.
   * @param missing_callback A function that is called for every key that was not present in this
   * database.
   *
   * @return The number of keys that were successfully retrieved from this database. Will throw if
   * an recoverable error is encountered.
   */
  virtual size_t fetch(const std::string& table_name, size_t num_indices, const size_t* indices,
                       const TKey* keys, char* values, size_t value_size,
                       MissingKeyCallback& missing_callback) const;

  /**
   * Attempt to remove a table and all associated values from the underlying database.
   *
   * @param table_name The name of the table to be queried (see also
   * paramter_server_base::make_tag_name).
   *
   * @return The number of keys/value pairs removed (not reliable in all implementations).
   */
  virtual size_t evict(const std::string& table_name) = 0;

  /**
   * Attempt to remove a set of keys from the underlying database table.
   *
   * @param table_name The name of the table to be queried (see also
   * paramter_server_base::make_tag_name).
   * @param num_keys Number of \p keys .
   * @param keys Pointer to the keys.
   *
   * @return The number of keys/value pairs removed (not reliable in all implementations).
   */
  virtual size_t evict(const std::string& table_name, size_t num_keys, const TKey* keys) = 0;
};

class DatabaseBackendError : std::exception {
 public:
  explicit DatabaseBackendError(const std::string& backend, size_t partition,
                                const std::string& what);

  DatabaseBackendError(const DatabaseBackendError&) = default;

  virtual ~DatabaseBackendError() = default;

  DatabaseBackendError& operator=(const DatabaseBackendError&) = default;

  virtual const std::string& backend() const noexcept { return backend_; }

  virtual size_t partition() const noexcept { return partition_; }

  virtual const char* what() const noexcept override { return what_.c_str(); }

  virtual std::string to_string() const;

 private:
  std::string backend_;
  size_t partition_;
  std::string what_;
};

template <typename TKey>
class VolatileBackend : public DatabaseBackend<TKey> {
 public:
  using TBase = DatabaseBackend<TKey>;

  VolatileBackend(size_t overflow_margin, DatabaseOverflowPolicy_t overflow_policy,
                  double overflow_resolution_target);

  VolatileBackend(const VolatileBackend&) = delete;

  virtual ~VolatileBackend() = default;

  VolatileBackend& operator=(const VolatileBackend&) = delete;

 protected:
  // Overflow-handling / pruning related parameters.
  const size_t overflow_margin_;
  const DatabaseOverflowPolicy_t overflow_policy_;
  const size_t overflow_resolution_target_;
};

template <typename TKey>
class PersistentBackend : public DatabaseBackend<TKey> {
 public:
  using TBase = DatabaseBackend<TKey>;

  PersistentBackend() = default;

  PersistentBackend(const PersistentBackend&) = delete;

  virtual ~PersistentBackend() = default;

  PersistentBackend& operator=(const PersistentBackend&) = delete;

  size_t max_capacity(const std::string& table_name) const override final {
    return std::numeric_limits<size_t>::max();
  }
};

}  // namespace HugeCTR