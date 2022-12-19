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

#include <rocksdb/sst_file_writer.h>

#include <algorithm>
#include <chrono>
#include <common.hpp>
#include <fstream>
#include <functional>
#include <hps/inference_utils.hpp>
#include <string>
#include <thread_pool.hpp>
#include <vector>

namespace HugeCTR {

// TODO: Remove me!
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wconversion"

/**
 * Format of callback that is invoked by \p fetch methods of the \p DatabaseBackend if a key does
 * not exist.
 */
using DatabaseMissCallback = std::function<void(size_t)>;
using DatabaseHitCallback = std::function<void(size_t, const char*, uint32_t)>;

enum class DatabaseTableDumpFormat_t {
  Automatic = 0,  // Try to deduce the storage format from the provided path.
  Raw,            // Use raw storage format.
  SST,            // Write data as an "Static Sorted Table" file.
};

/**
 * Base class for database backends. Implementations that inherit from this should override all
 * public members.
 *
 * @tparam Key The data-type that is used for keys in this database.
 */
template <typename Key>
class DatabaseBackendBase {
 public:
  DatabaseBackendBase() = delete;
  DatabaseBackendBase(size_t max_set_batch_size);
  DISALLOW_COPY_AND_MOVE(DatabaseBackendBase);

  virtual ~DatabaseBackendBase() = default;

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

  /**
   * @return Maximum capacity of the table (if every partition would be fully populated).
   */
  virtual size_t capacity(const std::string& table_name) const = 0;

  /**
   * Determine the number of entries currently stored in the table. This might be the actual value
   * current or past (if lock-free parallel updates are permitted) value, or just a rough
   * approximation.
   *
   * @return Current amount of entries stored in the table (or -1 if unable to determine).
   */
  virtual size_t size(const std::string& table_name) const = 0;

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
  virtual size_t contains(const std::string& table_name, size_t num_keys, const Key* keys,
                          const std::chrono::nanoseconds& time_budget) const = 0;

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
  virtual bool insert(const std::string& table_name, size_t num_pairs, const Key* keys,
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
   * @param on_miss A function that is called for every key that was not present in this
   * database.
   * @param time_budget A budget given to the function to do its work. This is a soft-limit. The
   * function will try to complete in time.
   *
   * @return The number of keys that were successfully retrieved from this database. Will throw if
   * an recoverable error is encountered.
   */
  virtual size_t fetch(const std::string& table_name, size_t num_keys, const Key* keys,
                       const DatabaseHitCallback& on_hit, const DatabaseMissCallback& on_miss,
                       const std::chrono::nanoseconds& time_budget) = 0;

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
   * @param on_miss A function that is called for every key that was not present in this
   * database.
   * @param time_budget A budget given to the function to do its work. This is a soft-limit. The
   * function will try to complete in time.
   *
   * @return The number of keys that were successfully retrieved from this database. Will throw if
   * an recoverable error is encountered.
   */
  virtual size_t fetch(const std::string& table_name, size_t num_indices, const size_t* indices,
                       const Key* keys, const DatabaseHitCallback& on_hit,
                       const DatabaseMissCallback& on_miss,
                       const std::chrono::nanoseconds& time_budget) = 0;

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
   * Same as calling evict for all provided table names.
   *
   * @param table_names List of table names.
   *
   * @return Total number of keys/value pairs removed.
   */
  size_t evict(const std::vector<std::string>& table_names);

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
  virtual size_t evict(const std::string& table_name, size_t num_keys, const Key* keys) = 0;

  /**
   * Find all tables belonging to a specific model.
   *
   * @param model_name The name of the model.
   *
   * @return List containing the names of the tables.
   */
  virtual std::vector<std::string> find_tables(const std::string& model_name) = 0;

  /*
   * Dumps the contents of an entire table to a file.
   *
   * @param table_name The name of the table to be dumped.
   * @param path File system path under which the dumped data should be stored.
   * @param format Dump format.
   */
  void dump(const std::string& table_name, const std::string& path,
            DatabaseTableDumpFormat_t format = DatabaseTableDumpFormat_t::Automatic);

  virtual void dump_bin(const std::string& table_name, std::ofstream& file) = 0;

  virtual void dump_sst(const std::string& table_name, rocksdb::SstFileWriter& file) = 0;

  /**
   * Loads the contents of a dump file into a table.
   *
   * @param table_name The destination table into which to insert the data.
   * @param path File system path under which the dumped data should be stored.
   */
  virtual void load_dump(const std::string& table_name, const std::string& path);

  virtual void load_dump_bin(const std::string& table_name, const std::string& path);

  virtual void load_dump_sst(const std::string& table_name, const std::string& path);

 private:
  const size_t max_set_batch_size_;  // Temporary, until find a better solution.
};

struct DatabaseBackendParams {
  size_t max_get_batch_size{64L * 1024};  // Maximum number of key/value pairs per read transaction.
  size_t max_set_batch_size{64L *
                            1024};  // Maximum number of key/value pairs per write transaction.
};

template <typename Key, typename Params>
class DatabaseBackend : public DatabaseBackendBase<Key> {
 public:
  using Base = DatabaseBackendBase<Key>;

  DatabaseBackend() = delete;
  DISALLOW_COPY_AND_MOVE(DatabaseBackend);
  DatabaseBackend(const Params& params) : Base(params.max_set_batch_size), params_{params} {}

  virtual ~DatabaseBackend() = default;

 protected:
  const Params params_;
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

struct VolatileBackendParams : public DatabaseBackendParams {
  size_t num_partitions{
      16};  // The number of parallel partitions. Determines the maximum degree of parallelization.
            // For Redis, this equates to the amount of separate storage partitions. For achieving
            // the best performance, this should be signficantly higher than the number of cluster
            // nodes! We use modulo-N to assign partitions. Hence, you must not change this value
            // after writing the first data to a table.

  size_t overflow_margin{std::numeric_limits<size_t>::max()};  // Margin at which further inserts
                                                               // will trigger overflow handling.
  DatabaseOverflowPolicy_t overflow_policy{
      DatabaseOverflowPolicy_t::EvictOldest};  // Policy to use in case an overflow has been
                                               // detected.
  double overflow_resolution_target{0.8};  // Target margin after applying overflow handling policy.

  inline size_t overflow_resolution_margin() const {
    const size_t margin = static_cast<size_t>(
        static_cast<double>(overflow_margin) * overflow_resolution_target + 0.5);
    HCTR_CHECK(margin <= overflow_margin);
    return margin;
  }
};

template <typename Key, typename Params>
class VolatileBackend : public DatabaseBackend<Key, Params> {
 public:
  using Base = DatabaseBackend<Key, Params>;

  VolatileBackend() = delete;
  DISALLOW_COPY_AND_MOVE(VolatileBackend);
  VolatileBackend(const Params& params)
      : Base(params), overflow_resolution_margin_{params.overflow_resolution_margin()} {}

  virtual ~VolatileBackend() = default;

  size_t capacity(const std::string& table_name) const override final {
    const size_t part_margin = this->params_.overflow_margin;
    const size_t total_margin = part_margin * this->params_.num_partitions;
    return std::max(total_margin, part_margin);
  }

 protected:
  const size_t overflow_resolution_margin_;
};

struct PersistentBackendParams : public DatabaseBackendParams {};

template <typename Key, typename Params>
class PersistentBackend : public DatabaseBackend<Key, Params> {
 public:
  using Base = DatabaseBackend<Key, Params>;

  PersistentBackend() = delete;
  DISALLOW_COPY_AND_MOVE(PersistentBackend);
  PersistentBackend(const Params& params) : Base(params) {}

  virtual ~PersistentBackend() = default;

  size_t capacity(const std::string& table_name) const override final {
    return std::numeric_limits<size_t>::max();
  }
};

#ifdef HCTR_ROCKSDB_CHECK
#error HCTR_ROCKSDB_CHECK is already defined. This could lead to unpredictable behavior!
#else
#define HCTR_ROCKSDB_CHECK(EXPR)                                                  \
  do {                                                                            \
    const rocksdb::Status status = (EXPR);                                        \
    HCTR_CHECK_HINT(status.ok(), "RocksDB error: %s", status.ToString().c_str()); \
  } while (0)
#endif

// TODO: Remove me!
#pragma GCC diagnostic pop

}  // namespace HugeCTR