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

#include <cuda_profiler_api.h>
#include <gtest/gtest.h>

#include <base/debug/logger.hpp>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <hps/database_backend.hpp>
#include <hps/hash_map_backend.hpp>
#include <hps/hier_parameter_server_base.hpp>
#include <hps/mp_hash_map_backend.hpp>
#include <hps/redis_backend.hpp>
#include <hps/rocksdb_backend.hpp>
#include <memory>
#include <vector>

using namespace HugeCTR;
namespace {

template <typename T>
std::unique_ptr<DatabaseBackendBase<T>> make_db(const DatabaseType_t database_type) {
  switch (database_type) {
    case DatabaseType_t::HashMap: {
      HashMapBackendParams params;
      params.num_partitions = 16;
      return std::make_unique<HashMapBackend<T>>(params);
    } break;

    case DatabaseType_t::RedisCluster: {
      RedisClusterBackendParams params;
      params.address = "127.0.0.1:7000,127.0.0.1:7001,127.0.0.1:7002";
      return std::make_unique<RedisClusterBackend<T>>(params);
    } break;

    case DatabaseType_t::RocksDB: {
      RocksDBBackendParams params;
      params.path = "/hugectr/Test_Data/rockdb";
      return std::make_unique<RocksDBBackend<T>>(params);
    } break;

    default:
      HCTR_DIE("Unsupported database type!");
      return nullptr;
  }
}

template <typename Key>
void db_backend_insert_fetch_test(const DatabaseType_t database_type) {
  std::unique_ptr<DatabaseBackendBase<Key>> db{make_db<Key>(database_type)};

  const std::string& tag{HierParameterServerBase::make_tag_name("insert_fetch", "test")};

  // Insert some KV pairs (single + vector mode).
  {
    std::vector<Key> keys(10);
    std::vector<double> values(keys.size());

    Key k{0};
    for (; k < 50; ++k) {
      keys[0] = k;
      values[0] = k * k;
      db->insert(tag, 1, keys.data(), reinterpret_cast<char*>(values.data()), sizeof(double),
                 sizeof(double));
    }

    while (k < 100) {
      for (size_t i{0}; i < keys.size(); ++i, ++k) {
        keys[i] = k;
        values[i] = k * k;
      }
      db->insert(tag, keys.size(), keys.data(), reinterpret_cast<char*>(values.data()),
                 sizeof(double), sizeof(double));
    }
  }

  // Try to fetch values (single + vector mode).
  {
    std::vector<Key> keys{3, 41, 69, 16, 99, 0, 22, 5, 40};
    std::vector<double> values(keys.size());

    for (const Key& k : keys) {
      keys[0] = k;
      db->fetch(tag, 1, keys.data(), reinterpret_cast<char*>(values.data()), sizeof(double),
                [&](size_t index) { FAIL(); });
      EXPECT_DOUBLE_EQ(values[0], k * k);
    }

    std::vector<double> expected_values;
    expected_values.reserve(keys.size());
    std::transform(keys.begin(), keys.end(), std::back_inserter(expected_values),
                   [](const Key k) -> double { return k * k; });

    db->fetch(tag, keys.size(), keys.data(), reinterpret_cast<char*>(values.data()), sizeof(double),
              [&](size_t index) { FAIL(); });
    for (size_t i{0}; i < values.size(); ++i) {
      EXPECT_DOUBLE_EQ(values[i], expected_values[i]);
    }
  }
}

template <typename Key>
void db_backend_multi_evict_test(const DatabaseType_t database_type) {
  std::unique_ptr<DatabaseBackendBase<Key>> db{make_db<Key>(database_type)};

  size_t num_tables[3];

  // Insert a couple of model parameters.
  for (size_t i{0}; i < 3; ++i) {
    std::ostringstream mdl;
    mdl << "mdl" << i;

    for (size_t j{0}; j < 3; ++j) {
      std::ostringstream tbl;
      tbl << "tbl" << j;

      const std::string& tag{HierParameterServerBase::make_tag_name(mdl.str(), tbl.str())};
      for (Key k{0}; k < 50; ++k) {
        const double kk{static_cast<double>(k * k)};
        db->insert(tag, 1, &k, reinterpret_cast<const char*>(&kk), sizeof(double), sizeof(double));
      }
    }
    num_tables[i] = 3;
  }

  // Checks number of tables.
  for (size_t i{0}; i < 3; ++i) {
    std::ostringstream mdl;
    mdl << "mdl" << i;

    const std::vector<std::string>& tables{db->find_tables(mdl.str())};
    for (const auto& table : tables) {
      std::cout << "Found table " << table << std::endl;
    }
    EXPECT_EQ(tables.size(), num_tables[i]);
  }

  // Delete 2nd table.
  {
    const std::vector<std::string>& tables{db->find_tables("mdl1")};
    db->evict(tables);
    num_tables[1] = 0;
  }

  // Checks number of tables.
  for (size_t i{0}; i < 3; ++i) {
    std::ostringstream mdl;
    mdl << "mdl" << i;

    const std::vector<std::string>& tables{db->find_tables(mdl.str())};
    std::cout << "[ " << i << " ] Found table " << tables.size() << std::endl;
    EXPECT_EQ(tables.size(), num_tables[i]);
  }
}

template <typename Key>
void db_backend_dump_test(DatabaseType_t database_type) {
  std::unique_ptr<DatabaseBackendBase<Key>> db{make_db<Key>(database_type)};

  // Populate a dummy table.
  const std::string tag0{HierParameterServerBase::make_tag_name("mdl", "tbl0")};
  {
    std::vector<Key> keys;
    std::vector<double> values;
    for (Key k{0}; k < 10; ++k) {
      keys.push_back(k);
      values.push_back(std::cos(static_cast<double>(k)));
    }
    db->insert(tag0, keys.size(), keys.data(), reinterpret_cast<const char*>(values.data()),
               sizeof(double), sizeof(double));
  }

  // Dump to disk as bin and sst.
  db->dump(tag0, "tbl0.bin");
  db->dump(tag0, "tbl0.sst");
  db->evict(tag0);

  // Reload binary dump.
  const std::string& tag1{HierParameterServerBase::make_tag_name("mdl", "tbl1")};
  db->load_dump(tag1, "tbl0.bin");
  std::cout << "tbl1 size " << db->size(tag1) << std::endl;

  for (Key k{0}; k < 10; ++k) {
    // std::cout << "key " << k << std::endl;
    double v;
    db->fetch(tag1, 1, &k, reinterpret_cast<char*>(&v), sizeof(double),
              [&](size_t index) { FAIL(); });

    EXPECT_EQ(v, std::cos(static_cast<double>(k)));
  }
  db->evict(tag1);

  // Reload binary sst dump.
  const std::string& tag2{HierParameterServerBase::make_tag_name("mdl", "tbl2")};
  db->load_dump(tag2, "tbl0.sst");
  std::cout << "tbl2 size " << db->size(tag2) << std::endl;

  for (Key k{0}; k < 10; ++k) {
    // std::cout << "key " << k << std::endl;
    double v;
    db->fetch(tag2, 1, &k, reinterpret_cast<char*>(&v), sizeof(double),
              [&](size_t index) { FAIL(); });

    EXPECT_EQ(v, std::cos(static_cast<double>(k)));
  }
  db->evict(tag2);

  // Special check. See if we can load a hashmap dump into RocksDB.
  if (database_type == DatabaseType_t::RocksDB) {
    {
      auto db2{make_db<Key>(DatabaseType_t::HashMap)};
      db2->load_dump(tag1, "tbl0.bin");
      db2->dump(tag1, "tbl2.sst");
    }

    const std::string& tag3{HierParameterServerBase::make_tag_name("mdl", "tbl2")};
    db->load_dump(tag3, "tbl2.sst");
    std::cout << "tag3 size " << db->size(tag3) << std::endl;

    for (Key k{0}; k < 10; ++k) {
      // std::cout << "key " << k << std::endl;
      double v;
      db->fetch(tag3, 1, &k, reinterpret_cast<char*>(&v), sizeof(double),
                [&](size_t index) { FAIL(); });

      EXPECT_EQ(v, std::cos(static_cast<double>(k)));
    }
    db->evict(tag3);
  }
}

}  // namespace

TEST(db_backend_insert_fetch_test, HashMap) {
  db_backend_insert_fetch_test<long long>(DatabaseType_t::HashMap);
}
TEST(db_backend_insert_fetch_test, Redis) {
  db_backend_insert_fetch_test<long long>(DatabaseType_t::RedisCluster);
}
TEST(db_backend_insert_fetch_test, Rocksdb) {
  db_backend_insert_fetch_test<long long>(DatabaseType_t::RocksDB);
}

TEST(db_backend_multi_evict, HashMap) {
  db_backend_multi_evict_test<long long>(DatabaseType_t::HashMap);
}
TEST(db_backend_multi_evict, Redis) {
  db_backend_multi_evict_test<long long>(DatabaseType_t::RedisCluster);
}
TEST(db_backend_multi_evict, Rocksdb) {
  db_backend_multi_evict_test<long long>(DatabaseType_t::RocksDB);
}

TEST(db_backend_dump_load, HashMap) { db_backend_dump_test<long long>(DatabaseType_t::HashMap); }
TEST(db_backend_dump_load, RedisCluster) {
  db_backend_dump_test<long long>(DatabaseType_t::RedisCluster);
}
TEST(db_backend_dump_load, RocksDB) { db_backend_dump_test<long long>(DatabaseType_t::RocksDB); }
