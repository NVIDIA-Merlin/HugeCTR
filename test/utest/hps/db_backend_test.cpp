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

#include <cassert>
#include <filesystem>
#include <fstream>
#include <hps/database_backend.hpp>
#include <hps/hash_map_backend.hpp>
#include <hps/hier_parameter_server_base.hpp>
#include <hps/redis_backend.hpp>
#include <hps/rocksdb_backend.hpp>
#include <memory>
#include <vector>

using namespace HugeCTR;
namespace {

template <typename TKey>
void db_backend_multi_evict(DatabaseType_t database_t) {
  std::unique_ptr<DatabaseBackend<TKey>> db;
  switch (database_t) {
    case DatabaseType_t::ParallelHashMap:
      db = std::make_unique<HashMapBackend<TKey>>();
      break;
    case DatabaseType_t::RedisCluster:
      db = std::make_unique<RedisClusterBackend<TKey>>(
          "127.0.0.1:7000,127.0.0.1:7001,127.0.0.1:7002");
      break;
    case DatabaseType_t::RocksDB:
      db = std::make_unique<RocksDBBackend<TKey>>("/hugectr/Test_Data/rockdb");
      break;
    default:
      break;
  }

  size_t num_tables[3];

  // Insert a couple of model parameters.
  for (size_t i = 0; i < 3; i++) {
    std::ostringstream mdl;
    mdl << "mdl" << i;

    for (size_t j = 0; j < 3; j++) {
      std::ostringstream tbl;
      tbl << "tbl" << j;

      const std::string& tag = HierParameterServerBase::make_tag_name(mdl.str(), tbl.str());
      for (TKey k = 0; k < 50; k++) {
        const double kk = k * k;
        db->insert(tag, 1, &k, reinterpret_cast<const char*>(&kk), sizeof(double));
      }
    }
    num_tables[i] = 3;
  }

  // Checks number of tables.
  for (size_t i = 0; i < 3; i++) {
    std::ostringstream mdl;
    mdl << "mdl" << i;

    const std::vector<std::string>& tables = db->find_tables(mdl.str());
    for (const auto& table : tables) {
      std::cout << "Found table " << table << std::endl;
    }
    EXPECT_EQ(tables.size(), num_tables[i]);
  }

  // Delete 2nd table.
  {
    const std::vector<std::string>& tables = db->find_tables("mdl1");
    db->evict(tables);
    num_tables[1] = 0;
  }

  // Checks number of tables.
  for (size_t i = 0; i < 3; i++) {
    std::ostringstream mdl;
    mdl << "mdl" << i;

    const std::vector<std::string>& tables = db->find_tables(mdl.str());
    std::cout << "[ " << i << " ] Found table " << tables.size() << std::endl;
    EXPECT_EQ(tables.size(), num_tables[i]);
  }
}

}  // namespace

TEST(db_backend_multi_evict, HashMap) {
  db_backend_multi_evict<long long>(DatabaseType_t::ParallelHashMap);
}
TEST(db_backend_multi_evict, Rocksdb) {
  db_backend_multi_evict<long long>(DatabaseType_t::RocksDB);
}
TEST(db_backend_multi_evict, Redis) {
  db_backend_multi_evict<long long>(DatabaseType_t::RedisCluster);
}
