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

#include <argparse/argparse.hpp>
#include <base/debug/logger.hpp>
#include <core/memory.hpp>
#include <hps/hash_map_backend.hpp>
#include <hps/hier_parameter_server_base.hpp>
#include <hps/mp_hash_map_backend.hpp>
#include <hps/redis_backend.hpp>
#include <hps/rocksdb_backend.hpp>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace HugeCTR;

typedef long long Key;

int main(int argc, char** argv) {
  argparse::ArgumentParser args;

  args.add_argument("--kafka_broker")
      .help("Kafka broker.")
      .default_value<std::string>("localhost:9092");

  args.add_argument("--model").help("Model name.").default_value<std::string>("mdl");
  args.add_argument("--table").help("Table name.").default_value<std::string>("tab1");

  args.add_argument("--db_type").help("Amount of values to prefill.").required();

  // Options of the test itself.
  args.add_argument("--no_test_insert_evict")
      .help("Disables insert / evict tests.")
      .default_value(false)
      .implicit_value(true);

  args.add_argument("--no_test_upsert")
      .help("Disables upsert test.")
      .default_value(false)
      .implicit_value(true);

  args.add_argument("--no_test_fetch")
      .help("Disables fetch test.")
      .default_value(false)
      .implicit_value(true);

  args.add_argument("--seed")
      .help("Seed for the random number generator.")
      .default_value<uint64_t>(4711)
      .scan<'u', uint64_t>();

  // HM parameters.
  args.add_argument("--hm_parts")
      .help("Number of threads for HashMap.")
      .default_value<size_t>(32)
      .scan<'u', size_t>();

  args.add_argument("--hm_alloc_rate")
      .help("Memory pool allocation rate.")
      .default_value<size_t>(256L * 1024 * 1024)
      .scan<'u', size_t>();

  args.add_argument("--hm_sm_size")
      .help("Maximum shared memory size.")
      .default_value<size_t>(256L * 1024 * 1024 * 1024)
      .scan<'u', size_t>();

  args.add_argument("--hm_batch_size")
      .help("Batch size for hashmap.")
      .default_value<size_t>(8L * 1024 * 1024)
      .scan<'u', size_t>();

  // Redis parameters.
  args.add_argument("--re_address")
      .help("Redis server address.")
      .default_value<std::string>("localhost:7000");

  args.add_argument("--re_parts")
      .help("Number of partitions for Redis.")
      .default_value<size_t>(15)
      .scan<'u', size_t>();

  args.add_argument("--re_connections")
      .help("Number of connections per Redis node.")
      .default_value<size_t>(5)
      .scan<'u', size_t>();

  args.add_argument("--re_batch_size")
      .help("Batch size for Redis.")
      .default_value<size_t>(256 * 1024)
      .scan<'u', size_t>();  // Max: 512 * 1024 - 1 (limit of OSS Redis)

  // RocksDB parameters.
  args.add_argument("--ro_path")
      .help("RocksDB database path.")
      .default_value<std::string>("/tmp/rocksdb");

  args.add_argument("--ro_threads")
      .help("Number of threads for RocksDB.")
      .default_value<size_t>(32)
      .scan<'u', size_t>();

  args.add_argument("--ro_batch_size")
      .help("Batch size for RocksDB.")
      .default_value<size_t>(1024 * 1024)
      .scan<'u', size_t>();

  // Other parmeters.
  args.add_argument("--emb_size")
      .help("Size of one embedding.")
      .default_value<size_t>(128)
      .scan<'u', size_t>();

  args.add_argument("--fill_amount")
      .help("Amount of values to fill.")
      .default_value<size_t>(250L * 1000 * 1000)
      .scan<'u', size_t>();

  args.add_argument("--fill_burst")
      .help("Batch size during fill.")
      .default_value<size_t>(50L * 1000 * 1000)
      .scan<'u', size_t>();

  args.add_argument("--query_amount")
      .help("Amount of values to query per burst.")
      .default_value<size_t>(50L * 1000 * 1000)
      .scan<'u', size_t>();

  args.add_argument("--query_repeat")
      .help("Amount of query burst repeats.")
      .default_value<size_t>(10)
      .scan<'u', size_t>();

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cout << args;
    return 1;
  }

  const auto kafka_broker = args.get<std::string>("--kafka_broker");
  const auto model_name = args.get<std::string>("--model");
  const auto table_name = args.get<std::string>("--table");
  const auto db_type = args.get<std::string>("--db_type");
  // Options of the test itself.
  const auto no_test_insert_evict = args.get<bool>("--no_test_insert_evict");
  const auto no_test_upsert = args.get<bool>("--no_test_upsert");
  const auto no_test_fetch = args.get<bool>("--no_test_fetch");
  const auto seed = args.get<uint64_t>("--seed");
  // HM parameters.
  const auto hm_parts = args.get<size_t>("--hm_parts");
  const auto hm_alloc_rate = args.get<size_t>("--hm_alloc_rate");
  const auto hm_sm_size = args.get<size_t>("--hm_sm_size");
  const auto hm_batch_size = args.get<size_t>("--hm_batch_size");
  // Redis parameters.
  const auto re_address = args.get<std::string>("--re_address");
  const auto re_parts = args.get<size_t>("--re_parts");
  const auto re_connections = args.get<size_t>("--re_connections");
  const auto re_batch_size = args.get<size_t>("--re_batch_size");
  // RocksDB parameters.
  const auto ro_path = args.get<std::string>("--ro_path");
  const auto ro_threads = args.get<size_t>("--ro_threads");
  const auto ro_batch_size = args.get<size_t>("--ro_batch_size");
  // Other parameters.
  const auto emb_size = args.get<size_t>("--emb_size");
  const auto fill_amount = args.get<size_t>("--fill_amount");
  const auto fill_burst = args.get<size_t>("--fill_burst");
  const auto query_amount = args.get<size_t>("--query_amount");
  const auto query_repeat = args.get<size_t>("--query_repeat");

  std::cout << "Options: " << std::endl
            << "  -----------------------------" << std::endl
            << "  no_test_insert_evict = " << no_test_insert_evict << std::endl
            << "  no_test_upsert       = " << no_test_upsert << std::endl
            << "  no_test_fetch        = " << no_test_fetch << std::endl
            << "  seed                 = " << seed << std::endl
            << "  -----------------------------" << std::endl
            << "  broker = " << kafka_broker << std::endl
            << "  -----------------------------" << std::endl
            << "  model = " << model_name << std::endl
            << "  table = " << table_name << std::endl
            << "  -----------------------------" << std::endl
            << "  db_type        = " << db_type << std::endl
            << "  hm_parts       = " << hm_parts << std::endl
            << "  hm_alloc_rate  = " << hm_alloc_rate << std::endl
            << "  hm_sm_size     = " << hm_sm_size << std::endl
            << "  hm_batch_size  = " << hm_batch_size << std::endl
            << std::endl
            << "  re_address     = " << re_address << std::endl
            << "  re_parts       = " << re_parts << std::endl
            << "  re_connections = " << re_connections << std::endl
            << "  re_batch_size  = " << re_batch_size << std::endl
            << std::endl
            << "  ro_path        = " << ro_path << std::endl
            << "  ro_threads     = " << ro_threads << std::endl
            << "  ro_batch_size  = " << ro_batch_size << std::endl
            << "  -----------------------------" << std::endl
            << "  emb_size     = " << emb_size << " x " << sizeof(float) << std::endl
            << "  fill_amount  = " << fill_amount << std::endl
            << "  fill_burst   = " << fill_burst << std::endl
            << "  query_amount = " << query_amount << std::endl
            << "  query_repeat = " << query_repeat << std::endl
            << "  -----------------------------" << std::endl;

  const std::string tag_name = HierParameterServerBase::make_tag_name(model_name, table_name);

  std::unique_ptr<DatabaseBackendBase<Key>> db;
  if (db_type == "hashmap") {
    HashMapBackendParams params;
    params.max_batch_size = hm_batch_size;
    params.num_partitions = hm_parts;
    params.allocation_rate = hm_alloc_rate;
    db = std::make_unique<HashMapBackend<Key>>(params);
  } else if (db_type == "mp_hashmap") {
    MultiProcessHashMapBackendParams params;
    params.max_batch_size = hm_batch_size;
    params.num_partitions = hm_parts;
    params.allocation_rate = hm_alloc_rate;
    params.shared_memory_size = hm_sm_size;
    db = std::make_unique<MultiProcessHashMapBackend<Key>>(params);
  } else if (db_type == "redis") {
    RedisClusterBackendParams params;
    params.max_batch_size = re_batch_size;
    params.num_partitions = re_parts;
    params.address = re_address;
    params.num_node_connections = re_connections;
    db = std::make_unique<RedisClusterBackend<Key>>(params);
  } else if (db_type == "rocksdb") {
    RocksDBBackendParams params;
    params.max_batch_size = ro_batch_size;
    params.path = ro_path;
    params.num_threads = ro_threads;
    db = std::make_unique<RocksDBBackend<Key>>(params);
  } else {
    HCTR_DIE("Invalid db_type!");
  }

  const size_t kv_size = sizeof(Key) + emb_size * sizeof(float);

  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);

  try {
    HCTR_LOG_S(INFO, WORLD) << "Create some random values..." << std::endl;
    std::vector<float> in_values(fill_burst * emb_size);
    {
      size_t i = 0;
      for (; i < 128; ++i) {
        in_values[i] = val_dist(gen);
      }
      for (; i < in_values.size(); ++i) {
        in_values[i] = in_values[i % 128];
      }
    }

    std::vector<Key> keys(std::max(fill_burst, query_amount));
    // std::vector<float> out_values(query_amount * emb_size);
    std::vector<float, AlignedAllocator<float>> out_values(query_amount * emb_size);

    HCTR_LOG_S(INFO, WORLD) << "Filling database..." << std::endl;
    for (size_t i = 0; i < fill_amount;) {
      const size_t burst_length = std::min(fill_amount - i, keys.size());

      for (size_t j = 0; j < burst_length; ++i, ++j) {
        keys[j] = i;
      }

      // Insert / evict.
      if (i == burst_length || db_type != "mp_hashmap") {
        for (size_t k = 0; !no_test_insert_evict && k < query_repeat; ++k) {
          {
            const auto t0 = std::chrono::high_resolution_clock::now();

            db->insert(tag_name, burst_length, keys.data(),
                       reinterpret_cast<const char*>(in_values.data()), emb_size * sizeof(float),
                       emb_size * sizeof(float));

            const auto t1 = std::chrono::high_resolution_clock::now();
            const auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);

            HCTR_LOG_S(INFO, WORLD)
                << "DB size = " << db->size(tag_name) << ", k = " << k
                << ", insert time = " << dur.count() << " us, " << std::fixed
                << std::setprecision(3) << (kv_size * query_amount / 1000.0 / dur.count())
                << " GB/s" << std::endl;
          }

          // Temporary hack to allow checking insert. Otherwise it is too slow with mp_hashmap.
          if (db_type == "mp_hashmap") {
            const auto t0 = std::chrono::high_resolution_clock::now();

            db->evict(tag_name);

            const auto t1 = std::chrono::high_resolution_clock::now();
            const auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
            HCTR_LOG_S(INFO, WORLD)
                << "DB size = " << db->size(tag_name) << ", k = " << k
                << ", evict time = " << dur.count() << " us, " << std::fixed << std::setprecision(3)
                << (kv_size * query_amount / 1000.0 / dur.count()) << " GB/s" << std::endl;
          } else {
            const auto t0 = std::chrono::high_resolution_clock::now();

            db->evict(tag_name, burst_length, keys.data());

            const auto t1 = std::chrono::high_resolution_clock::now();
            const auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
            HCTR_LOG_S(INFO, WORLD)
                << "DB size = " << db->size(tag_name) << ", k = " << k
                << ", evict time = " << dur.count() << " us, " << std::fixed << std::setprecision(3)
                << (kv_size * query_amount / 1000.0 / dur.count()) << " GB/s" << std::endl;
          }
        }
      }

      // Upsert
      db->insert(tag_name, burst_length, keys.data(),
                 reinterpret_cast<const char*>(in_values.data()), emb_size * sizeof(float),
                 emb_size * sizeof(float));

      // Replace
      for (size_t k = 0; !no_test_upsert && k < query_repeat; ++k) {
        const auto t0 = std::chrono::high_resolution_clock::now();

        db->insert(tag_name, burst_length, keys.data(),
                   reinterpret_cast<const char*>(in_values.data()), emb_size * sizeof(float),
                   emb_size * sizeof(float));

        const auto t1 = std::chrono::high_resolution_clock::now();
        const auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
        HCTR_LOG_S(INFO, WORLD) << "DB size = " << db->size(tag_name) << ", k = " << k
                                << ", replace time = " << dur.count() << " us, " << std::fixed
                                << std::setprecision(3)
                                << (kv_size * query_amount / 1000.0 / dur.count()) << " GB/s"
                                << std::endl;
      }

      // Query.
      std::uniform_int_distribution<size_t> key_dist(0, i - 1);
      for (size_t k = 0; !no_test_fetch && k < query_repeat; ++k) {
        for (size_t j = 0; j < keys.size(); ++j) {
          keys[j] = key_dist(gen);
        }

        {
          const auto t0 = std::chrono::high_resolution_clock::now();
          const size_t num_hits = db->fetch(tag_name, query_amount, keys.data(),
                                            reinterpret_cast<char*>(out_values.data()),
                                            emb_size * sizeof(float), [&](const size_t) {});
          const auto t1 = std::chrono::high_resolution_clock::now();
          const auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
          HCTR_LOG_S(INFO, WORLD) << "DB size = " << db->size(tag_name) << ", k = " << k
                                  << ", num hits = " << num_hits
                                  << ", num misses = " << query_amount - num_hits
                                  << ", query time = " << dur.count() << " us, " << std::fixed
                                  << std::setprecision(3)
                                  << (kv_size * query_amount / 1000.0 / dur.count()) << " GB/s"
                                  << std::endl;
        }
      }
    }
  } catch (const DatabaseBackendError& error) {
    HCTR_LOG_S(ERROR, WORLD) << "Partition #" << error.partition() << ": " << error.what()
                             << std::endl;
  } catch (const std::exception& error) {
    HCTR_LOG_S(ERROR, WORLD) << "Error: " << error.what() << std::endl;
  }

  HCTR_LOG_S(INFO, WORLD) << "Destroying database..." << std::endl;
  db.reset();
}
