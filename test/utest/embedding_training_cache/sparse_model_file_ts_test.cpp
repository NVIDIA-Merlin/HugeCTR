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

#include <gtest/gtest.h>

#include <algorithm>
#include <embedding_training_cache/hmem_cache/sparse_model_file_ts.hpp>
#include <random>
#include <type_traits>
#include <utest/embedding_training_cache/etc_test_utils.hpp>

using namespace HugeCTR;
using namespace etc_test;

namespace {

const char* prefix = "./embedding_training_cache_test_data/tmp_";
const char* file_list_name_train = "file_list_train.txt";
const char* file_list_name_eval = "file_list_eval.txt";
const char* snapshot_src_file = "distributed_snapshot_src";
const char* snapshot_dst_file = "distributed_snapshot_dst";
const char* snapshot_bkp_file_unsigned = "distributed_snapshot_unsigned";
const char* snapshot_bkp_file_longlong = "distributed_snapshot_longlong";
// const char* keyset_file_name = "keyset_file.bin";

const int batchsize = 4096;
const long long label_dim = 1;
const long long dense_dim = 0;
const int slot_num = 128;
const int max_nnz_per_slot = 1;
const int max_feature_num = max_nnz_per_slot * slot_num;
const long long vocabulary_size = 100000;
const int emb_vec_size = 64;
const int combiner = 0;
const float scaler = 1.0f;
const int num_workers = 1;
const int num_files = 1;

const Check_t check = Check_t::Sum;
const Update_t update_type = Update_t::Local;

// const int batch_num_train = 10;
const int batch_num_eval = 1;

template <typename TypeKey>
void compare_with_ssd(std::vector<std::string>& data_files, std::vector<TypeKey>& keys,
                      bool use_slot_id, std::vector<size_t>& slot_ids,
                      std::vector<std::vector<float>>& data_vecs) {
  if (std::is_same<TypeKey, long long>::value) {
    std::string src_data_file(std::string(snapshot_dst_file) + "/key");
    auto file_size{keys.size() * sizeof(size_t)};
    // ASSERT_EQ(file_size, std::filesystem::file_size(src_data_file));
    std::vector<char> data_vec(file_size);
    std::ifstream ifs(src_data_file);
    ifs.read(data_vec.data(), file_size);
    HCTR_LOG(INFO, WORLD, "check key\n");
    ASSERT_TRUE(test::compare_array_approx<char>(reinterpret_cast<char*>(keys.data()),
                                                 data_vec.data(), file_size, 0));
    HCTR_LOG(INFO, WORLD, "Done!\n");
  }
  if (use_slot_id) {
    std::string src_data_file(std::string(snapshot_dst_file) + "/slot_id");
    auto file_size{keys.size() * sizeof(size_t)};
    // ASSERT_EQ(file_size, std::filesystem::file_size(src_data_file));
    std::vector<char> data_vec(file_size);
    std::ifstream ifs(src_data_file);
    ifs.read(data_vec.data(), file_size);
    HCTR_LOG(INFO, WORLD, "check slot_id\n");
    ASSERT_TRUE(test::compare_array_approx<char>(reinterpret_cast<char*>(slot_ids.data()),
                                                 data_vec.data(), file_size, 0));
    HCTR_LOG(INFO, WORLD, "Done!\n");
  }
  size_t counter{0};
  for (const auto& data_file : data_files) {
    std::string src_data_file(std::string(snapshot_dst_file) + "/" + data_file);
    auto file_size{keys.size() * emb_vec_size * sizeof(float)};
    // ASSERT_EQ(file_size, std::filesystem::file_size(src_data_file));
    std::vector<char> data_vec(file_size);
    std::ifstream ifs(src_data_file);
    ifs.read(data_vec.data(), file_size);
    HCTR_LOG_S(INFO, WORLD) << "check " << data_file << std::endl;
    ASSERT_TRUE(test::compare_array_approx<char>(reinterpret_cast<char*>(data_vecs[counter].data()),
                                                 data_vec.data(), file_size, 0));
    HCTR_LOG(INFO, WORLD, "Done!\n");
    counter++;
  }
}

template <typename TypeKey>
void ctor_test_scratch(Optimizer_t opt_type) {
  // create a resource manager for a single GPU
  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back({0});
  const auto resource_manager{ResourceManagerExt::create(vvgpu, 0)};
  std::string path_prefix{"./global_sparse_model"};
  if (std::filesystem::exists(path_prefix)) std::filesystem::remove_all(path_prefix);

  SparseModelFileTS<TypeKey> sparse_model_ts(path_prefix, "./", true, opt_type, emb_vec_size,
                                             resource_manager);
  auto data_files{get_data_file(opt_type)};
  for (const auto& data_file : data_files) {
    auto file_path{path_prefix + "/" + data_file};
    ASSERT_TRUE(std::filesystem::exists(file_path));
  }
}

template <typename TypeKey>
void load_api_test(int batch_num_train, bool use_slot_id, Optimizer_t opt_type, int num_thread) {
  // create a resource manager for a single GPU
  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back({0});
  const auto resource_manager{ResourceManagerExt::create(vvgpu, 0)};

  generate_sparse_model<TypeKey, check>(
      snapshot_src_file, snapshot_dst_file, snapshot_bkp_file_unsigned, snapshot_bkp_file_longlong,
      file_list_name_train, file_list_name_eval, prefix, num_files, label_dim, dense_dim, slot_num,
      max_nnz_per_slot, max_feature_num, vocabulary_size, emb_vec_size, combiner, scaler,
      num_workers, batchsize, batch_num_train, batch_num_eval, update_type, resource_manager);
  generate_opt_state(snapshot_src_file, opt_type);
  if (std::filesystem::exists(snapshot_dst_file)) {
    std::filesystem::remove_all(snapshot_dst_file);
  }
  std::filesystem::copy(snapshot_src_file, snapshot_dst_file);

  SparseModelFileTS<TypeKey> sparse_model_ts(snapshot_dst_file, "./", use_slot_id, opt_type,
                                             emb_vec_size, resource_manager);

  const std::string key_file{std::string(snapshot_dst_file) + "/key"};
  auto num_key{std::filesystem::file_size(key_file) / sizeof(long long)};
  std::vector<TypeKey> keys;
  {
    std::vector<long long> key_i64(num_key);
    std::ifstream ifs(key_file);
    ifs.read(reinterpret_cast<char*>(key_i64.data()), std::filesystem::file_size(key_file));
    if (std::is_same<TypeKey, long long>::value) {
      keys.resize(num_key);
      std::transform(key_i64.begin(), key_i64.end(), keys.begin(),
                     [](long long key) { return key; });
    } else {
      keys.resize(num_key);
      std::transform(key_i64.begin(), key_i64.end(), keys.begin(),
                     [](long long key) { return static_cast<unsigned>(key); });
    }
  }

  auto data_files{get_data_file(opt_type)};
  std::vector<std::vector<float>> data_vecs(data_files.size());
  std::vector<size_t> slot_ids;
  std::vector<float*> data_ptrs;
  size_t* slot_id_ptr{nullptr};
  for (auto& data_vec : data_vecs) {
    data_vec.resize(num_key * emb_vec_size);
    data_ptrs.push_back(data_vec.data());
  }
  if (use_slot_id) {
    slot_ids.resize(num_key);
    slot_id_ptr = slot_ids.data();
  }

  ////////////////////////////// load
  // parallel find
  std::vector<size_t> ssd_idx_vec(num_key);
#pragma omp parallel num_threads(num_thread)
  for (size_t i = 0; i < num_key; i++) {
    auto dst_idx{sparse_model_ts.find(keys[i])};
    if (dst_idx == SparseModelFileTS<TypeKey>::end_flag) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Key doesn't exist");
    }
    ssd_idx_vec[i] = dst_idx;
  }
  sparse_model_ts.load(ssd_idx_vec, slot_id_ptr, data_ptrs);

  // load from src model and check for equality
  compare_with_ssd(data_files, keys, use_slot_id, slot_ids, data_vecs);

  ////////////////////////////// dump_update
  // randomly update some vectors and write back to ssd
  std::vector<size_t> mem_idx_vec(1024);
  std::default_random_engine generator;
  std::uniform_int_distribution<size_t> distribution(0, num_key - 1);
  auto gen_rand_op = [&generator, &distribution](auto& elem) { elem = distribution(generator); };
  for_each(mem_idx_vec.begin(), mem_idx_vec.end(), gen_rand_op);
  sort(mem_idx_vec.begin(), mem_idx_vec.end());
  mem_idx_vec.erase(std::unique(mem_idx_vec.begin(), mem_idx_vec.end()), mem_idx_vec.end());
  HCTR_LOG_S(INFO, ROOT) << mem_idx_vec.size() << " keys selected" << std::endl;

  // determine the ssd_idx_vec
  ssd_idx_vec.resize(mem_idx_vec.size());
#pragma omp parallel num_threads(num_thread)
  for (size_t i = 0; i < mem_idx_vec.size(); i++) {
    auto dst_idx{sparse_model_ts.find(keys[mem_idx_vec[i]])};
    if (dst_idx == SparseModelFileTS<TypeKey>::end_flag) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Key doesn't exist");
    }
    ssd_idx_vec[i] = dst_idx;
  }

  // modify corresponding vectors
  std::uniform_real_distribution<float> real_distribution(0.0f, 1.0f);
  auto gen_real_rand_op = [&generator, &real_distribution](float& elem) {
    elem = real_distribution(generator);
  };
  for (auto idx : mem_idx_vec) {
    for (auto& data_vec : data_vecs) {
      auto srt_it{data_vec.begin() + idx * emb_vec_size};
      auto end_it{srt_it + emb_vec_size};
      for_each(srt_it, end_it, gen_real_rand_op);
    }
    if (use_slot_id) slot_id_ptr[idx] = distribution(generator);
  }
  sparse_model_ts.dump_update(ssd_idx_vec, mem_idx_vec, slot_id_ptr, data_ptrs);

  // load from src model and check for equality
  sparse_model_ts.update_global_model();
  compare_with_ssd(data_files, keys, use_slot_id, slot_ids, data_vecs);

  ////////////////////////////// dump_insert
  // generate new keys
  size_t const num_new_keys(819200);

  size_t max_key_val(*std::max_element(keys.begin(), keys.end()));
  keys.resize(keys.size() + num_new_keys);
  std::iota(keys.begin() + num_key, keys.end(), max_key_val + 1);
  data_ptrs.clear();
  for (auto& data_vec : data_vecs) {
    data_vec.resize(keys.size() * emb_vec_size);
    data_ptrs.push_back(data_vec.data());
    for_each(data_vec.begin() + num_key * emb_vec_size, data_vec.end(), gen_real_rand_op);
  }
  if (use_slot_id) {
    slot_ids.resize(keys.size());
    slot_id_ptr = slot_ids.data();
    for_each(slot_ids.begin() + num_key, slot_ids.end(), gen_rand_op);
  }
  mem_idx_vec.clear();
  mem_idx_vec.resize(num_new_keys);
  std::iota(mem_idx_vec.begin(), mem_idx_vec.end(), num_key);
  sparse_model_ts.dump_insert(keys.data(), mem_idx_vec, slot_id_ptr, data_ptrs);

  // load from src model and check for equality
  sparse_model_ts.update_global_model();
  compare_with_ssd(data_files, keys, use_slot_id, slot_ids, data_vecs);

  ////////////////////////////// dump_update
  std::unordered_map<TypeKey, size_t> key_idx_map;
  key_idx_map.reserve(keys.size());
  size_t count(0);
  for (auto key : keys) {
    key_idx_map.insert({key, count++});
  }
  sparse_model_ts.dump_update(key_idx_map, slot_ids, data_vecs);
  sparse_model_ts.update_global_model();

  std::vector<std::vector<float>> tmp_data_vecs(data_files.size());
  std::vector<size_t> tmp_slot_ids;
  std::vector<float*> tmp_data_ptrs;
  size_t* tmp_slot_id_ptr{nullptr};
  for (auto& data_vec : tmp_data_vecs) {
    data_vec.resize(keys.size() * emb_vec_size);
    tmp_data_ptrs.push_back(data_vec.data());
  }
  if (use_slot_id) {
    tmp_slot_ids.resize(keys.size());
    tmp_slot_id_ptr = tmp_slot_ids.data();
  }
  // determine the ssd_idx_vec
  ssd_idx_vec.resize(keys.size());
#pragma omp parallel num_threads(num_thread)
  for (size_t i = 0; i < keys.size(); i++) {
    auto dst_idx{sparse_model_ts.find(keys[i])};
    if (dst_idx == SparseModelFileTS<TypeKey>::end_flag) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Key doesn't exist");
    }
    ssd_idx_vec[i] = dst_idx;
  }
  sparse_model_ts.load(ssd_idx_vec, tmp_slot_id_ptr, tmp_data_ptrs);

  if (use_slot_id) {
    HCTR_LOG(INFO, WORLD, "check slot_id\n");
    ASSERT_TRUE(test::compare_array_approx<char>(reinterpret_cast<char*>(tmp_slot_ids.data()),
                                                 reinterpret_cast<char*>(slot_ids.data()),
                                                 slot_ids.size() * sizeof(size_t), 0));
    HCTR_LOG(INFO, WORLD, "Done!\n");
  }
  size_t counter{0};
  for (auto const& file : data_files) {
    HCTR_LOG_S(INFO, WORLD) << "check " << file << std::endl;
    ASSERT_TRUE(
        test::compare_array_approx<char>(reinterpret_cast<char*>(tmp_data_vecs[counter].data()),
                                         reinterpret_cast<char*>(data_vecs[counter].data()),
                                         data_vecs[counter].size() * sizeof(float), 0));
    HCTR_LOG(INFO, WORLD, "Done!\n");
    counter++;
  }
}
/*
TEST(sparse_model_file_ts_test, ctor_scratch_long_long_adam) {
  ctor_test_scratch<long long>(Optimizer_t::Adam);
}
TEST(sparse_model_file_ts_test, ctor_scratch_long_long_adagrad) {
  ctor_test_scratch<long long>(Optimizer_t::AdaGrad);
}
TEST(sparse_model_file_ts_test, ctor_scratch_long_long_momentumsgd) {
  ctor_test_scratch<long long>(Optimizer_t::MomentumSGD);
}
TEST(sparse_model_file_ts_test, ctor_scratch_long_long_nesterov) {
  ctor_test_scratch<long long>(Optimizer_t::Nesterov);
}
TEST(sparse_model_file_ts_test, ctor_scratch_long_long_sgd) {
  ctor_test_scratch<long long>(Optimizer_t::SGD);
}
TEST(sparse_model_file_ts_test, ctor_scratch_unsigned_adam) {
  ctor_test_scratch<unsigned>(Optimizer_t::Adam);
}
TEST(sparse_model_file_ts_test, ctor_scratch_unsigned_adagrad) {
  ctor_test_scratch<unsigned>(Optimizer_t::AdaGrad);
}
TEST(sparse_model_file_ts_test, ctor_scratch_unsigned_momentumsgd) {
  ctor_test_scratch<unsigned>(Optimizer_t::MomentumSGD);
}
TEST(sparse_model_file_ts_test, ctor_scratch_unsigned_nesterov) {
  ctor_test_scratch<unsigned>(Optimizer_t::Nesterov);
}
TEST(sparse_model_file_ts_test, ctor_scratch_unsigned_sgd) {
  ctor_test_scratch<unsigned>(Optimizer_t::SGD);
}
*/
TEST(sparse_model_file_ts_test, load_api_long_long_Adam) {
  load_api_test<long long>(30, true, Optimizer_t::Adam, 1);
  // load_api_test<long long>(30, true, Optimizer_t::Adam, 32);
}
/*
TEST(sparse_model_file_ts_test, load_api_unsigned_Adam) {
  load_api_test<unsigned>(20, false, Optimizer_t::Adam, 1);
  load_api_test<unsigned>(20, false, Optimizer_t::Adam, 32);
}
TEST(sparse_model_file_ts_test, load_api_long_long_AdaGrad) {
  load_api_test<long long>(30, true, Optimizer_t::AdaGrad, 1);
  load_api_test<long long>(30, true, Optimizer_t::AdaGrad, 32);
}
TEST(sparse_model_file_ts_test, load_api_unsigned_AdaGrad) {
  load_api_test<unsigned>(20, false, Optimizer_t::AdaGrad, 1);
  load_api_test<unsigned>(20, false, Optimizer_t::AdaGrad, 32);
}
TEST(sparse_model_file_ts_test, load_api_long_long_MomentumSGD) {
  load_api_test<long long>(30, true, Optimizer_t::MomentumSGD, 1);
  load_api_test<long long>(30, true, Optimizer_t::MomentumSGD, 32);
}
TEST(sparse_model_file_ts_test, load_api_unsigned_MomentumSGD) {
  load_api_test<unsigned>(20, false, Optimizer_t::MomentumSGD, 1);
  load_api_test<unsigned>(20, false, Optimizer_t::MomentumSGD, 32);
}
TEST(sparse_model_file_ts_test, load_api_long_long_Nesterov) {
  load_api_test<long long>(30, true, Optimizer_t::Nesterov, 1);
  load_api_test<long long>(30, true, Optimizer_t::Nesterov, 32);
}
TEST(sparse_model_file_ts_test, load_api_unsigned_Nesterov) {
  load_api_test<unsigned>(20, false, Optimizer_t::Nesterov, 1);
  load_api_test<unsigned>(20, false, Optimizer_t::Nesterov, 32);
}
TEST(sparse_model_file_ts_test, load_api_long_long_SGD) {
  load_api_test<long long>(30, true, Optimizer_t::SGD, 1);
  load_api_test<long long>(30, true, Optimizer_t::SGD, 32);
}
TEST(sparse_model_file_ts_test, load_api_unsigned_SGD) {
  load_api_test<unsigned>(20, false, Optimizer_t::SGD, 1);
  load_api_test<unsigned>(20, false, Optimizer_t::SGD, 32);
}
*/
}  // namespace
