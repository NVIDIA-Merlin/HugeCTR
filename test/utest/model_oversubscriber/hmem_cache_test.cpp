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

#include <gtest/gtest.h>
#include "utest/model_oversubscriber/mos_test_utils.hpp"
#include "model_oversubscriber/hmem_cache/hmem_cache.hpp"

#include <algorithm>
#include <fstream>
#include <random>
#include <type_traits>
#include <omp.h>

using namespace HugeCTR;
using namespace mos_test;

namespace {

std::string snapshot_src_file{"hmem_cache_table_src"};
std::string snapshot_dst_file{"hmem_cache_table_dst"};
const long long vocabulary_size = 100000;
const int emb_vec_size = 64;

void generate_embedding_table(std::string table_name, double table_size_in_gb,
                              Optimizer_t opt_type, size_t emb_vec_size) {
  size_t num_target_key{static_cast<size_t>(
      table_size_in_gb * (pow(1024, 3) / emb_vec_size / sizeof(float)))};

  std::string key_file{table_name + "/key"};
  std::string slot_id_file{table_name + "/slot_id"};
  auto data_files{get_data_file(opt_type)};

  bool is_exist{fs::exists(table_name)};
  bool is_valid{true};
  if (is_exist) {
    size_t num_key{fs::file_size(key_file) / sizeof(long long)};
    size_t num_slot_id{fs::file_size(slot_id_file) / sizeof(size_t)};
    if (num_key != num_slot_id || num_key != num_target_key) {
      is_valid = false;
    }
    for (auto& data_file : data_files) {
      std::string file_name{table_name + "/" + data_file};
      size_t num_vec{fs::file_size(file_name) / emb_vec_size / sizeof(float)};
      if (num_key != num_vec) {
        is_valid = false;
        break;
      }
    }
  }
  if (is_exist && is_valid) return;

  data_files.clear();
  data_files = std::vector<std::string>({"emb_vector", "Adam.m", "Adam.v", "AdaGrad.accm",
                                         "MomentumSGD.momtentum", "Nesterov.accm"});
  if (is_exist) fs::remove_all(table_name);
  fs::create_directories(table_name);
  std::vector<long long> keys(num_target_key);
  std::vector<size_t> slot_ids(num_target_key);
  std::vector<std::vector<float>> data_vecs(data_files.size(),
                                            std::vector<float>(num_target_key * emb_vec_size));

  std::default_random_engine generator;
  std::uniform_int_distribution<size_t> distribution(0, num_target_key - 1);
  auto gen_rand_op = [&generator, &distribution](auto& elem) { elem = distribution(generator); };
  std::uniform_real_distribution<float> real_distribution(0.0f, 1.0f);
  auto gen_real_rand_op = [&generator, &real_distribution](float& elem) {
    elem = real_distribution(generator); };

  {
    std::iota(keys.begin(), keys.end(), 0);
    std::ofstream ofs(key_file, std::ofstream::trunc);
    if (!ofs.is_open()) CK_THROW_(Error_t::FileCannotOpen, "File open error");
    ofs.write(reinterpret_cast<char *>(keys.data()), keys.size() * sizeof(long long));
  }
  {
    std::for_each(slot_ids.begin(), slot_ids.end(), gen_rand_op);
    std::ofstream ofs(slot_id_file, std::ofstream::trunc);
    if (!ofs.is_open()) CK_THROW_(Error_t::FileCannotOpen, "File open error");
    ofs.write(reinterpret_cast<char *>(slot_ids.data()), slot_ids.size() * sizeof(size_t));
  }
#pragma omp parallel for num_threads(data_files.size())
  for (size_t i = 0; i < data_files.size(); i++) {
    for_each(data_vecs[i].begin(), data_vecs[i].end(), gen_real_rand_op);
    std::string file_name{table_name + "/" + data_files[i]};
    std::ofstream ofs(file_name, std::ofstream::trunc);
    if (!ofs.is_open()) CK_THROW_(Error_t::FileCannotOpen, "File open error");
    ofs.write(reinterpret_cast<char *>(data_vecs[i].data()), data_vecs[i].size() * sizeof(float));
  }
}

template <typename TypeKey>
void read_api_test(double table_size, size_t num_pass, size_t num_cached_pass,
                   double target_hit_rate, size_t max_eviction,
                   bool use_slot_id, Optimizer_t opt_type) {
  // create a resource manager for a single GPU
  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back({0});
  const auto resource_manager{ResourceManagerExt::create(vvgpu, 0)};

  generate_embedding_table(snapshot_src_file, table_size, opt_type, emb_vec_size);
  if (fs::exists(snapshot_dst_file)) fs::remove_all(snapshot_dst_file);
  fs::copy(snapshot_src_file, snapshot_dst_file);

  const std::string key_file{std::string(snapshot_dst_file) + "/key"};
  auto num_key{fs::file_size(key_file) / sizeof(long long)};
  std::vector<TypeKey> keys;
  {
    std::vector<long long> key_i64(num_key);
    std::ifstream ifs(key_file);
    ifs.read(reinterpret_cast<char *>(key_i64.data()), fs::file_size(key_file));
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
  auto max_key_val{*std::max_element(keys.begin(), keys.end())};

  auto overlap_rate{0.2};
  size_t max_vocabulary_size{num_key / num_pass * 2};
  HMemCache<TypeKey> hmem_cache(num_cached_pass, target_hit_rate, max_eviction, max_vocabulary_size,
      snapshot_dst_file, "./", use_slot_id, opt_type, emb_vec_size, resource_manager);

  std::vector<std::vector<TypeKey>> key_vecs(num_pass);
  auto num_overlap{static_cast<size_t>(std::floor(num_key / num_pass * overlap_rate))};
  for (auto& key_vec : key_vecs) key_vec.reserve(num_key / num_pass + num_overlap);
#pragma omp parallel for num_threads(num_pass)
  for (size_t i = 0; i < num_pass; i++) {
    size_t srt_idx{i * num_key / num_pass};
    size_t end_idx{(i + 1) * num_key / num_pass};
    if (i != key_vecs.size() - 1) end_idx += num_overlap;
    key_vecs[i].resize(end_idx - srt_idx);
    std::copy(keys.begin() + srt_idx, keys.begin() + end_idx, key_vecs[i].begin());
  }

  auto data_files{get_data_file(opt_type)};
  std::vector<size_t> slot_ids;
  if (use_slot_id) slot_ids.resize(max_vocabulary_size);
  std::vector<std::vector<float>> data_vecs(data_files.size());
  std::vector<float *> data_ptrs;
  for (auto& data_vec : data_vecs) {
    data_vec.resize(max_vocabulary_size * emb_vec_size);
    data_ptrs.push_back(data_vec.data());
  }

  auto check_vector_equality = [use_slot_id, &data_files](size_t len,
      std::vector<size_t>& slot_ids_src,
      std::vector<size_t>& slot_ids_dst,
      std::vector<std::vector<float>>& data_vecs_src,
      std::vector<std::vector<float>>& data_vecs_dst) {
    // check equality
    if (use_slot_id) {
      MESSAGE_("check slot_id", true, false);
      ASSERT_TRUE(test::compare_array_approx<char>(
          reinterpret_cast<char *>(slot_ids_src.data()),
          reinterpret_cast<char *>(slot_ids_dst.data()),
          len * sizeof(size_t), 0));
      MESSAGE_(" [DONE]", true, true, false);
    }
    size_t counter{0};
    for (const auto& data_file : data_files) {
      MESSAGE_(std::string("check ") + data_file, true, false);
      ASSERT_TRUE(test::compare_array_approx<char>(
          reinterpret_cast<char *>(data_vecs_src[counter].data()),
          reinterpret_cast<char *>(data_vecs_dst[counter].data()),
          len * emb_vec_size * sizeof(float), 0));
      MESSAGE_(" [DONE]", true, true, false);
      counter++;
    }
  };

  MESSAGE_("Check HMemCache::read()");
  auto sparse_model_ptr{hmem_cache.get_sparse_model_file()};
  std::vector<size_t> tmp_slot_ids;
  if (use_slot_id) tmp_slot_ids.resize(max_vocabulary_size);
  std::vector<std::vector<float>> tmp_data_vecs(data_files.size());
  std::vector<float *> tmp_data_ptrs;
  for (auto& data_vec : tmp_data_vecs) {
    data_vec.resize(max_vocabulary_size * emb_vec_size);
    tmp_data_ptrs.push_back(data_vec.data());
  }
  for (size_t iter{0}; iter < 2; iter++) {
    for (size_t pass_id{0}; pass_id < num_pass; pass_id++) {
      std::cout << "Load pass " << std::to_string(pass_id) << ": ";
      size_t len{key_vecs[pass_id].size()};
      hmem_cache.read(key_vecs[pass_id].data(), len, slot_ids.data(), data_ptrs);

      // parallel find
      std::vector<size_t> ssd_idx_vec(len);
    #pragma omp parallel num_threads(12)
      for (size_t i = 0; i < len; i++) {
        auto dst_idx{sparse_model_ptr->find(key_vecs[pass_id][i])};
        if (dst_idx == SparseModelFileTS<TypeKey>::end_flag) {
          CK_THROW_(Error_t::WrongInput, "Key doesn't exist");
        }
        ssd_idx_vec[i] = dst_idx;
      }
      sparse_model_ptr->load(ssd_idx_vec, tmp_slot_ids.data(), tmp_data_ptrs);
      check_vector_equality(len, slot_ids, tmp_slot_ids, data_vecs, tmp_data_vecs);
    }
  }

  // write test
  MESSAGE_("Check HMemCache::write()");
  auto num_dump{num_key / num_pass};
  std::vector<TypeKey> dump_keys(num_dump);
  size_t offset{0};
  for (size_t pass_id{0}; pass_id < num_pass; pass_id++) {
    auto num_dump_per_pass{num_dump / num_pass};
    auto srt_it{key_vecs[pass_id].begin()};
    auto dst_it{dump_keys.begin()};
    std::copy(srt_it, srt_it + num_dump_per_pass, dst_it + offset);
    offset += num_dump_per_pass;
  }
  {
    std::vector<size_t> ssd_idx_vec(num_dump);
  #pragma omp parallel num_threads(12)
    for (size_t i = 0; i < num_dump; i++) {
      auto dst_idx{sparse_model_ptr->find(dump_keys[i])};
      if (dst_idx == SparseModelFileTS<TypeKey>::end_flag) {
        CK_THROW_(Error_t::WrongInput, "Key doesn't exist");
      }
      ssd_idx_vec[i] = dst_idx;
    }
    sparse_model_ptr->load(ssd_idx_vec, tmp_slot_ids.data(), tmp_data_ptrs);
  }
  auto num_new_keys{8192};
  dump_keys.resize(num_dump + num_new_keys);
  std::iota(dump_keys.begin() + num_dump, dump_keys.end(), max_key_val + 1);

  std::sort(dump_keys.begin(), dump_keys.end());
  dump_keys.erase(std::unique(dump_keys.begin(), dump_keys.end()), dump_keys.end());
  num_dump = dump_keys.size();

  std::default_random_engine generator;
  std::uniform_int_distribution<size_t> distribution(0, num_key - 1);
  auto gen_rand_op = [&generator, &distribution](auto& elem) { elem = distribution(generator); };
  std::uniform_real_distribution<float> real_distribution(0.0f, 1.0f);
  auto gen_real_rand_op = [&generator, &real_distribution](float& elem) {
    elem = real_distribution(generator); };
  if (use_slot_id) {
    for_each(tmp_slot_ids.begin(), tmp_slot_ids.begin() + num_dump, gen_rand_op);
  }
  for (auto& data_vec : tmp_data_vecs) {
    for_each(data_vec.begin(), data_vec.begin() + num_dump * emb_vec_size, gen_real_rand_op);
  }
  hmem_cache.write(dump_keys.data(), num_dump, tmp_slot_ids.data(), tmp_data_ptrs);

  size_t load_len{num_dump};
  hmem_cache.read(dump_keys.data(), load_len, slot_ids.data(), data_ptrs);

  ASSERT_EQ(load_len, num_dump);
  check_vector_equality(num_dump, slot_ids, tmp_slot_ids, data_vecs, tmp_data_vecs);

  HugeCTR::MESSAGE_("Check HMemCache::sync_to_ssd()");
  hmem_cache.sync_to_ssd();
  {
    std::vector<size_t> ssd_idx_vec(num_dump);
  #pragma omp parallel num_threads(12)
    for (size_t i = 0; i < num_dump; i++) {
      auto dst_idx{sparse_model_ptr->find(dump_keys[i])};
      if (dst_idx == SparseModelFileTS<TypeKey>::end_flag) {
        CK_THROW_(Error_t::WrongInput, "Key doesn't exist");
      }
      ssd_idx_vec[i] = dst_idx;
    }
    sparse_model_ptr->load(ssd_idx_vec, tmp_slot_ids.data(), tmp_data_ptrs);
  }
  check_vector_equality(num_dump, slot_ids, tmp_slot_ids, data_vecs, tmp_data_vecs);
}

TEST(hmem_cache_test, long_long_adam_4_2_80_4) {
  read_api_test<long long>(0.8, 4, 2, 0.8, 4, true, Optimizer_t::Adam);
}

TEST(hmem_cache_test, unsigned_sgd_8_8_80_4) {
  read_api_test<unsigned>(0.8, 8, 8, 0.8, 4, false, Optimizer_t::SGD);
}

}  // namespace
