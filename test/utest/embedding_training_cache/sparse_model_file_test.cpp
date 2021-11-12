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

#include "utest/embedding_training_cache/etc_test_utils.hpp"
#include "HugeCTR/include/embedding_training_cache/sparse_model_file.hpp"

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
void sparse_model_file_test(int batch_num_train, bool is_distributed) {
  Embedding_t embedding_type = is_distributed ? Embedding_t::DistributedSlotSparseEmbeddingHash :
                                                Embedding_t::LocalizedSlotSparseEmbeddingHash;

  // create a resource manager for a single GPU
  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back({0});
  const auto resource_manager = ResourceManagerExt::create(vvgpu, 0);

  generate_sparse_model<TypeKey, check>(snapshot_src_file, snapshot_dst_file,
      snapshot_bkp_file_unsigned, snapshot_bkp_file_longlong,
      file_list_name_train, file_list_name_eval, prefix, num_files, label_dim,
      dense_dim, slot_num, max_nnz_per_slot, max_feature_num,
      vocabulary_size, emb_vec_size, combiner, scaler, num_workers, batchsize,
      batch_num_train, batch_num_eval, update_type, resource_manager);
  copy_sparse_model(snapshot_src_file, snapshot_dst_file);

  auto get_ext_file = [](const std::string& sparse_model_file, std::string ext) {
    return std::string(sparse_model_file) + "/" + ext;
  };

  // test load_emb_tbl_to_mem
  {
    MESSAGE_("[TEST] sparse_model_file::load_emb_tbl_to_mem");
    HugeCTR::SparseModelFile<TypeKey> sparse_model_file(snapshot_dst_file,
        embedding_type, emb_vec_size, resource_manager);

    std::unordered_map<TypeKey, std::pair<size_t, size_t>> mem_key_index_map;
    std::vector<float> mem_emb_table;
    sparse_model_file.load_emb_tbl_to_mem(mem_key_index_map, mem_emb_table);

    size_t key_file_size_in_byte = fs::file_size(get_ext_file(snapshot_dst_file, "key"));
    size_t vec_file_size_in_byte = fs::file_size(get_ext_file(snapshot_dst_file, "emb_vector"));
    size_t num_keys = key_file_size_in_byte / sizeof(long long);

    ASSERT_TRUE(mem_key_index_map.size() == num_keys);
    ASSERT_TRUE(mem_emb_table.size() == vec_file_size_in_byte / sizeof(float));
    ASSERT_TRUE(num_keys == mem_emb_table.size() / emb_vec_size);

    if (!is_distributed) {
      size_t slot_file_size_in_byte = fs::file_size(get_ext_file(snapshot_dst_file, "slot_id"));
      ASSERT_TRUE(num_keys == slot_file_size_in_byte / sizeof(size_t));
    }

    std::map<size_t, std::pair<TypeKey, size_t>> index_key_map;
    for_each(mem_key_index_map.begin(), mem_key_index_map.end(),
      [&index_key_map](const auto& pair) {
        index_key_map.insert({pair.second.second, {pair.first, pair.second.first}});
    });
    ASSERT_TRUE(index_key_map.size() == mem_key_index_map.size());

    std::vector<TypeKey> mem_key(mem_key_index_map.size());
    auto get_key_op = [](const auto& e) { return e.second.first; };
    transform(index_key_map.begin(), index_key_map.end(), mem_key.begin(), get_key_op);

    std::vector<size_t> mem_idx(mem_key_index_map.size());
    auto get_idx_op = [](const auto& e) { return e.first; };
    transform(index_key_map.begin(), index_key_map.end(), mem_idx.begin(), get_idx_op);

    std::vector<size_t> mem_slot(mem_key_index_map.size());
    auto get_slot_op = [](const auto& e) { return e.second.second; };
    if (!is_distributed) {
      transform(index_key_map.begin(), index_key_map.end(), mem_slot.begin(), get_slot_op);
    }

    std::vector<size_t> bench_mem_idx(mem_idx.size());
    iota(bench_mem_idx.begin(), bench_mem_idx.end(), 0);
    ASSERT_TRUE(test::compare_array_approx<char>(
        reinterpret_cast<char *>(bench_mem_idx.data()),
        reinterpret_cast<char *>(mem_idx.data()), mem_idx.size() * sizeof(size_t), 0));

    std::vector<TypeKey> key_in_file(num_keys);
    std::vector<float> vec_in_file(num_keys * emb_vec_size);
    std::ifstream key_ifs(get_ext_file(snapshot_dst_file, "key"));
    std::ifstream vec_ifs(get_ext_file(snapshot_dst_file, "emb_vector"));
    load_key_to_vec(key_in_file, key_ifs, num_keys, key_file_size_in_byte);
    vec_ifs.read(reinterpret_cast<char *>(vec_in_file.data()), vec_file_size_in_byte);

    std::vector<size_t> slot_in_file(num_keys);
    if (!is_distributed) {
      size_t slot_file_size_in_byte = fs::file_size(get_ext_file(snapshot_dst_file, "slot_id"));
      std::ifstream slot_ifs(get_ext_file(snapshot_dst_file, "slot_id"));
      slot_ifs.read(reinterpret_cast<char *>(slot_in_file.data()), slot_file_size_in_byte);
    }

    typedef struct TypeHashValue_ { float data[emb_vec_size]; } TypeHashValue;
    ASSERT_TRUE(HugeCTR::embedding_test::compare_hash_table(mem_key.size(),
      mem_key.data(), reinterpret_cast<TypeHashValue *>(mem_emb_table.data()),
      key_in_file.data(), reinterpret_cast<TypeHashValue *>(vec_in_file.data()),
      1e-4));

    if (!is_distributed) {
      ASSERT_TRUE(HugeCTR::embedding_test::compare_key_slot(mem_key.size(),
        mem_key.data(), mem_slot.data(), key_in_file.data(), slot_in_file.data()));
    }
  }

  {
    MESSAGE_("[TEST] sparse_model_file::append_new_vec_and_key");
    const char *temp_snapshot_file = "tmp_emb_file";
    if (fs::exists(temp_snapshot_file)) {
      fs::remove_all(temp_snapshot_file);
    }

    HugeCTR::SparseModelFile<TypeKey> sparse_model_file(temp_snapshot_file,
        embedding_type, emb_vec_size, resource_manager);

    size_t key_file_size_in_byte = fs::file_size(get_ext_file(snapshot_dst_file, "key"));
    size_t vec_file_size_in_byte = fs::file_size(get_ext_file(snapshot_dst_file, "emb_vector"));
    size_t num_keys = key_file_size_in_byte / sizeof(long long);

    std::vector<TypeKey> key_in_file(num_keys);
    std::ifstream key_ifs(get_ext_file(snapshot_dst_file, "key"));
    load_key_to_vec(key_in_file, key_ifs, num_keys, key_file_size_in_byte);

    std::vector<float> vec_in_file(num_keys * emb_vec_size);
    std::ifstream vec_ifs(get_ext_file(snapshot_dst_file, "emb_vector"));
    vec_ifs.read(reinterpret_cast<char *>(vec_in_file.data()), vec_file_size_in_byte);

    std::vector<size_t> slot_in_file(num_keys);
    if (!is_distributed) {
      size_t slot_file_size_in_byte = fs::file_size(get_ext_file(snapshot_dst_file, "slot_id"));
      std::ifstream slot_ifs(get_ext_file(snapshot_dst_file, "slot_id"));
      slot_ifs.read(reinterpret_cast<char *>(slot_in_file.data()), slot_file_size_in_byte);
    }

    // test append_new_vec_and_key
    std::vector<size_t> vec_indices(key_in_file.size());
    iota(vec_indices.begin(), vec_indices.end(), 0);

    if (is_distributed) {
      sparse_model_file.append_new_vec_and_key(key_in_file, nullptr,
                                               vec_indices, vec_in_file.data());
    } else {
      sparse_model_file.append_new_vec_and_key(key_in_file, slot_in_file.data(),
                                               vec_indices, vec_in_file.data());
    }

    check_vector_equality(temp_snapshot_file, snapshot_dst_file, "key");
    check_vector_equality(temp_snapshot_file, snapshot_dst_file, "emb_vector");
    if (!is_distributed) {
      check_vector_equality(temp_snapshot_file, snapshot_dst_file, "slot_id");
    }

    // test load_exist_vec_by_key
    MESSAGE_("[TEST] sparse_model_file::load_exist_vec_by_key");
    std::vector<size_t> slot_vec;
    std::vector<float> mem_vec;
    sparse_model_file.load_exist_vec_by_key(key_in_file, slot_vec, mem_vec);
    ASSERT_TRUE(test::compare_array_approx<char>(
        reinterpret_cast<char *>(mem_vec.data()),
        reinterpret_cast<char *>(vec_in_file.data()), mem_vec.size() * sizeof(float), 0));
  }

  // test dump_exist_vec_by_key
  {
    MESSAGE_("[TEST] sparse_model_file::dump_exist_vec_by_key");
    HugeCTR::SparseModelFile<TypeKey> sparse_model_file(snapshot_dst_file,
        embedding_type, emb_vec_size, resource_manager);

    size_t key_file_size_in_byte = fs::file_size(get_ext_file(snapshot_dst_file, "key"));
    size_t vec_file_size_in_byte = fs::file_size(get_ext_file(snapshot_dst_file, "emb_vector"));
    size_t num_keys = key_file_size_in_byte / sizeof(long long);
    size_t num_vecs = vec_file_size_in_byte / (emb_vec_size * sizeof(float));
    ASSERT_TRUE(num_keys == num_vecs);

    std::vector<TypeKey> all_keys(num_keys);
    {
      std::ifstream key_ifs(get_ext_file(snapshot_dst_file, "key"));
      load_key_to_vec(all_keys, key_ifs, num_keys, key_file_size_in_byte);
    }

    std::vector<TypeKey> rand_idx(1024);
    std::default_random_engine generator;
    std::uniform_int_distribution<size_t> distribution(0, num_keys);
    auto gen_rand_op = [&generator, &distribution](auto& elem) { elem = distribution(generator); };
    for_each(rand_idx.begin(), rand_idx.end(), gen_rand_op);
    sort(rand_idx.begin(), rand_idx.end());
    rand_idx.erase(std::unique(rand_idx.begin(), rand_idx.end()), rand_idx.end());
    MESSAGE_(std::to_string(rand_idx.size()) + " keys selected");

    std::vector<TypeKey> selt_keys(rand_idx.size());
    auto get_key_op = [&all_keys](size_t idx) { return all_keys[idx]; };
    transform(rand_idx.begin(), rand_idx.end(), selt_keys.begin(), get_key_op);

    // load vectors according to keys
    std::vector<size_t> load_slots;
    std::vector<float> load_vecs;
    sparse_model_file.load_exist_vec_by_key(selt_keys, load_slots, load_vecs);

    // modify loaded vectors
    std::uniform_real_distribution<float> real_distribution(0.0f, 1.0f);
    auto gen_real_rand_op = [&generator, &real_distribution](float& elem) {
      elem = real_distribution(generator); };
    for_each(load_vecs.begin(), load_vecs.end(), gen_real_rand_op);

    // dump loaded vector back
    std::vector<size_t> vec_indices(selt_keys.size());
    iota(vec_indices.begin(), vec_indices.end(), 0);
    sparse_model_file.dump_exist_vec_by_key(selt_keys, vec_indices, load_vecs.data());

    // reload updated vectors
    std::vector<size_t> re_load_slots;
    std::vector<float> re_load_vecs;
    sparse_model_file.load_exist_vec_by_key(selt_keys, re_load_slots, re_load_vecs);

    ASSERT_TRUE(test::compare_array_approx<char>(
        reinterpret_cast<char *>(load_vecs.data()),
        reinterpret_cast<char *>(re_load_vecs.data()), load_vecs.size() * sizeof(float), 0));

    if (!is_distributed) {
      ASSERT_TRUE(test::compare_array_approx<char>(
        reinterpret_cast<char *>(load_slots.data()),
        reinterpret_cast<char *>(re_load_slots.data()), load_slots.size() * sizeof(size_t), 0));
    }
  }
}

TEST(sparse_model_file_test, long_long_distributed) {
  sparse_model_file_test<long long>(30, true);
}

TEST(sparse_model_file_test, unsigned_localized) {
  sparse_model_file_test<unsigned>(20, false);
}

}  // namespace
