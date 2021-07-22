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
#include "HugeCTR/include/model_oversubscriber/sparse_model_entity.hpp"

using namespace HugeCTR;
using namespace mos_test;

namespace {

const char* prefix = "./model_oversubscriber_test_data/tmp_";
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
void sparse_model_entity_test(int batch_num_train, bool use_host_mem, bool is_distributed) {
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

  BufferBag buf_bag;
  {
    std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> blobs_buff =
      GeneralBuffer2<CudaHostAllocator>::create();

    Tensor2<TypeKey> tensor_keys;
    Tensor2<size_t> tensor_slot_id;
    blobs_buff->reserve({vocabulary_size}, &tensor_keys);
    blobs_buff->reserve({vocabulary_size}, &tensor_slot_id);

    blobs_buff->reserve({vocabulary_size, emb_vec_size}, &(buf_bag.embedding));
    blobs_buff->allocate();

    buf_bag.keys = tensor_keys.shrink();
    buf_bag.slot_id = tensor_slot_id.shrink();
  }
  float *emb_ptr = buf_bag.embedding.get_ptr();

  // test load_vec_by_key
  MESSAGE_("[TEST] sparse_model_entity::load_vec_by_key");
  HugeCTR::SparseModelEntity<TypeKey> sparse_model_entity(use_host_mem, snapshot_dst_file,
      embedding_type, emb_vec_size, resource_manager);

  size_t key_file_size_in_byte = fs::file_size(get_ext_file(snapshot_dst_file, "key"));
  size_t vec_file_size_in_byte = fs::file_size(get_ext_file(snapshot_dst_file, "emb_vector"));
  size_t num_keys = key_file_size_in_byte / sizeof(long long);
  ASSERT_TRUE(num_keys == vec_file_size_in_byte / sizeof(float) / emb_vec_size);

  std::vector<TypeKey> key_in_file(num_keys);
  std::ifstream key_ifs(get_ext_file(snapshot_dst_file, "key"));
  load_key_to_vec(key_in_file, key_ifs, num_keys, key_file_size_in_byte);

  std::vector<float> vec_in_file(num_keys * emb_vec_size);
  std::ifstream vec_ifs(get_ext_file(snapshot_dst_file, "emb_vector"));
  vec_ifs.read(reinterpret_cast<char *>(vec_in_file.data()), vec_file_size_in_byte);

  // load all embedding features
  size_t hit_size;
  sparse_model_entity.load_vec_by_key(key_in_file, buf_bag, hit_size);
  ASSERT_EQ(hit_size, key_in_file.size());

  ASSERT_TRUE(test::compare_array_approx<char>(reinterpret_cast<char *>(vec_in_file.data()),
      reinterpret_cast<char *>(emb_ptr), vec_in_file.size() * sizeof(float), 0));
  
  std::vector<size_t> slot_in_file(num_keys);
  if (!is_distributed) {
    size_t slot_file_size_in_byte = fs::file_size(get_ext_file(snapshot_dst_file, "slot_id"));
    std::ifstream slot_ifs(get_ext_file(snapshot_dst_file, "slot_id"));
    slot_ifs.read(reinterpret_cast<char *>(slot_in_file.data()), slot_file_size_in_byte);

    size_t *slot_id_ptr = Tensor2<size_t>::stretch_from(buf_bag.slot_id).get_ptr();
    ASSERT_TRUE(test::compare_array_approx<size_t>(slot_in_file.data(), slot_id_ptr,
        hit_size, 0));
  }

  MESSAGE_("[TEST] sparse_model_entity::dump_vec_by_key");
  std::default_random_engine generator;
  std::uniform_real_distribution<float> real_distribution(0.0f, 1.0f);
  auto gen_real_rand_op = [&generator, &real_distribution](float& elem) {
    elem = real_distribution(generator); };
  for_each(vec_in_file.begin(), vec_in_file.end(), gen_real_rand_op);
  memcpy(emb_ptr, vec_in_file.data(), vec_in_file.size() * sizeof(float));
  {
    std::ofstream vec_ofs(get_ext_file(snapshot_src_file, "emb_vector"), std::ofstream::trunc);
    vec_ofs.write(reinterpret_cast<char *>(vec_in_file.data()), vec_in_file.size() * sizeof(float));
  }

  // dump all embedding features
  sparse_model_entity.dump_vec_by_key(buf_bag, hit_size);
  if (use_host_mem) sparse_model_entity.flush_emb_tbl_to_ssd();
  ASSERT_TRUE(check_vector_equality(snapshot_src_file, snapshot_dst_file, "emb_vector"));

  // load part embedding features
  MESSAGE_("[TEST] both load and dump");
  std::vector<TypeKey> rand_idx(static_cast<size_t>(key_in_file.size() * 0.4));
  std::uniform_int_distribution<size_t> distribution(0, num_keys - 1);
  auto gen_rand_op = [&generator, &distribution](auto& elem) { elem = distribution(generator); };
  for_each(rand_idx.begin(), rand_idx.end(), gen_rand_op);
  sort(rand_idx.begin(), rand_idx.end());
  rand_idx.erase(std::unique(rand_idx.begin(), rand_idx.end()), rand_idx.end());
  MESSAGE_(std::to_string(rand_idx.size()) + " keys selected");

  std::vector<TypeKey> selt_keys(rand_idx.size());
  auto get_key_op = [&key_in_file](size_t idx) { return key_in_file[idx]; };
  transform(rand_idx.begin(), rand_idx.end(), selt_keys.begin(), get_key_op);

  sparse_model_entity.load_vec_by_key(selt_keys, buf_bag, hit_size);
  ASSERT_EQ(hit_size, selt_keys.size());

  // following is not appliable to multi-node test
  std::vector<float> selt_vecs(hit_size * emb_vec_size, 0.0f);
  for_each(selt_vecs.begin(), selt_vecs.end(), gen_real_rand_op);
  memcpy(emb_ptr, selt_vecs.data(), selt_vecs.size() * sizeof(float));
  sparse_model_entity.dump_vec_by_key(buf_bag, hit_size);
  sparse_model_entity.flush_emb_tbl_to_ssd();

  {
    HugeCTR::SparseModelFile<TypeKey> sparse_model_file(snapshot_src_file,
          embedding_type, emb_vec_size, resource_manager);

    std::vector<size_t> vec_indices(selt_keys.size());
    iota(vec_indices.begin(), vec_indices.end(), 0);
    sparse_model_file.dump_exist_vec_by_key(selt_keys, vec_indices, selt_vecs.data());
  }
  ASSERT_TRUE(check_vector_equality(snapshot_src_file, snapshot_dst_file, "emb_vector"));
}

TEST(sparse_model_entity_test, long_long_ssd_distributed) {
  sparse_model_entity_test<long long>(30, false, true);
}

TEST(sparse_model_entity_test, unsigned_host_distributed) {
  sparse_model_entity_test<unsigned>(30, true, true);
}

TEST(sparse_model_entity_test, long_long_ssd_localized) {
  sparse_model_entity_test<long long>(20, false, false);
}

TEST(sparse_model_entity_test, unsigned_host_localized) {
  sparse_model_entity_test<unsigned>(20, true, false);
}


}  // namespace
