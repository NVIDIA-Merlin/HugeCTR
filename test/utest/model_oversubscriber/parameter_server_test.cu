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
#include "HugeCTR/include/model_oversubscriber/parameter_server.hpp"

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
const char* keyset_file_name = "keyset_file.bin";

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
void do_upload_and_download_snapshot(
    size_t batch_num_train, bool use_host_ps, bool is_distributed) {
  Embedding_t embedding_type = is_distributed ? 
                               Embedding_t::DistributedSlotSparseEmbeddingHash :
                               Embedding_t::LocalizedSlotSparseEmbeddingHash;
  // create a resource manager for a single GPU
  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back({0});
  const auto resource_manager = ResourceManager::create(vvgpu, 0);

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

  // Create a ParameterServer
  ParameterServer<TypeKey> parameter_server(use_host_ps, snapshot_dst_file,
      embedding_type, emb_vec_size, resource_manager);

  // Make a synthetic keyset files
  {
    size_t key_file_size_in_byte =
        fs::file_size(get_ext_file(snapshot_dst_file, "key"));
    size_t num_keys = key_file_size_in_byte / sizeof(long long);
    std::vector<long long> keys_in_file(num_keys);
    std::ifstream key_ifs(get_ext_file(snapshot_dst_file, "key"));
    key_ifs.read(reinterpret_cast<char *>(keys_in_file.data()),
                                          key_file_size_in_byte);
    TypeKey *key_ptr = nullptr;
    std::vector<TypeKey> key_vec;
    if (std::is_same<TypeKey, long long>::value) {
      key_ptr = reinterpret_cast<TypeKey*>(keys_in_file.data());
    } else {
      key_vec.resize(num_keys);
      std::transform(keys_in_file.begin(), keys_in_file.end(), key_vec.begin(),
                     [](long long key) { return static_cast<unsigned>(key); });
      key_ptr = key_vec.data();
    }
    std::ofstream key_ofs(keyset_file_name, std::ofstream::binary |
                                            std::ofstream::trunc);
    key_ofs.write(reinterpret_cast<char *>(key_ptr), num_keys * sizeof(TypeKey));
  }

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

  Timer timer_ps;
  timer_ps.start();

  parameter_server.load_keyset_from_file(keyset_file_name);

  size_t size_tmp = 0;
  parameter_server.pull(buf_bag, size_tmp);
  parameter_server.push(buf_bag, size_tmp);
  parameter_server.flush_emb_tbl_to_ssd();

  MESSAGE_("Batch_num=" + std::to_string(batch_num_train) +
           ", embedding_vec_size=" + std::to_string(emb_vec_size) +
           ", elapsed time=" + std::to_string(timer_ps.elapsedSeconds()) + "s");

  // Check if the result is correct
  ASSERT_TRUE(check_vector_equality(snapshot_src_file, snapshot_dst_file, "key"));
  ASSERT_TRUE(check_vector_equality(snapshot_src_file, snapshot_dst_file, "emb_vector"));
  if (!is_distributed)
    ASSERT_TRUE(check_vector_equality(snapshot_src_file, snapshot_dst_file, "slot_id"));
}

TEST(parameter_server_test, long_long_ssd_distributed) {
  do_upload_and_download_snapshot<long long>(30, false, true);
}

TEST(parameter_server_test, unsigned_host_distributed) {
  do_upload_and_download_snapshot<unsigned>(20, true, true);
}

TEST(parameter_server_test, long_long_ssd_localized) {
  do_upload_and_download_snapshot<long long>(30, false, false);
}

TEST(parameter_server_test, unsigned_host_localized) {
  do_upload_and_download_snapshot<unsigned>(20, true, false);
}

}  // namespace
