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
#include "mos_test_utils.hpp"
#include "model_oversubscriber/hmem_cache/hmem_cache.hpp"
#include "model_oversubscriber/parameter_server.hpp"

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
    int batch_num_train, TrainPSType_t ps_type, bool is_distributed,
    Optimizer_t opt_type = Optimizer_t::Adam, std::string local_path = "./",
    HMemCacheConfig hc_config = HMemCacheConfig()) {
  Embedding_t embedding_type = is_distributed ? 
                               Embedding_t::DistributedSlotSparseEmbeddingHash :
                               Embedding_t::LocalizedSlotSparseEmbeddingHash;
  // create a resource manager for a single GPU
  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back({0});
  const auto resource_manager{ResourceManagerExt::create(vvgpu, 0)};

  generate_sparse_model<TypeKey, check>(snapshot_src_file, snapshot_dst_file,
      snapshot_bkp_file_unsigned, snapshot_bkp_file_longlong,
      file_list_name_train, file_list_name_eval, prefix, num_files, label_dim,
      dense_dim, slot_num, max_nnz_per_slot, max_feature_num,
      vocabulary_size, emb_vec_size, combiner, scaler, num_workers, batchsize,
      batch_num_train, batch_num_eval, update_type, resource_manager);
  generate_opt_state(snapshot_src_file, opt_type);
  if (fs::exists(snapshot_dst_file)) {
    fs::remove_all(snapshot_dst_file);
  }
  fs::copy(snapshot_src_file, snapshot_dst_file, fs::copy_options::recursive);

  auto get_ext_file = [](const std::string& sparse_model_file, std::string ext) {
    return std::string(sparse_model_file) + "/" + ext;
  };

  // Create a ParameterServer
  hc_config.block_capacity = vocabulary_size;
  ParameterServer<TypeKey> parameter_server(ps_type, snapshot_dst_file,
      embedding_type, opt_type, emb_vec_size, resource_manager, local_path, hc_config);

  // Make a synthetic keyset files
  std::vector<long long> keys_in_file;
  {
    size_t key_file_size_in_byte =
        fs::file_size(get_ext_file(snapshot_dst_file, "key"));
    size_t num_keys = key_file_size_in_byte / sizeof(long long);
    keys_in_file.resize(num_keys);
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
    auto blobs_buff{GeneralBuffer2<CudaHostAllocator>::create()};
    buf_bag.opt_states.resize(vec_per_line[opt_type] - 1);

    Tensor2<TypeKey> tensor_keys;
    Tensor2<size_t> tensor_slot_id;
    blobs_buff->reserve({vocabulary_size}, &tensor_keys);
    blobs_buff->reserve({vocabulary_size}, &tensor_slot_id);
    blobs_buff->reserve({vocabulary_size, emb_vec_size}, &(buf_bag.embedding));
    for (auto& opt_state : buf_bag.opt_states) {
      blobs_buff->reserve({vocabulary_size, emb_vec_size}, &opt_state);
    }
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
  std::vector<std::string> data_files{"key"};
  if (!is_distributed) data_files.push_back("slot_id");
  auto vec_files{get_data_file(opt_type)};
  if (ps_type == TrainPSType_t::Cached) {
    for (auto const& vec_file : vec_files) data_files.push_back(vec_file);
  } else {
    data_files.push_back(vec_files[0]);
  }
  for (const auto& data_file : data_files) {
    std::string dst_name(snapshot_dst_file);
    MESSAGE_(std::string("check ") + dst_name + "/" + data_file, true, false);
    ASSERT_TRUE(check_vector_equality(snapshot_src_file, dst_name.c_str(), data_file.c_str()));
    MESSAGE_(" [DONE]", true, true, false);
  }

  auto key_vec_pair{parameter_server.pull(keys_in_file)};
  std::string vec_file_name("./emb_vector");
  std::ofstream vec_ofs(vec_file_name, std::ofstream::binary | std::ofstream::trunc);
  vec_ofs.write(reinterpret_cast<char *>(key_vec_pair.second.data()),
      key_vec_pair.second.size() * sizeof(float));

  ASSERT_EQ(key_vec_pair.first.size(), keys_in_file.size());
  ASSERT_TRUE(check_vector_equality(snapshot_src_file, "./", "emb_vector"));
}

TEST(parameter_server_test, unsigned_host_distributed) {
  do_upload_and_download_snapshot<unsigned>(20, TrainPSType_t::Staged, true);
}
TEST(parameter_server_test, long_long_cache_distributed_Adam) {
  HMemCacheConfig hc_config(1, 0.5, 0);
  do_upload_and_download_snapshot<long long>(
      20, TrainPSType_t::Cached, true,  Optimizer_t::Adam, "./", hc_config);
}

TEST(parameter_server_test, unsigned_host_localized) {
  do_upload_and_download_snapshot<unsigned>(20, TrainPSType_t::Staged, false);
}
TEST(parameter_server_test, unsigned_cache_localized_SGD) {
  HMemCacheConfig hc_config(1, 0.5, 0);
  do_upload_and_download_snapshot<unsigned>(
      20, TrainPSType_t::Cached, false,  Optimizer_t::SGD, "./", hc_config);
}

}  // namespace
