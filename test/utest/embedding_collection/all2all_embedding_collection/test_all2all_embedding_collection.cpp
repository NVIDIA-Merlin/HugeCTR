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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/time.h>

#include <numeric>

#include "../embedding_collection_utils.hpp"
#include "HugeCTR/core/hctr_impl/hctr_backend.hpp"
#include "HugeCTR/embedding/all2all_embedding_collection.hpp"
#include "HugeCTR/embedding/embedding.hpp"
#include "HugeCTR/embedding_storage/ragged_static_embedding.hpp"
#include "HugeCTR/include/resource_managers/resource_manager_ext.hpp"
using namespace embedding;

const std::vector<int> device_list = {0, 1};
// embedding parameter
const int batch_size = 32;
const int num_embedding = 5;
const std::vector<int> id_space_list = {0, 1, 2, 3, 4};
const std::vector<int> hotness_list = {10, 10, 10, 10, 10};
const std::vector<Combiner> combiner_list = {Combiner::Sum, Combiner::Average, Combiner::Average,
                                             Combiner::Average, Combiner::Average};
// embedding table parameter
const int num_table = 5;
const std::vector<int> table_ev_size_list = {8, 16, 32, 64, 128};
const std::vector<int> table_min_key_list = {0, 0, 0, 0, 0};
const std::vector<int> table_max_key_list = {100, 100, 100, 100, 100};
// shard parameter
const std::vector<std::vector<int>> shard_matrix = {{-1, 0, -1, 0, -1}, {0, -1, 0, -1, 0}};
// const std::vector<std::vector<int>> shard_matrix = {{0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}};

template <typename key_t, typename offset_t, typename index_t, typename emb_t>
void all2all_embedding_collection_test() {
  auto resource_manager = HugeCTR::ResourceManagerExt::create({device_list}, 0);
  int num_gpus = static_cast<int>(device_list.size());
  EmbeddingCollectionParam ebc_param;
  ebc_param.num_embedding = num_embedding;
  for (int embedding_id = 0; embedding_id < num_embedding; ++embedding_id) {
    EmbeddingParam emb_param;
    emb_param.embedding_id = embedding_id;
    emb_param.id_space = id_space_list[embedding_id];
    emb_param.combiner = combiner_list[embedding_id];
    emb_param.hotness = hotness_list[embedding_id];
    emb_param.ev_size = table_ev_size_list[id_space_list[embedding_id]];
    ebc_param.embedding_params.push_back(std::move(emb_param));
  }
  ebc_param.universal_batch_size = batch_size;
  ebc_param.is_table_first_input = true;
  ebc_param.is_utest = true;
  ebc_param.key_type = HugeCTR::TensorScalarTypeFunc<key_t>::get_type();
  ebc_param.index_type = HugeCTR::TensorScalarTypeFunc<index_t>::get_type();
  ebc_param.offset_type = HugeCTR::TensorScalarTypeFunc<offset_t>::get_type();
  ebc_param.emb_type = HugeCTR::TensorScalarTypeFunc<emb_t>::get_type();

  EmbeddingShardParam shard_param{shard_matrix, TablePlacementStrategy::ModelParallel};

  std::vector<EmbeddingTableParam> table_param_list;
  for (int id = 0; id < num_table; ++id) {
    EmbeddingTableParam table_param;
    table_param.id_space = id;
    table_param.max_vocabulary_size = 5;
    table_param.ev_size = table_ev_size_list[id];
    table_param.min_key = table_min_key_list[id];
    table_param.max_key = table_max_key_list[id];
    HugeCTR::OptParams opt_param;
    opt_param.optimizer = HugeCTR::Optimizer_t::SGD;
    opt_param.lr = 1e-1;
    opt_param.scaler = (ebc_param.emb_type == TensorScalarType::Float16) ? 1024 : 1;
    table_param.opt_param = opt_param;
    table_param_list.push_back(std::move(table_param));
  }

  HugeCTR::OptParams opt_param;
  opt_param.optimizer = HugeCTR::Optimizer_t::SGD;
  opt_param.lr = 1e-1;
  opt_param.scaler = (ebc_param.emb_type == TensorScalarType::Float16) ? 1024 : 1;

  std::vector<std::shared_ptr<core::CoreResourceManager>> core_list;
  std::vector<std::unique_ptr<tf::IAll2AllEmbeddingCollectionSwizzleKey>> swizzle_key_list;
  std::vector<std::unique_ptr<tf::IAll2AllEmbeddingCollectionModelForward>> model_forward_list;
  std::vector<std::unique_ptr<tf::IAll2AllEmbeddingCollectionNetworkForward>> network_forward_list;
  std::vector<std::unique_ptr<tf::IAll2AllEmbeddingCollectionNetworkBackward>>
      network_backward_list;
  std::vector<std::unique_ptr<tf::IAll2AllEmbeddingCollectionModelBackward>> model_backward_list;
  std::vector<std::unique_ptr<IEmbeddingTable>> ebc_table_list;
  for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
    auto core = std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, gpu_id);
    core_list.push_back(core);

    swizzle_key_list.push_back(
        std::move(std::make_unique<tf::All2AllEmbeddingCollectionSwizzleKey>(core)));
    model_forward_list.push_back(
        std::move(std::make_unique<tf::All2AllEmbeddingCollectionModelForward>(core, ebc_param,
                                                                               shard_param)));
    network_forward_list.push_back(
        std::move(std::make_unique<tf::All2AllEmbeddingCollectionNetworkForward>(core, ebc_param,
                                                                                 shard_param)));
    network_backward_list.push_back(
        std::move(std::make_unique<tf::All2AllEmbeddingCollectionNetworkBackward>(core, ebc_param,
                                                                                  shard_param)));
    model_backward_list.push_back(
        std::move(std::make_unique<tf::All2AllEmbeddingCollectionModelBackward>(core, ebc_param,
                                                                                shard_param)));
    ebc_table_list.push_back(std::make_unique<RaggedStaticEmbeddingTable>(
        *resource_manager->get_local_gpu(gpu_id), core, table_param_list, ebc_param, shard_param,
        opt_param));
  }

  std::vector<std::vector<std::vector<key_t>>> key_list;
  std::vector<std::vector<std::vector<offset_t>>> row_lengths;
  auto prepare_input = [&] {
    timeval t1;
    gettimeofday(&t1, NULL);
    srand(t1.tv_usec * t1.tv_sec);
    key_list.clear();
    row_lengths.clear();
    key_list.resize(num_gpus);
    row_lengths.resize(num_gpus);
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      key_list[gpu_id].resize(num_embedding);
      row_lengths[gpu_id].resize(num_embedding);
    }

    int batch_size_per_gpu = ebc_param.universal_batch_size / num_gpus;
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      for (int embedding_id = 0; embedding_id < num_embedding; ++embedding_id) {
        auto &embedding_param = ebc_param.embedding_params[embedding_id];
        int id_space = embedding_param.id_space;
        int hotness = embedding_param.hotness;
        auto &table_param = table_param_list[id_space];
        HCTR_CHECK_HINT(ebc_param.embedding_params[embedding_id].combiner != Combiner::Concat,
                        "sparse embedding does not support concat combiner.");
        for (int b = 0; b < batch_size_per_gpu; ++b) {
          int nnz = rand() % (hotness + 1);
          int gpu_id = b % num_gpus;
          row_lengths[gpu_id][embedding_id].push_back(nnz);
          for (int i = 0; i < nnz; ++i) {
            key_t key = rand() % (table_param.max_key - table_param.min_key) + table_param.min_key;
            key_list[gpu_id][embedding_id].push_back(key);
          }
        }
      }
    }
  };

  int num_iteration = 1;
  for (int iter = 0; iter < num_iteration; ++iter) {
    prepare_input();

    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
      HugeCTR::CudaDeviceContext context(core_list[gpu_id]->get_device_id());

      auto buffer = GetBuffer(core_list[gpu_id]);
      std::vector<core::Tensor> ebc_key_list;
      std::vector<core::Tensor> ebc_row_lengths_list;

      auto copy_gpu_input = [&] {
        for (int embedding_id = 0; embedding_id < num_embedding; ++embedding_id) {
          ebc_key_list.push_back(buffer->reserve(key_list[gpu_id][embedding_id].size(),
                                                 DeviceType::GPU, ebc_param.key_type));
          ebc_row_lengths_list.push_back(buffer->reserve(row_lengths[gpu_id][embedding_id].size(),
                                                         DeviceType::GPU, ebc_param.offset_type));
          buffer->allocate();
          ebc_key_list[embedding_id].copy_from(key_list[gpu_id][embedding_id]);
          ebc_row_lengths_list[embedding_id].copy_from(row_lengths[gpu_id][embedding_id]);
        }
      };
      copy_gpu_input();

      core::Tensor key_all_gather_send_buffer;
      core::Tensor row_lengths_all_gather_send_buffer;
      auto allocate_all_gather_buffer = [&] {
        int64_t num_key_all_gather_send_buffer = 0;
        for (auto &t : ebc_key_list) {
          num_key_all_gather_send_buffer += t.get_num_elements();
        }
        key_all_gather_send_buffer =
            buffer->reserve(num_key_all_gather_send_buffer, DeviceType::GPU, ebc_param.key_type);

        int64_t num_row_lengths_all_gather_send_buffer = 0;
        for (auto &t : ebc_row_lengths_list) {
          num_row_lengths_all_gather_send_buffer += t.get_num_elements();
        }
        row_lengths_all_gather_send_buffer = buffer->reserve(
            num_row_lengths_all_gather_send_buffer, DeviceType::GPU, ebc_param.offset_type);
        buffer->allocate();
      };
      allocate_all_gather_buffer();

      swizzle_key_list[gpu_id]->sparse_forward_per_gpu(ebc_key_list, ebc_row_lengths_list,
                                                       key_all_gather_send_buffer,
                                                       row_lengths_all_gather_send_buffer);

      HCTR_LIB_THROW(cudaStreamSynchronize(core_list[gpu_id]->get_local_gpu()->get_stream()));
      auto print_input_key = [&] {
        std::cout << "gpu_id:" << gpu_id << ",ebc_key_list:\n";
        for (int embedding_id = 0; embedding_id < num_embedding; ++embedding_id) {
          std::vector<key_t> gpu_ebc_key;
          ebc_key_list[embedding_id].to(&gpu_ebc_key);
          print_array(gpu_ebc_key.size(), gpu_ebc_key);
        }

        std::cout << "gpu_id:" << gpu_id << ",ebc_row_lengths:\n";
        for (int embedding_id = 0; embedding_id < num_embedding; ++embedding_id) {
          std::vector<key_t> gpu_ebc_row_length;
          ebc_row_lengths_list[embedding_id].to(&gpu_ebc_row_length);
          print_array(gpu_ebc_row_length.size(), gpu_ebc_row_length);
        }
      };
      print_input_key();

      auto print_all_gather_result = [&] {
        std::vector<key_t> gpu_key_all_gather_send_buffer;
        key_all_gather_send_buffer.to(&gpu_key_all_gather_send_buffer);
        std::vector<offset_t> gpu_row_lengths_all_gather_send_buffer;
        row_lengths_all_gather_send_buffer.to(&gpu_row_lengths_all_gather_send_buffer);

        std::cout << "gpu_id:" << gpu_id << ",gpu_key_all_gather_send_buffer:\n";
        print_array(gpu_key_all_gather_send_buffer.size(), gpu_key_all_gather_send_buffer);
        std::cout << "gpu_row_lengths_all_gather_send_buffer num_elements: "
                  << gpu_row_lengths_all_gather_send_buffer.size() << "\n";
        std::cout << "gpu_id:" << gpu_id << ",gpu_row_lengths_all_gather_send_buffer:\n";
        print_array(gpu_row_lengths_all_gather_send_buffer.size(),
                    gpu_row_lengths_all_gather_send_buffer);
      };
      print_all_gather_result();

      Tensor key_all_gather_recv_buffer;
      Tensor row_lengths_all_gather_recv_buffer;
      auto prepare_all_gather_recv_buffer = [&] {
        std::vector<key_t> cpu_key_all_gather_recv_buffer;
        std::vector<offset_t> cpu_row_lengths_all_gather_recv_buffer;
        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
          for (int embedding_id = 0; embedding_id < num_embedding; ++embedding_id) {
            cpu_key_all_gather_recv_buffer.insert(cpu_key_all_gather_recv_buffer.end(),
                                                  key_list[gpu_id][embedding_id].begin(),
                                                  key_list[gpu_id][embedding_id].end());
            cpu_row_lengths_all_gather_recv_buffer.insert(
                cpu_row_lengths_all_gather_recv_buffer.end(),
                row_lengths[gpu_id][embedding_id].begin(), row_lengths[gpu_id][embedding_id].end());
          }
        }

        key_all_gather_recv_buffer = buffer->reserve(cpu_key_all_gather_recv_buffer.size(),
                                                     DeviceType::GPU, ebc_param.key_type);
        row_lengths_all_gather_recv_buffer = buffer->reserve(
            cpu_row_lengths_all_gather_recv_buffer.size(), DeviceType::GPU, ebc_param.offset_type);
        buffer->allocate();
        key_all_gather_recv_buffer.copy_from(cpu_key_all_gather_recv_buffer);
        row_lengths_all_gather_recv_buffer.copy_from(cpu_row_lengths_all_gather_recv_buffer);
      };
      prepare_all_gather_recv_buffer();

      std::vector<Tensor> emb_vec_model_buffer;
      auto prepare_model_buffer = [&] {
        auto buffer_size_list = model_forward_list[gpu_id]->get_model_comm_buffer_size(batch_size);
        for (size_t buffer_size : buffer_size_list) {
          auto t = buffer->reserve(buffer_size, DeviceType::GPU, ebc_param.emb_type);
          emb_vec_model_buffer.push_back(t);
          std::cout << "buffer_size:" << buffer_size << "\n";
        }
        buffer->allocate();
      };
      prepare_model_buffer();

      int64_t num_model_key, num_model_offsets;
      model_forward_list[gpu_id]->sparse_forward_per_gpu(
          key_all_gather_recv_buffer, row_lengths_all_gather_recv_buffer,
          ebc_table_list[gpu_id].get(), emb_vec_model_buffer, &num_model_key, &num_model_offsets);

      Tensor model_key, model_offsets;
      auto allocate_model_key_and_offsets = [&] {
        model_key = buffer->reserve(num_model_key, DeviceType::GPU, ebc_param.key_type);
        model_offsets =
            buffer->reserve(num_model_offsets, DeviceType::GPU, TensorScalarType::UInt32);
        buffer->allocate();
      };
      allocate_model_key_and_offsets();
      model_forward_list[gpu_id]->copy_model_keys_and_offsets(model_key, model_offsets);

      HCTR_LIB_THROW(cudaStreamSynchronize(core_list[gpu_id]->get_local_gpu()->get_stream()));
      auto print_model_buffer = [&] {
        std::cout << "model buffer:\n";
        for (int dst_gpu_id = 0; dst_gpu_id < num_gpus; ++dst_gpu_id) {
          std::vector<emb_t> gpu_model_buffer;
          emb_vec_model_buffer[dst_gpu_id].to(&gpu_model_buffer);
          std::cout << "dst_gpu_id " << dst_gpu_id << ":";
          print_array(gpu_model_buffer.size(), gpu_model_buffer);
        }
      };
      print_model_buffer();

      auto print_model_key_and_offsets = [&] {
        std::vector<key_t> gpu_model_key;
        model_key.to(&gpu_model_key);
        std::cout << "gpu model key:\n";
        for (auto i : gpu_model_key) {
          std::cout << i << " ";
        }
        std::cout << "\n";

        std::vector<uint32_t> gpu_model_offsets;
        model_offsets.to(&gpu_model_offsets);
        std::cout << "gpu model offsets:\n";
        for (auto i : gpu_model_offsets) {
          std::cout << i << " ";
        }
        std::cout << "\n";
      };
      print_model_key_and_offsets();

      std::vector<Tensor> emb_vec_network_buffer;
      auto prepare_network_buffer = [&] {
        timeval t1;
        gettimeofday(&t1, NULL);
        srand(t1.tv_usec * t1.tv_sec);
        for (int global_gpu_id = 0; global_gpu_id < num_gpus; ++global_gpu_id) {
          int num_elements = 0;
          for (int embedding_id = 0; embedding_id < ebc_param.num_embedding; ++embedding_id) {
            if (shard_param.shard_matrix[global_gpu_id][embedding_id] < 0) continue;
            num_elements +=
                ebc_param.embedding_params[embedding_id].ev_size * batch_size / num_gpus;
          }
          std::vector<emb_t> cpu_network_buffer;
          for (int i = 0; i < num_elements; ++i) {
            float x = (float)rand() / (float)(RAND_MAX);
            cpu_network_buffer.push_back(x);
          }

          emb_vec_network_buffer.push_back(
              buffer->reserve(num_elements, DeviceType::GPU, ebc_param.emb_type));
          buffer->allocate();
          emb_vec_network_buffer.back().copy_from(cpu_network_buffer);
        }
      };
      prepare_network_buffer();

      auto print_network_buffer = [&] {
        for (int dst_gpu_id = 0; dst_gpu_id < num_gpus; ++dst_gpu_id) {
          std::vector<emb_t> gpu_network_buffer;
          emb_vec_network_buffer[dst_gpu_id].to(&gpu_network_buffer);
          std::cout << "dst_gpu_id " << dst_gpu_id << ", num_elements:" << gpu_network_buffer.size()
                    << "\n";
          print_array(gpu_network_buffer.size(), gpu_network_buffer);
        }
      };
      std::cout << "network buffer:\n";
      print_network_buffer();

      std::vector<Tensor> forward_emb_vec;
      auto allocate_forward_emb_vec = [&] {
        for (int embedding_id = 0; embedding_id < ebc_param.num_embedding; ++embedding_id) {
          forward_emb_vec.push_back(buffer->reserve(
              ebc_param.embedding_params[embedding_id].ev_size * batch_size / num_gpus,
              DeviceType::GPU, ebc_param.emb_type));
        }
        buffer->allocate();
      };
      allocate_forward_emb_vec();

      network_forward_list[gpu_id]->sparse_forward_per_gpu(emb_vec_network_buffer,
                                                           ebc_row_lengths_list, forward_emb_vec);
      HCTR_LIB_THROW(cudaStreamSynchronize(core_list[gpu_id]->get_local_gpu()->get_stream()));
      auto print_output_buffer = [&] {
        std::cout << "forward emb result:\n";
        for (auto &t : forward_emb_vec) {
          std::vector<emb_t> gpu_forward_result;
          t.to(&gpu_forward_result);
          print_array(gpu_forward_result.size(), gpu_forward_result);
        }
      };
      print_output_buffer();

      network_backward_list[gpu_id]->backward_per_gpu(forward_emb_vec, ebc_row_lengths_list,
                                                      emb_vec_network_buffer);
      std::cout << "backward network buffer:\n";
      print_network_buffer();

      std::vector<int> num_unique_key_per_table, unique_id_space_list;
      model_backward_list[gpu_id]->sparse_backward_per_gpu(emb_vec_model_buffer, model_key,
                                                           model_offsets, &num_unique_key_per_table,
                                                           &unique_id_space_list);

      std::vector<Tensor> unique_key, grad_emb_vec;
      auto allocate_grad = [&] {
        for (size_t i = 0; i < num_unique_key_per_table.size(); ++i) {
          int id_space = unique_id_space_list[i];
          int ev_size = table_ev_size_list[id_space];
          std::cout << "num_unique_key_per_table:" << num_unique_key_per_table[i] << "\n";
          unique_key.push_back(
              buffer->reserve(num_unique_key_per_table[i], DeviceType::GPU, ebc_param.key_type));
          grad_emb_vec.push_back(buffer->reserve(num_unique_key_per_table[i] * ev_size,
                                                 DeviceType::GPU, TensorScalarType::Float32));
        }
        buffer->allocate();
      };
      allocate_grad();

      model_backward_list[gpu_id]->copy_backward_key_and_emb_vec(unique_key, grad_emb_vec);
      HCTR_LIB_THROW(cudaStreamSynchronize(core_list[gpu_id]->get_local_gpu()->get_stream()));

      auto print_grad = [&] {
        std::cout << "unique_key:";
        for (auto &t : unique_key) {
          std::vector<key_t> gpu_unique_key;
          t.to(&gpu_unique_key);
          print_array(gpu_unique_key.size(), gpu_unique_key);
        }
        std::cout << "grad_emb_vec:";
        for (auto &t : grad_emb_vec) {
          std::vector<float> gpu_grad_emb_vec;
          t.to(&gpu_grad_emb_vec);
          print_array(gpu_grad_emb_vec.size(), gpu_grad_emb_vec);
        }
      };
      print_grad();
    }
  }
}

TEST(all2all_embedding_collection, all2all_embedding_collection_test) {
  all2all_embedding_collection_test<int32_t, int32_t, uint32_t, float>();
}
