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
#define private public
#define protected public
#include "HugeCTR/include/embeddings/hybrid_sparse_embedding.hpp"
#include "HugeCTR/include/utils.cuh"
#include "hybrid_embedding/input_generator.hpp"

using namespace HugeCTR;
using namespace hybrid_embedding;

namespace {
// const int numprocs = 8;
// const size_t train_batch_size = 55296;
// const size_t evaluate_batch_size = 55296;
const size_t num_iterations_statistics = 100;
const size_t max_num_frequent_categories = 10;
const double p_dup_max = 1. / 100;
const double max_all_reduce_bandwidth = 1.3e11;
const double max_all_to_all_bandwidth = 1.9e11;
const size_t slot_num = 26;
const size_t embedding_vec_size = 128;
std::vector<size_t> slot_size_array{39884406, 39043,  17289,    7420,    20263,  3,        7120,
                                    1543,     63,     38532951, 2953546, 403346, 10,       2208,
                                    11938,    155,    4,        976,     14,     39979771, 25641295,
                                    39664984, 585935, 12972,    108,     36};
const float scaler = 1.0f;
const float lr = 0.01f;
const DeviceMap::Layout layout = DeviceMap::LOCAL_FIRST;
template <typename dtype>
void print_vector(const std::vector<dtype> &vec, size_t num_elment, const std::string &vec_name) {
  std::cout << "vector name: " << vec_name << ",vector size: " << vec.size() << std::endl;
  for (size_t i = 0; i < std::min(num_elment, vec.size()); ++i) {
    std::cout << vec[i] << ",";
  }
  std::cout << std::endl;
}
template <typename TypeKey, typename TypeFP>
void hybrid_sparse_embedding_construct(const std::vector<int> &device_list, size_t train_batch_size,
                                       size_t evaluate_batch_size, int numprocs,
                                       hybrid_embedding::CommunicationType communication_type,
                                       hybrid_embedding::HybridEmbeddingType hybrid_embedding_type,
                                       const Optimizer_t &optimizer, const Update_t &update_type) {
  // CK_NVML_THROW_(nvmlInit_v2());
  std::vector<std::vector<int>> vvgpu;
  for (int i = 0; i < numprocs; i++) {
    vvgpu.push_back(device_list);
  }
  /*
  const auto &resource_manager = ResourceManager::create(vvgpu, 0, layout);
  */

  DeviceMap device_map(vvgpu, 0, layout);
  std::shared_ptr<ResourceManager> resource_manager(
      new ResourceManager(numprocs, 0, std::move(device_map), (unsigned long long)1234));
  size_t total_gpu_count = resource_manager->get_global_gpu_count();
  size_t local_gpu_count = resource_manager->get_local_gpu_count();
  size_t total_categories = 0;
  for (size_t i = 0; i < slot_size_array.size(); ++i) {
    // slot_size_array[i] = (slot_size_array[i] + 8)/8;
    total_categories += slot_size_array[i];
  }

  HybridEmbeddingConfig<TypeKey> test_config = {
      (size_t)numprocs,
      total_gpu_count,
      slot_num,
      embedding_vec_size,
      (TypeKey)total_categories,
      (TypeKey)0,  // irrelevent here
      1.0          // irrelevent here
  };
  HybridEmbeddingInputGenerator<TypeKey> generator(test_config, slot_size_array, 848484);

  OptHyperParams<TypeFP> hyper_params;
  hyper_params.sgd.atomic_update = true;
  const OptParams<TypeFP> opt_params = {optimizer, lr, hyper_params, update_type, scaler};
  const HybridSparseEmbeddingParams<TypeFP> embedding_params = {
      train_batch_size,
      evaluate_batch_size,
      num_iterations_statistics,
      max_num_frequent_categories * train_batch_size,
      p_dup_max,
      embedding_vec_size,
      slot_num,
      slot_size_array,
      communication_type,
      max_all_reduce_bandwidth,
      max_all_to_all_bandwidth,
      hybrid_embedding_type,
      opt_params};

  Tensors2<TypeKey> train_input_tensors;
  Tensors2<TypeKey> evaluate_input_tensors;
  Tensors2<TypeKey> inits;
  auto initial_input = generator.generate_categorical_input(train_batch_size *
                                                                        num_iterations_statistics);
  auto input = generator.generate_categorical_input(train_batch_size);
  CudaDeviceContext context;

  GpuLearningRateSchedulers lr_scheds;

  for (size_t lgpu = 0; lgpu < local_gpu_count; ++lgpu) {
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
    int cur_device = resource_manager->get_local_gpu(lgpu)->get_device_id();

    context.set_device(cur_device);

    auto stream = resource_manager->get_local_gpu(lgpu)->get_stream();

    Tensor2<TypeKey> tensor0;
    buf->reserve({train_batch_size, slot_num}, &tensor0);
    train_input_tensors.push_back(tensor0);

    Tensor2<TypeKey> tensor1;
    buf->reserve({evaluate_batch_size, slot_num}, &tensor1);
    evaluate_input_tensors.push_back(tensor1);
    Tensor2<TypeKey> tensor2;
    buf->reserve({train_batch_size * num_iterations_statistics, slot_num}, &tensor2);
    inits.push_back(tensor2);
    buf->allocate();
    // print_vector(initial_input, 26, "initial_input");
    // print_vector(input, 26, "input");
    upload_tensor(initial_input, inits[lgpu], stream);
    upload_tensor(input, train_input_tensors[lgpu], stream);

    lr_scheds.emplace_back(new GpuLearningRateScheduler(
          lr, 1, 0, 1, 2.f, 0.f, resource_manager->get_local_gpu(lgpu)));
  }
  std::cout << "hybridEmbdeding" << std::endl;
  std::vector<std::shared_ptr<BufferBlock2<TypeFP>>> placeholder(
      resource_manager->get_local_gpu_count(), NULL);
  std::unique_ptr<HybridSparseEmbedding<TypeKey, TypeFP>> embedding(
      new HybridSparseEmbedding<TypeKey, TypeFP>(train_input_tensors, evaluate_input_tensors,
                                                 embedding_params, placeholder, lr_scheds, false, resource_manager));
  std::cout << "init_model" << std::endl;
  embedding->init_model(inits);
  // std::cout << "forward" << std::endl;
  std::cout << "batch size = " << train_batch_size << std::endl;
  std::cout << "total_categories = " << total_categories
            << ", num_frequent = " << embedding->model_[0].num_frequent << std::endl;
  for (size_t lgpu = 0; lgpu < local_gpu_count; ++lgpu) {
    std::cout << "GPU[" << lgpu << "]"
              << " num_infrequent = "
              << embedding->model_[lgpu].h_infrequent_model_table_offsets[slot_num] << std::endl;
  }

#ifdef ENABLE_PROFILING
  global_profiler.initialize(false, false);
  global_profiler.profiling_dir += std::string("/") + std::to_string(train_batch_size);
  bool finished = false;
  while (1) {
    for (int i = 0; i < int(resource_manager->get_local_gpu_count()); i++) {
      auto device_id = resource_manager->get_local_gpu(i)->get_device_id();
      context.set_device(device_id);
      CK_CUDA_THROW_(cudaDeviceSynchronize());
    }

    finished = global_profiler.iter_check();
    if (finished) {
      break;
    }
#else
  std::chrono::time_point<std::chrono::steady_clock> check;
  for (int j = 0; j < 10000; ++j) {
    for (int i = 0; i < int(resource_manager->get_local_gpu_count()); i++) {
      auto device_id = resource_manager->get_local_gpu(i)->get_device_id();
      context.set_device(device_id);
      CK_CUDA_THROW_(cudaDeviceSynchronize());
    }
    if (j % 100 == 0) {
      auto cost = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now() - check)
                      .count() /
                  1000000.0;
      MESSAGE_(std::string("100 iter time: ") + std::to_string(cost));
      check = std::chrono::steady_clock::now();
    }

#endif
    embedding->forward(true);
    // std::cout << i << ": fwd" << std::endl;
    embedding->backward();
    // std::cout << i << ": bwd" << std::endl;
    embedding->update_params();
    // std::cout << i << ": update" << std::endl;
    // std::cout << "forward, i = " << i << std::endl;
  }
  // std::cout << "backward" << std::endl;
}

}  // namespace

// TEST(hybrid_sparse_embedding_profile, multi_node_uin32_float) {
//   std::vector<size_t> local_batch_sizes{1024, 2048, 3072, 4096, 6144, 8192};
//   // std::vector<size_t> local_batch_sizes{6912};
//   size_t num_procs = 8;
//   for (auto local_batch : local_batch_sizes) {
//     hybrid_sparse_embedding_construct<uint32_t, float>(
//         {0}, local_batch * num_procs, local_batch * num_procs, num_procs,
//         hybrid_embedding::CommunicationType::IB_NVLink,
//         hybrid_embedding::HybridEmbeddingType::Distributed, Optimizer_t::SGD, Update_t::Local);
//   }
// }

// TEST(hybrid_sparse_embedding_profile, single_node_uin32_float) {
//   std::vector<size_t> local_batch_sizes{1024, 2048, 3072, 4096, 6144, 8192};
//   // std::vector<size_t> local_batch_sizes{6912};
//   size_t num_procs = 1;
//   for (auto local_batch : local_batch_sizes) {
//     hybrid_sparse_embedding_construct<uint32_t, float>(
//         {0, 1, 2, 3, 4, 5, 6, 7}, local_batch * 8, local_batch * 8, num_procs,
//         hybrid_embedding::CommunicationType::NVLink_SingleNode,
//         hybrid_embedding::HybridEmbeddingType::Distributed, Optimizer_t::SGD, Update_t::Local);
//   }
// }
