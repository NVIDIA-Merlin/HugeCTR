/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "utest/embedding/unified_embedding.hpp"

#include "HugeCTR/include/resource_managers/resource_manager_ext.hpp"

using namespace HugeCTR;
using namespace unified_embedding_test;
namespace {

template <template <typename, typename> typename EmbeddingType, typename Key,
          typename EmbeddingComp>
void unified_embedding_forward(const TestParams &test_param, const std::vector<int> &device_list,
                               const std::vector<size_t> &slot_size_array) {
  test::mpi_init();
  int numprocs = 1;
#ifdef ENABLE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
#endif
  std::vector<std::vector<int>> vvgpu;
  for (int i = 0; i < numprocs; ++i) {
    vvgpu.push_back(device_list);
  }
  const auto &resource_manager = ResourceManagerExt::create(vvgpu, 0);
  size_t total_gpu_count = resource_manager->get_global_gpu_count();
  size_t local_gpu_count = resource_manager->get_local_gpu_count();

  SparseTensors<Key> train_keys;
  SparseTensors<Key> evaluate_keys;
  for (size_t id = 0; id < local_gpu_count; ++id) {
    CudaDeviceContext context(resource_manager->get_local_gpu(id)->get_device_id());
    std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();

    auto push_back_sparse_tensor = [&buf](const std::vector<size_t> &dimension, size_t slot_num,
                                          std::vector<SparseTensor<Key>> &tensor_vec) {
      SparseTensor<Key> tensor;
      buf->reserve(dimension, slot_num, &tensor);
      tensor_vec.push_back(tensor);
    };

    push_back_sparse_tensor(
        {test_param.batch_size_per_gpu * local_gpu_count, test_param.max_nnz_per_sample},
        test_param.slot_num, train_keys);
    push_back_sparse_tensor(
        {test_param.batch_size_per_gpu * local_gpu_count, test_param.max_nnz_per_sample},
        test_param.slot_num, evaluate_keys);
    buf->allocate();
  }

  SparseEmbeddingHashParams embedding_param;
  embedding_param.train_batch_size = test_param.batch_size_per_gpu * total_gpu_count;
  embedding_param.evaluate_batch_size = test_param.batch_size_per_gpu * total_gpu_count;
  embedding_param.max_vocabulary_size_per_gpu = test_param.vocabulary_size;
  embedding_param.embedding_vec_size = test_param.embedding_vec_size;
  embedding_param.max_feature_num = test_param.max_nnz_per_sample;
  embedding_param.slot_num = test_param.slot_num;
  embedding_param.combiner = 0;
  embedding_param.is_data_parallel = true;
  embedding_param.opt_params.optimizer = Optimizer_t::SGD;
  embedding_param.opt_params.lr = test_param.lr;
  embedding_param.opt_params.update_type = Update_t::Local;
  embedding_param.opt_params.hyperparams.sgd.atomic_update = false;
  embedding_param.slot_size_array = slot_size_array;

  std::cout << " train_batch_size:" << embedding_param.train_batch_size << std::endl;
  std::cout << " evaluate_batch_size:" << embedding_param.evaluate_batch_size << std::endl;
  std::cout << " max_vocabulary_size_per_gpu:" << embedding_param.max_vocabulary_size_per_gpu
            << std::endl;
  std::cout << " embedding_vec_size:" << embedding_param.embedding_vec_size << std::endl;
  std::cout << " max_feature_num:" << embedding_param.max_feature_num << std::endl;

  std::unique_ptr<EmbeddingType<Key, EmbeddingComp>> embedding =
      std::make_unique<EmbeddingType<Key, EmbeddingComp>>(train_keys, evaluate_keys,
                                                          embedding_param, resource_manager);

  // prepare input
  SparseTensors<Key> host_input_keys;
  host_input_keys.resize(test_param.train_steps);
  {
    auto buf = GeneralBuffer2<HostAllocator>::create();
    for (size_t train_step_id = 0; train_step_id < test_param.train_steps; ++train_step_id) {
      SparseTensor<Key> tensor;
      buf->reserve({test_param.batch_size_per_gpu * local_gpu_count, test_param.max_nnz_per_sample},
                   test_param.slot_num, &host_input_keys[train_step_id]);
      host_input_keys.push_back(tensor);
    }
    buf->allocate();
    for (size_t train_step_id = 0; train_step_id < test_param.train_steps; ++train_step_id) {
      init_sparse_tensor(host_input_keys[train_step_id], test_param.feature_num_per_sample, false,
                         test_param.fixed_length);
    }
  }

  auto sync_all_gpus = [](const ResourceManager &resource_manager) {
    CudaDeviceContext context;

    size_t local_gpu_count = resource_manager.get_local_gpu_count();
    for (size_t id = 0; id < local_gpu_count; id++) {
      const auto &local_gpu = resource_manager.get_local_gpu(id);
      context.set_device(local_gpu->get_device_id());
      CK_CUDA_THROW_(cudaStreamSynchronize(local_gpu->get_stream()));
    }
  };

  // init and insert
  embedding->init_params();
  for (size_t i = 0; i < test_param.train_steps; ++i) {
    for (size_t id = 0; id < local_gpu_count; ++id) {
      auto &local_gpu = resource_manager->get_local_gpu(id);
      CudaDeviceContext context(local_gpu->get_device_id());
      sparse_tensor_helper::cuda::copy_async(train_keys[id], host_input_keys[i],
                                             cudaMemcpyHostToDevice, local_gpu->get_stream());
    }
    embedding->forward(true);
    sync_all_gpus(*resource_manager);
  }
  if (test_param.reference_check) {
    MESSAGE_("start reference check.");
    EmbeddingCpu<Key, EmbeddingComp> embedding_cpu(embedding_param, total_gpu_count);
    std::vector<Tensor2<EmbeddingComp>> forward_result_from_gpu(
        test_param.train_steps);  // buffer to save gpu embedding result
    {
      std::shared_ptr<GeneralBuffer2<HostAllocator>> host_buff =
          GeneralBuffer2<HostAllocator>::create();
      for (size_t i = 0; i < test_param.train_steps; ++i) {
        host_buff->reserve({embedding_param.train_batch_size, embedding_param.slot_num,
                            embedding_param.embedding_vec_size},
                           &forward_result_from_gpu[i]);
      }
      host_buff->allocate();
    }

    BufferBag buf_bag;
    buf_bag.keys = embedding_cpu.hash_table_key_tensors_.shrink();
    buf_bag.slot_id = embedding_cpu.slot_id_.shrink();
    buf_bag.embedding = embedding_cpu.hash_table_value_tensors_;
    {
      size_t max_voc_size_per_gpu = embedding_param.max_vocabulary_size_per_gpu;

      auto host_blobs_buff = GeneralBuffer2<CudaHostAllocator>::create();

      const size_t local_gpu_count = resource_manager->get_local_gpu_count();

      for (size_t id = 0; id < local_gpu_count; id++) {
        Tensor2<float> tensor;
        host_blobs_buff->reserve({max_voc_size_per_gpu, embedding_param.embedding_vec_size},
                                 &tensor);
        buf_bag.h_value_tensors.push_back(tensor);

        Tensor2<size_t> tensor_slot_id;
        host_blobs_buff->reserve({max_voc_size_per_gpu}, &tensor_slot_id);
        buf_bag.h_slot_id_tensors.push_back(tensor_slot_id);
      }
      host_blobs_buff->allocate();

      CudaDeviceContext context;
      for (size_t id = 0; id < local_gpu_count; id++) {
        context.set_device(resource_manager->get_local_gpu(id)->get_device_id());
        {
          auto uvm_blobs_buff = GeneralBuffer2<CudaManagedAllocator>::create();
          Tensor2<Key> tensor;
          uvm_blobs_buff->reserve({max_voc_size_per_gpu}, &tensor);
          buf_bag.uvm_key_tensor_bags.push_back(tensor.shrink());
          uvm_blobs_buff->allocate();
        }
        {
          auto hbm_blobs_buff = GeneralBuffer2<CudaAllocator>::create();
          Tensor2<size_t> tensor;
          hbm_blobs_buff->reserve({max_voc_size_per_gpu}, &tensor);
          buf_bag.d_value_index_tensors.push_back(tensor);
          hbm_blobs_buff->allocate();
        }
      }
    }
    size_t dump_size;
    embedding->dump_parameters(buf_bag, &dump_size);
    embedding->reset();
    embedding->load_parameters(buf_bag, dump_size);

    embedding_cpu.init(dump_size);

    // run
    for (size_t i = 0; i < test_param.train_steps; ++i) {
      for (size_t id = 0; id < local_gpu_count; ++id) {
        auto &local_gpu = resource_manager->get_local_gpu(id);
        CudaDeviceContext context(local_gpu->get_device_id());
        sparse_tensor_helper::cuda::copy_async(train_keys[id], host_input_keys[i],
                                               cudaMemcpyHostToDevice, local_gpu->get_stream());
      }
      embedding->forward(true);
      sync_all_gpus(*resource_manager);
      embedding->get_forward_results(true, forward_result_from_gpu[i]);
    }

    for (size_t i = 0; i < test_param.train_steps; ++i) {
      MESSAGE_("reference check cpu iter:" + std::to_string(i));
      embedding_cpu.forward(true, host_input_keys[i], host_input_keys[i]);

      ASSERT_TRUE(compare_array(embedding_cpu.embedding_feature_tensors_.get_num_elements(),
                                embedding_cpu.embedding_feature_tensors_.get_ptr(),
                                forward_result_from_gpu[i].get_ptr(), test_param.eposilon));
    }
  }
}
}  // namespace

TEST(unified_embedding_test, test_distributed_embedding_train_multi_gpu_reference_check) {
  TestParams param{10, 1024, 100, 1500, false, 100000, 1, 0.01, true, false, 1e-6};
  unified_embedding_forward<DistributedSlotSparseEmbeddingHash, unsigned int, float>(param, {0, 1},
                                                                                     {});
}

TEST(unified_embedding_test, test_localized_embedding_train_multi_gpu_reference_check) {
  TestParams param{10, 1024, 26, 26, false, 100000, 1, 0.01, true, false, 1e-6};
  unified_embedding_forward<LocalizedSlotSparseEmbeddingHash, unsigned int, float>(param, {0, 1},
                                                                                   {});
}

// TEST(unified_embedding_test, test_localized_embedding_onehot_train_multi_gpu_reference_check) {
//   TestParams param{10, 1024, 2, 2, true, 100000, 1, 0.01, true, false, 1e-6};
//   unified_embedding_forward<LocalizedSlotSparseEmbeddingOneHot, unsigned int, float>(param,
//                                                  {0, 1}, {});
// }
