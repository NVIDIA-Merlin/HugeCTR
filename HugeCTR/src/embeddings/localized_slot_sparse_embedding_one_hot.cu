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

#include "HugeCTR/include/embeddings/localized_slot_sparse_embedding_one_hot.hpp"
#include "HugeCTR/include/io/filesystem.hpp"
#include "HugeCTR/include/io/io_utils.hpp"

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

#include <filesystem>
#include <numeric>

namespace HugeCTR {
namespace localized_onehot_filter_keys_kernel {

template <typename TypeKey>
__global__ void select_value_by_slot_id_kernel(const TypeKey *value, size_t num,
                                               TypeKey *filter_value, size_t slot_num_per_gpu,
                                               size_t slot_num, size_t global_id,
                                               size_t global_num) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num) {
    int batch_size = tid / slot_num;
    int slot_id = tid % slot_num;
    if (slot_id % global_num == global_id) {
      int res_slot_id = slot_id / global_num;
      filter_value[batch_size * slot_num_per_gpu + res_slot_id] = __ldg(value + tid);
    }
  }
}
}  // namespace localized_onehot_filter_keys_kernel

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::filter_keys_per_gpu(
    bool is_train, size_t id, size_t global_id, size_t global_num) {
  const SparseTensor<TypeHashKey> &all_gather_key = embedding_data_.get_input_keys(is_train)[id];
  auto &local_gpu = embedding_data_.get_local_gpu(id);
  Tensor2<TypeHashKey> value_tensor = embedding_data_.get_value_tensors(is_train)[id];
  std::shared_ptr<size_t> nnz_ptr = embedding_data_.get_nnz_array(is_train)[id];

  if (all_gather_key.get_dimensions().size() != 2) {
    HCTR_OWN_THROW(Error_t::WrongInput, "localized embedding all gather key dimension != 2");
  }

  size_t batch_size = embedding_data_.embedding_params_.get_batch_size(is_train);
  size_t slot_num_per_gpu = slot_num_per_gpu_[id];
  size_t slot_num = (all_gather_key.rowoffset_count() - 1) / batch_size;

  constexpr size_t block_size = 256;
  size_t grid_size = (all_gather_key.nnz() - 1) / block_size + 1;
  localized_onehot_filter_keys_kernel::
      select_value_by_slot_id_kernel<<<grid_size, block_size, 0, local_gpu.get_stream()>>>(
          all_gather_key.get_value_ptr(), all_gather_key.nnz(), value_tensor.get_ptr(),
          slot_num_per_gpu, slot_num, global_id, global_num);

  *nnz_ptr = (all_gather_key.nnz() / slot_num) * slot_num_per_gpu;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<
    TypeHashKey, TypeEmbeddingComp>::data_to_unique_categories_per_gpu(bool is_train, size_t id) {
  SparseTensor<TypeHashKey> &all_gather_key = embedding_data_.get_input_keys(is_train)[id];
  auto &local_gpu = embedding_data_.get_local_gpu(id);

  if (all_gather_key.get_dimensions().size() != 2) {
    HCTR_OWN_THROW(Error_t::WrongInput, "localized embedding all gather key dimension != 2");
  }

  size_t batch_size = embedding_data_.embedding_params_.get_batch_size(is_train);
  size_t nnz = all_gather_key.nnz();
  size_t slot_num = (all_gather_key.rowoffset_count() - 1) / batch_size;

  data_to_unique_categories(all_gather_key.get_value_ptr(),
                            embedding_data_.embedding_offsets_[id].get_ptr(), slot_num, nnz,
                            local_gpu.get_stream());
}

namespace {

template <typename value_type>
__global__ void upload_value_tensor_kernel(value_type *value_buf, size_t *index_buf,
                                           value_type *dst_tensor, int emb_vec_size, size_t len) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < len) {
    size_t src_offset = gid * emb_vec_size;
    size_t dst_offset = index_buf[gid] * emb_vec_size;
    for (int i = 0; i < emb_vec_size; i++) {
      dst_tensor[dst_offset + i] = value_buf[src_offset + i];
    }
  }
}

}  // namespace

template <typename TypeHashKey, typename TypeEmbeddingComp>
LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::
    LocalizedSlotSparseEmbeddingOneHot(
        const Tensors2<TypeHashKey> &train_row_offsets_tensors,
        const Tensors2<TypeHashKey> &train_value_tensors,
        const std::vector<std::shared_ptr<size_t>> &train_nnz_array,
        const Tensors2<TypeHashKey> &evaluate_row_offsets_tensors,
        const Tensors2<TypeHashKey> &evaluate_value_tensors,
        const std::vector<std::shared_ptr<size_t>> &evaluate_nnz_array,
        const SparseEmbeddingHashParams &embedding_params,
        const std::shared_ptr<ResourceManager> &resource_manager)
    : embedding_data_(train_row_offsets_tensors, train_value_tensors, train_nnz_array,
                      evaluate_row_offsets_tensors, evaluate_value_tensors, evaluate_nnz_array,
                      Embedding_t::LocalizedSlotSparseEmbeddingOneHot, embedding_params,
                      resource_manager),
      slot_size_array_(embedding_params.slot_size_array) {
  embedding_data_.embedding_params_.is_data_parallel =
      false;  // this ctor is only used for embedding plugin
  try {
    max_vocabulary_size_ = 0;
    for (size_t slot_size : slot_size_array_) {
      max_vocabulary_size_ += slot_size;
    }

    max_vocabulary_size_per_gpu_ =
        cal_max_voc_size_per_gpu(slot_size_array_, embedding_data_.get_resource_manager());

    HCTR_LOG_S(INFO, ROOT) << "max_vocabulary_size_per_gpu_=" << max_vocabulary_size_per_gpu_
                           << std::endl;

    CudaDeviceContext context;
    for (size_t id = 0; id < embedding_data_.get_resource_manager().get_local_gpu_count(); id++) {
      context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

      size_t gid = embedding_data_.get_local_gpu(id).get_global_id();
      size_t slot_num_per_gpu =
          embedding_data_.embedding_params_.slot_num /
              embedding_data_.get_resource_manager().get_global_gpu_count() +
          ((gid < embedding_data_.embedding_params_.slot_num %
                      embedding_data_.get_resource_manager().get_global_gpu_count())
               ? 1
               : 0);
      slot_num_per_gpu_.push_back(slot_num_per_gpu);

      // new GeneralBuffer objects
      const std::shared_ptr<GeneralBuffer2<CudaAllocator>> &buf = embedding_data_.get_buffer(id);

      // new hash table value vectors
      {
        const std::shared_ptr<BufferBlock2<float>> &block = buf->create_block<float>();
        Tensors2<float> tensors;
        for (size_t i = 0; i < slot_size_array_.size(); i++) {
          if ((i % embedding_data_.get_resource_manager().get_global_gpu_count()) == gid) {
            Tensor2<float> tensor;
            block->reserve(
                {slot_size_array_[i], embedding_data_.embedding_params_.embedding_vec_size},
                &tensor);
            tensors.push_back(tensor);
          }
        }
        value_table_tensors_.push_back(tensors);
        hash_table_value_tensors_.push_back(block->as_tensor());
      }

      // list of top categories, from single iteration worth of data, so max size is same as
      // hash_table_value_index_ array
      {
        HCTR_LOG_S(INFO, WORLD) << "Initializing size_top_categories_ and top_categories.."
                                << std::endl;
        Tensor2<size_t> tensor;
        buf->reserve({1, embedding_data_.embedding_params_.get_universal_batch_size() *
                             embedding_data_.embedding_params_.max_feature_num},
                     &tensor);
        size_top_categories_.push_back(0);
        top_categories_.push_back(tensor);
        // HCTR_LOG_S(INFO, WORLD) << "top_categories size : " << Base::get_universal_batch_size() *
        // Base::get_max_feature_num() << std::endl;
      }

      // new hash table value_index that get() from HashTable
      {
        Tensor2<size_t> tensor;
        buf->reserve({1, embedding_data_.embedding_params_.get_universal_batch_size() *
                             embedding_data_.embedding_params_.max_feature_num},
                     &tensor);
        hash_value_index_tensors_.push_back(tensor);
      }

      // new embedding features reduced by hash table values(results of forward)
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve(
            {embedding_data_.embedding_params_.get_universal_batch_size() * slot_num_per_gpu,
             embedding_data_.embedding_params_.embedding_vec_size},
            &tensor);
        embedding_feature_tensors_.push_back(tensor);
      }

      // new wgrad used by backward
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({embedding_data_.embedding_params_.get_batch_size(true) * slot_num_per_gpu,
                      embedding_data_.embedding_params_.embedding_vec_size},
                     &tensor);
        wgrad_tensors_.push_back(tensor);
      }

      // new optimizer params used by update_params
      switch (embedding_data_.embedding_params_.opt_params.optimizer) {
        case Optimizer_t::SGD:
          break;

        default:
          throw std::runtime_error(
              std::string("[HCDEBUG][ERROR] Runtime error: Invalid optimizer type\n"));
      }

      // the tenosrs for storing slot ids
      // TODO: init to -1 ?
      {
        Tensor2<size_t> tensor;
        buf->reserve({max_vocabulary_size_per_gpu_, 1}, &tensor);
        hash_table_slot_id_tensors_.push_back(tensor);
      }

      // temp tensors for all2all
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({embedding_data_.get_universal_batch_size_per_gpu() *
                          embedding_data_.embedding_params_.slot_num,
                      embedding_data_.embedding_params_.embedding_vec_size},
                     &tensor);
        all2all_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({embedding_data_.embedding_params_.get_universal_batch_size() *
                          embedding_data_.embedding_params_.slot_num,
                      embedding_data_.embedding_params_.embedding_vec_size},
                     &tensor);
        utest_forward_temp_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({embedding_data_.get_batch_size_per_gpu(true) *
                          embedding_data_.embedding_params_.slot_num,
                      embedding_data_.embedding_params_.embedding_vec_size},
                     &tensor);
        utest_all2all_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({embedding_data_.get_batch_size_per_gpu(true) *
                          embedding_data_.embedding_params_.slot_num,
                      embedding_data_.embedding_params_.embedding_vec_size},
                     &tensor);
        utest_reorder_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({embedding_data_.embedding_params_.get_batch_size(true) *
                          embedding_data_.embedding_params_.slot_num,
                      embedding_data_.embedding_params_.embedding_vec_size},
                     &tensor);
        utest_backward_temp_tensors_.push_back(tensor);
      }
      {
        Tensor2<uint32_t> tensor;
        buf->reserve({1, slot_num_per_gpu}, &tensor);
        mapping_offsets_per_gpu_tensors_.push_back(tensor);
      }

// init GenenralBuffers to do real allocation
#ifndef NDEBUG
      HCTR_LOG_S(DEBUG, WORLD) << " max_feature_num_:"
                               << embedding_data_.embedding_params_.max_feature_num << std::endl;
#endif

    }  // end of for(int id = 0; id < embedding_data_.get_local_gpu_count(); id++)

#pragma omp parallel num_threads(embedding_data_.get_resource_manager().get_local_gpu_count())
    {
      size_t id = omp_get_thread_num();
      CudaDeviceContext context(embedding_data_.get_local_gpu(id).get_device_id());
      embedding_data_.get_buffer(id)->allocate();
      HCTR_LIB_THROW(cudaStreamSynchronize(embedding_data_.get_local_gpu(id).get_stream()));
    }

    // get the mapping table between local value_index and input value_index
    for (size_t id = 0; id < embedding_data_.get_resource_manager().get_local_gpu_count(); id++) {
      context.set_device(embedding_data_.get_local_gpu(id).get_device_id());
      uint32_t slot_sizes_prefix_sum = 0;
      uint32_t slot_sizes_prefix_sum_local = 0;
      int slot_num = 0;
      for (size_t i = 0; i < slot_size_array_.size(); i++) {
        size_t global_id = embedding_data_.get_local_gpu(id).get_global_id();
        size_t slot_size = slot_size_array_[i];
        if (i % embedding_data_.get_resource_manager().get_global_gpu_count() == global_id) {
          uint32_t mapping_offset = slot_sizes_prefix_sum - slot_sizes_prefix_sum_local;
          HCTR_LIB_THROW(cudaMemcpy(&((mapping_offsets_per_gpu_tensors_[id].get_ptr())[slot_num]),
                                    &mapping_offset, sizeof(uint32_t), cudaMemcpyHostToDevice));
          slot_sizes_prefix_sum_local += slot_size;
          slot_num++;
        }
        slot_sizes_prefix_sum += slot_size;
      }
    }

    // Check whether the P2P access can be enabled
    if (embedding_data_.get_resource_manager().get_local_gpu_count() > 1 &&
        !embedding_data_.get_resource_manager().all_p2p_enabled()) {
      throw std::runtime_error(
          std::string("[HCDEBUG][ERROR] Runtime error: Localized_slot_sparse_embedding_one_hot "
                      "cannot be used on machine without GPU peer2peer access support. \n"));
    }
#ifdef ENABLE_MPI
    {
      int num_processor;
      HCTR_MPI_THROW(MPI_Comm_size(MPI_COMM_WORLD, &num_processor));
      if (num_processor > 1) {
        throw std::runtime_error(
            std::string("[HCDEBUG][ERROR] Runtime error: Localized_slot_sparse_embedding_one_hot "
                        "cannot support multi-node currently. \n"));
      }
    }
#endif

    std::shared_ptr<GeneralBuffer2<CudaManagedAllocator>> unified_buf =
        GeneralBuffer2<CudaManagedAllocator>::create();
    unified_buf->reserve({embedding_data_.get_resource_manager().get_local_gpu_count()},
                         &train_embedding_features_);
    unified_buf->reserve({embedding_data_.get_resource_manager().get_local_gpu_count()},
                         &evaluate_embedding_features_);
    unified_buf->allocate();

    for (size_t id = 0; id < embedding_data_.get_resource_manager().get_local_gpu_count(); id++) {
      train_embedding_features_.get_ptr()[id] =
          embedding_data_.get_output_tensors(true)[id].get_ptr();
      evaluate_embedding_features_.get_ptr()[id] =
          embedding_data_.get_output_tensors(false)[id].get_ptr();
    }

  } catch (const std::runtime_error &rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }

  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::
    LocalizedSlotSparseEmbeddingOneHot(const SparseTensors<TypeHashKey> &train_keys,
                                       const SparseTensors<TypeHashKey> &evaluate_keys,
                                       const SparseEmbeddingHashParams &embedding_params,
                                       const std::shared_ptr<ResourceManager> &resource_manager)
    : embedding_data_(Embedding_t::LocalizedSlotSparseEmbeddingOneHot, train_keys, evaluate_keys,
                      embedding_params, resource_manager),
      slot_size_array_(embedding_params.slot_size_array) {
  try {
    max_vocabulary_size_ = 0;
    for (size_t slot_size : slot_size_array_) {
      max_vocabulary_size_ += slot_size;
    }

    max_vocabulary_size_per_gpu_ =
        cal_max_voc_size_per_gpu(slot_size_array_, embedding_data_.get_resource_manager());

    HCTR_LOG_S(INFO, ROOT) << "max_vocabulary_size_per_gpu_=" << max_vocabulary_size_per_gpu_
                           << std::endl;

    CudaDeviceContext context;
    for (size_t id = 0; id < embedding_data_.get_resource_manager().get_local_gpu_count(); id++) {
      context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

      size_t gid = embedding_data_.get_local_gpu(id).get_global_id();
      size_t slot_num_per_gpu =
          embedding_data_.embedding_params_.slot_num /
              embedding_data_.get_resource_manager().get_global_gpu_count() +
          ((gid < embedding_data_.embedding_params_.slot_num %
                      embedding_data_.get_resource_manager().get_global_gpu_count())
               ? 1
               : 0);
      slot_num_per_gpu_.push_back(slot_num_per_gpu);

      // new GeneralBuffer objects
      const std::shared_ptr<GeneralBuffer2<CudaAllocator>> &buf = embedding_data_.get_buffer(id);

      // new hash table value vectors
      {
        const std::shared_ptr<BufferBlock2<float>> &block = buf->create_block<float>();
        Tensors2<float> tensors;
        for (size_t i = 0; i < slot_size_array_.size(); i++) {
          if ((i % embedding_data_.get_resource_manager().get_global_gpu_count()) == gid) {
            Tensor2<float> tensor;
            block->reserve(
                {slot_size_array_[i], embedding_data_.embedding_params_.embedding_vec_size},
                &tensor);
            tensors.push_back(tensor);
          }
        }
        value_table_tensors_.push_back(tensors);
        hash_table_value_tensors_.push_back(block->as_tensor());
      }
      {
        Tensor2<TypeHashKey> tensor;
        buf->reserve({embedding_data_.embedding_params_.get_batch_size(true),
                      embedding_data_.embedding_params_.max_feature_num},
                     &tensor);
        embedding_data_.train_value_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeHashKey> tensor;
        buf->reserve({embedding_data_.embedding_params_.get_batch_size(false),
                      embedding_data_.embedding_params_.max_feature_num},
                     &tensor);
        embedding_data_.evaluate_value_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeHashKey> tensor;
        buf->reserve({embedding_data_.embedding_params_.get_batch_size(true) *
                          embedding_data_.embedding_params_.slot_num +
                      1},
                     &tensor);
        embedding_data_.train_row_offsets_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeHashKey> tensor;
        buf->reserve({embedding_data_.embedding_params_.get_batch_size(false) *
                          embedding_data_.embedding_params_.slot_num +
                      1},
                     &tensor);
        embedding_data_.evaluate_row_offsets_tensors_.push_back(tensor);
      }
      { embedding_data_.train_nnz_array_.push_back(std::make_shared<size_t>(0)); }
      { embedding_data_.evaluate_nnz_array_.push_back(std::make_shared<size_t>(0)); }

      // list of top categories, from single iteration worth of data, so max size is same as
      {
        HCTR_LOG_S(INFO, WORLD) << "Initializing size_top_categories_ and top_categories.."
                                << std::endl;
        Tensor2<size_t> tensor;
        buf->reserve({1, embedding_data_.embedding_params_.get_universal_batch_size() *
                             embedding_data_.embedding_params_.max_feature_num},
                     &tensor);
        size_top_categories_.push_back(0);
        top_categories_.push_back(tensor);
      }

      // new hash table value_index that get() from HashTable
      {
        Tensor2<size_t> tensor;
        buf->reserve({1, embedding_data_.embedding_params_.get_universal_batch_size() *
                             embedding_data_.embedding_params_.max_feature_num},
                     &tensor);
        hash_value_index_tensors_.push_back(tensor);
      }

      // new embedding features reduced by hash table values(results of forward)
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve(
            {embedding_data_.embedding_params_.get_universal_batch_size() * slot_num_per_gpu,
             embedding_data_.embedding_params_.embedding_vec_size},
            &tensor);
        embedding_feature_tensors_.push_back(tensor);
      }

      // new wgrad used by backward
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({embedding_data_.embedding_params_.get_batch_size(true) * slot_num_per_gpu,
                      embedding_data_.embedding_params_.embedding_vec_size},
                     &tensor);
        wgrad_tensors_.push_back(tensor);
      }

      // new optimizer params used by update_params
      switch (embedding_data_.embedding_params_.opt_params.optimizer) {
        case Optimizer_t::SGD:
          break;

        default:
          throw std::runtime_error(
              std::string("[HCDEBUG][ERROR] Runtime error: Invalid optimizer type\n"));
      }

      // the tenosrs for storing slot ids
      // TODO: init to -1 ?
      {
        Tensor2<size_t> tensor;
        buf->reserve({max_vocabulary_size_per_gpu_, 1}, &tensor);
        hash_table_slot_id_tensors_.push_back(tensor);
      }

      // temp tensors for all2all
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({embedding_data_.get_universal_batch_size_per_gpu() *
                          embedding_data_.embedding_params_.slot_num,
                      embedding_data_.embedding_params_.embedding_vec_size},
                     &tensor);
        all2all_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({embedding_data_.embedding_params_.get_universal_batch_size() *
                          embedding_data_.embedding_params_.slot_num,
                      embedding_data_.embedding_params_.embedding_vec_size},
                     &tensor);
        utest_forward_temp_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({embedding_data_.get_batch_size_per_gpu(true) *
                          embedding_data_.embedding_params_.slot_num,
                      embedding_data_.embedding_params_.embedding_vec_size},
                     &tensor);
        utest_all2all_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({embedding_data_.get_batch_size_per_gpu(true) *
                          embedding_data_.embedding_params_.slot_num,
                      embedding_data_.embedding_params_.embedding_vec_size},
                     &tensor);
        utest_reorder_tensors_.push_back(tensor);
      }
      {
        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({embedding_data_.embedding_params_.get_batch_size(true) *
                          embedding_data_.embedding_params_.slot_num,
                      embedding_data_.embedding_params_.embedding_vec_size},
                     &tensor);
        utest_backward_temp_tensors_.push_back(tensor);
      }
      {
        Tensor2<uint32_t> tensor;
        buf->reserve({1, slot_num_per_gpu}, &tensor);
        mapping_offsets_per_gpu_tensors_.push_back(tensor);
      }

// init GenenralBuffers to do real allocation
#ifndef NDEBUG
      HCTR_LOG_S(DEBUG, WORLD) << " max_feature_num_:"
                               << embedding_data_.embedding_params_.max_feature_num << std::endl;
#endif

    }  // end of for(int id = 0; id < embedding_data_.get_local_gpu_count(); id++)

#pragma omp parallel for num_threads(embedding_data_.get_resource_manager().get_local_gpu_count())
    for (size_t id = 0; id < embedding_data_.get_resource_manager().get_local_gpu_count(); ++id) {
      CudaDeviceContext context(embedding_data_.get_local_gpu(id).get_device_id());
      embedding_data_.get_buffer(id)->allocate();

      // filling rowoffset and slot_size_array
      cudaStream_t stream = embedding_data_.get_local_gpu(id).get_stream();
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    }

    {
      std::vector<TypeHashKey> embedding_offsets;
      TypeHashKey slot_sizes_prefix_sum = 0;
      for (size_t i = 0; i < slot_size_array_.size(); i++) {
        embedding_offsets.push_back(slot_sizes_prefix_sum);
        slot_sizes_prefix_sum += slot_size_array_[i];
      }
      for (size_t id = 0; id < embedding_data_.get_resource_manager().get_local_gpu_count(); ++id) {
        CudaDeviceContext context(embedding_data_.get_local_gpu(id).get_device_id());

        HCTR_LIB_THROW(
            cudaMemcpy(embedding_data_.embedding_offsets_[id].get_ptr(), embedding_offsets.data(),
                       embedding_offsets.size() * sizeof(TypeHashKey), cudaMemcpyHostToDevice));

        size_t slot_num_per_gpu = slot_num_per_gpu_[id];
        {
          std::vector<TypeHashKey> rowoffset_host(
              embedding_data_.embedding_params_.get_batch_size(true) *
                  embedding_data_.embedding_params_.slot_num +
              1);
          std::iota(rowoffset_host.begin(), rowoffset_host.end(), 0);
          HCTR_LIB_THROW(cudaMemcpy(
              embedding_data_.train_row_offsets_tensors_[id].get_ptr(), rowoffset_host.data(),
              rowoffset_host.size() * sizeof(TypeHashKey), cudaMemcpyHostToDevice));
        }
        {
          std::vector<TypeHashKey> rowoffset_host(
              embedding_data_.embedding_params_.get_batch_size(false) *
                  embedding_data_.embedding_params_.slot_num +
              1);
          std::iota(rowoffset_host.begin(), rowoffset_host.end(), 0);
          HCTR_LIB_THROW(cudaMemcpy(
              embedding_data_.evaluate_row_offsets_tensors_[id].get_ptr(), rowoffset_host.data(),
              rowoffset_host.size() * sizeof(TypeHashKey), cudaMemcpyHostToDevice));
        }
      }
    }

    // get the mapping table between local value_index and input value_index
    for (size_t id = 0; id < embedding_data_.get_resource_manager().get_local_gpu_count(); id++) {
      context.set_device(embedding_data_.get_local_gpu(id).get_device_id());
      uint32_t slot_sizes_prefix_sum = 0;
      uint32_t slot_sizes_prefix_sum_local = 0;
      int slot_num = 0;
      for (size_t i = 0; i < slot_size_array_.size(); i++) {
        size_t global_id = embedding_data_.get_local_gpu(id).get_global_id();
        size_t slot_size = slot_size_array_[i];
        if (i % embedding_data_.get_resource_manager().get_global_gpu_count() == global_id) {
          uint32_t mapping_offset = slot_sizes_prefix_sum - slot_sizes_prefix_sum_local;
          HCTR_LIB_THROW(cudaMemcpy(&((mapping_offsets_per_gpu_tensors_[id].get_ptr())[slot_num]),
                                    &mapping_offset, sizeof(uint32_t), cudaMemcpyHostToDevice));
          slot_sizes_prefix_sum_local += slot_size;
          slot_num++;
        }
        slot_sizes_prefix_sum += slot_size;
      }
    }

    // Check whether the P2P access can be enabled
    if (embedding_data_.get_resource_manager().get_local_gpu_count() > 1 &&
        !embedding_data_.get_resource_manager().all_p2p_enabled()) {
      throw std::runtime_error(
          "[HCDEBUG][ERROR] Runtime error: Localized_slot_sparse_embedding_one_hot cannot be used "
          "on machine without GPU peer2peer access support.\n");
    }
#ifdef ENABLE_MPI
    {
      int num_processor;
      HCTR_MPI_THROW(MPI_Comm_size(MPI_COMM_WORLD, &num_processor));
      if (num_processor > 1) {
        throw std::runtime_error(
            "[HCDEBUG][ERROR] Runtime error: Localized_slot_sparse_embedding_one_hot cannot "
            "support multi-node currently.\n");
      }
    }
#endif

    std::shared_ptr<GeneralBuffer2<CudaManagedAllocator>> unified_buf =
        GeneralBuffer2<CudaManagedAllocator>::create();
    unified_buf->reserve({embedding_data_.get_resource_manager().get_local_gpu_count()},
                         &train_embedding_features_);
    unified_buf->reserve({embedding_data_.get_resource_manager().get_local_gpu_count()},
                         &evaluate_embedding_features_);
    unified_buf->allocate();

    for (size_t id = 0; id < embedding_data_.get_resource_manager().get_local_gpu_count(); id++) {
      train_embedding_features_.get_ptr()[id] =
          embedding_data_.get_output_tensors(true)[id].get_ptr();
      evaluate_embedding_features_.get_ptr()[id] =
          embedding_data_.get_output_tensors(false)[id].get_ptr();
    }

  } catch (const std::runtime_error &rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }

  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::load_parameters(
    std::string sparse_model) {
  const std::string key_file(sparse_model + "/key");
  const std::string slot_file(sparse_model + "/slot_id");
  const std::string vec_file(sparse_model + "/emb_vector");

  auto fs = FileSystemBuilder::build_unique_by_path(sparse_model);

  size_t key_file_size_in_byte = fs->get_file_size(key_file);
  size_t slot_file_size_in_byte = fs->get_file_size(slot_file);
  size_t vec_file_size_in_byte = fs->get_file_size(vec_file);

  size_t key_size = sizeof(long long);
  size_t slot_size = sizeof(size_t);
  size_t vec_size = sizeof(float) * embedding_data_.embedding_params_.embedding_vec_size;
  size_t key_num = key_file_size_in_byte / key_size;
  size_t slot_num = slot_file_size_in_byte / slot_size;
  size_t vec_num = vec_file_size_in_byte / vec_size;

  if (key_num != vec_num || key_file_size_in_byte % key_size != 0 ||
      vec_file_size_in_byte % vec_size != 0 || key_num != slot_num ||
      slot_file_size_in_byte % slot_size != 0) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Error: file size is not correct");
  }

  auto blobs_buff = GeneralBuffer2<CudaHostAllocator>::create();

  Tensor2<TypeHashKey> keys;
  blobs_buff->reserve({key_num}, &keys);

  Tensor2<size_t> slot_id;
  blobs_buff->reserve({slot_num}, &slot_id);

  Tensor2<float> embeddings;
  blobs_buff->reserve({vec_num, embedding_data_.embedding_params_.embedding_vec_size}, &embeddings);

  blobs_buff->allocate();

  TypeHashKey *key_ptr = keys.get_ptr();
  size_t *slot_id_ptr = slot_id.get_ptr();
  float *embedding_ptr = embeddings.get_ptr();

  if (std::is_same<TypeHashKey, long long>::value) {
    fs->read(key_file, reinterpret_cast<char *>(key_ptr), key_file_size_in_byte, 0);
  } else {
    std::vector<long long> i64_key_vec(key_num, 0);
    fs->read(key_file, reinterpret_cast<char *>(i64_key_vec.data()), key_file_size_in_byte, 0);
    std::transform(i64_key_vec.begin(), i64_key_vec.end(), key_ptr,
                   [](long long key) { return static_cast<unsigned>(key); });
  }
  fs->read(slot_file, reinterpret_cast<char *>(slot_id_ptr), slot_file_size_in_byte, 0);
  fs->read(vec_file, reinterpret_cast<char *>(embedding_ptr), vec_file_size_in_byte, 0);

  load_parameters(keys, slot_id, embeddings, key_num,
                  embedding_data_.embedding_params_.embedding_vec_size, hash_table_value_tensors_,
                  slot_size_array_, mapping_offsets_per_gpu_tensors_);
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::load_parameters(
    BufferBag &buf_bag, size_t num) {
  const TensorBag2 &keys_bag = buf_bag.keys;
  const TensorBag2 &slot_id_bag = buf_bag.slot_id;
  const Tensor2<float> &embeddings = buf_bag.embedding;
  Tensor2<TypeHashKey> keys = Tensor2<TypeHashKey>::stretch_from(keys_bag);
  Tensor2<size_t> slot_id = Tensor2<size_t>::stretch_from(slot_id_bag);

  load_parameters(keys, slot_id, embeddings, num,
                  embedding_data_.embedding_params_.embedding_vec_size, hash_table_value_tensors_,
                  slot_size_array_, mapping_offsets_per_gpu_tensors_);
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::load_parameters(
    const Tensor2<TypeHashKey> &keys, const Tensor2<size_t> &slot_id,
    const Tensor2<float> &embeddings, size_t num, size_t embedding_vec_size,
    Tensors2<float> &hash_table_value_tensors, const std::vector<size_t> &slot_sizes,
    const Tensors2<uint32_t> &mapping_offsets_per_gpu_tensors) {
  if (num == 0) return;

  CudaDeviceContext context;
  if (keys.get_dimensions()[0] < num || embeddings.get_dimensions()[0] < num) {
    HCTR_OWN_THROW(Error_t::WrongInput, "The rows of keys and embeddings are not consistent.");
  }

  const TypeHashKey *key_ptr = keys.get_ptr();
  const size_t *slot_id_ptr = slot_id.get_ptr();
  const float *embedding_ptr = embeddings.get_ptr();

  // define size
  size_t local_gpu_count = embedding_data_.get_resource_manager().get_local_gpu_count();
  size_t chunk_size = 1000;
  size_t tile_size = 1;  // must be 1, because we need to cal (key&local_gpu_count) to decide
                         // gpu_id for each <key,value>
  size_t hash_table_value_tile_size = tile_size * embedding_vec_size;
  size_t hash_table_value_tile_size_in_B = hash_table_value_tile_size * sizeof(float);
  size_t hash_table_value_chunk_size = hash_table_value_tile_size * chunk_size;
  size_t hash_table_value_chunk_size_in_B = hash_table_value_chunk_size * sizeof(float);
  size_t total_gpu_count = embedding_data_.get_resource_manager().get_global_gpu_count();

  // CAUSION: can not decide how many values for each GPU, so need to allocate enough memory for
  // each GPU allocate CPU/GPU memory for value/index chunk
  std::unique_ptr<float *[]> h_hash_table_value_chunk_per_gpu(new float *[local_gpu_count]);
  for (size_t id = 0; id < local_gpu_count; id++) {
    HCTR_LIB_THROW(
        cudaMallocHost(&h_hash_table_value_chunk_per_gpu[id], hash_table_value_chunk_size_in_B));
  }
  std::unique_ptr<float *[]> d_hash_table_value_chunk_per_gpu(new float *[local_gpu_count]);
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());
    HCTR_LIB_THROW(
        cudaMalloc(&d_hash_table_value_chunk_per_gpu[id], hash_table_value_chunk_size_in_B));
  }
  std::unique_ptr<size_t *[]> h_hash_table_index_chunk_per_gpu(new size_t *[local_gpu_count]);
  for (size_t id = 0; id < local_gpu_count; id++) {
    HCTR_LIB_THROW(
        cudaMallocHost(&h_hash_table_index_chunk_per_gpu[id], chunk_size * sizeof(size_t)));
  }
  std::unique_ptr<size_t *[]> d_hash_table_index_chunk_per_gpu(new size_t *[local_gpu_count]);
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());
    HCTR_LIB_THROW(cudaMalloc(&d_hash_table_index_chunk_per_gpu[id], chunk_size * sizeof(size_t)));
  }

  std::unique_ptr<size_t[]> tile_counter_in_chunk_per_gpu(new size_t[local_gpu_count]);
  memset(tile_counter_in_chunk_per_gpu.get(), 0, sizeof(size_t) * local_gpu_count);

  // The vector that store the relationship between slot_id and slot order on the specific GPU
  std::vector<size_t> local_slot_id(slot_sizes.size());
  std::vector<size_t> local_slot_num(local_gpu_count, 0);
  for (size_t i = 0; i < slot_sizes.size(); i++) {
    size_t gid = i % total_gpu_count;  // global GPU ID
    size_t id = embedding_data_.get_resource_manager().get_gpu_local_id_from_global_id(
        gid);  // local GPU ID (not gpudevice id)
    int dst_rank =
        embedding_data_.get_resource_manager().get_process_id_from_gpu_global_id(gid);  // node id
    if (embedding_data_.get_resource_manager().get_process_id() == dst_rank) {
      local_slot_id[i] = local_slot_num[id];
      local_slot_num[id]++;
    }
  }

  // Host buffer to keep mapping_offset
  std::vector<uint32_t *> h_mapping_offsets_per_gpu_tensors(local_gpu_count);
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());
    HCTR_LIB_THROW(cudaMallocHost(&h_mapping_offsets_per_gpu_tensors[id],
                                  local_slot_num[id] * sizeof(uint32_t)));
    // Copy the mapping offset from GPU to Host
    HCTR_LIB_THROW(cudaMemcpyAsync(h_mapping_offsets_per_gpu_tensors[id],
                                   mapping_offsets_per_gpu_tensors[id].get_ptr(),
                                   local_slot_num[id] * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                   embedding_data_.get_local_gpu(id).get_stream()));
  }

  // sync wait
  functors_.sync_all_gpus(embedding_data_.get_resource_manager());

  // do upload
  const size_t loop_num = num / chunk_size;
  HCTR_LOG_S(INFO, ROOT) << "Start to upload embedding table file to GPUs, total loop_num: "
                         << loop_num << std::endl;
  for (size_t i = 0; i < loop_num; i++) {
    float *value_dst_buf;
    size_t *tensor_index_dst_buf;
    for (size_t k = 0; k < chunk_size; k++) {  // process a tile in each loop
      size_t slot_id = slot_id_ptr[i * chunk_size + k];
      size_t gid = slot_id % total_gpu_count;  // global GPU ID
      size_t id = embedding_data_.get_resource_manager().get_gpu_local_id_from_global_id(
          gid);  // local GPU ID (not gpudevice id)
      int dst_rank =
          embedding_data_.get_resource_manager().get_process_id_from_gpu_global_id(gid);  // node id

      if (embedding_data_.get_resource_manager().get_process_id() == dst_rank) {
        TypeHashKey tile_key = key_ptr[i * chunk_size + k];
        size_t tensor_index =
            tile_key - (h_mapping_offsets_per_gpu_tensors[id][local_slot_id[slot_id]]);

        // memcpy hash_table_value to corresponding GPU
        value_dst_buf = h_hash_table_value_chunk_per_gpu[id] +
                        tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
        memcpy(value_dst_buf, embedding_ptr + (i * chunk_size + k) * embedding_vec_size,
               hash_table_value_tile_size_in_B);

        tensor_index_dst_buf =
            h_hash_table_index_chunk_per_gpu[id] + tile_counter_in_chunk_per_gpu[id];
        *tensor_index_dst_buf = tensor_index;
        tile_counter_in_chunk_per_gpu[id] += 1;
      } else {
        continue;
      }
    }  // end of for(int k = 0; k < (chunk_size * local_gpu_count); k++)

    // memcpy hash_table_slot_id and hash_table_value from CPU to GPU
    for (size_t id = 0; id < local_gpu_count; id++) {
      if (tile_counter_in_chunk_per_gpu[id] == 0) {
        continue;
      }

      context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

      // Copy value buffer and tensor_index buffer to GPU
      size_t value_chunk_size = tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
      float *src_buf_value = h_hash_table_value_chunk_per_gpu[id];
      float *dst_buf_value = d_hash_table_value_chunk_per_gpu[id];
      HCTR_LIB_THROW(cudaMemcpyAsync(dst_buf_value, src_buf_value, value_chunk_size * sizeof(float),
                                     cudaMemcpyHostToDevice,
                                     embedding_data_.get_local_gpu(id).get_stream()));
      size_t *src_buf_index = h_hash_table_index_chunk_per_gpu[id];
      size_t *dst_buf_index = d_hash_table_index_chunk_per_gpu[id];
      value_chunk_size = tile_counter_in_chunk_per_gpu[id];
      HCTR_LIB_THROW(cudaMemcpyAsync(dst_buf_index, src_buf_index,
                                     value_chunk_size * sizeof(size_t), cudaMemcpyHostToDevice,
                                     embedding_data_.get_local_gpu(id).get_stream()));

      // Call kernel to insert the value into embedding value tensor
      const size_t grid_size = (tile_counter_in_chunk_per_gpu[id] - 1) / 256 + 1;
      upload_value_tensor_kernel<<<grid_size, 256, 0,
                                   embedding_data_.get_local_gpu(id).get_stream()>>>(
          d_hash_table_value_chunk_per_gpu[id], d_hash_table_index_chunk_per_gpu[id],
          hash_table_value_tensors[id].get_ptr(), hash_table_value_tile_size,
          tile_counter_in_chunk_per_gpu[id]);
    }

    functors_.sync_all_gpus(embedding_data_.get_resource_manager());

    // set counter value
    for (size_t id = 0; id < local_gpu_count; id++) {
      tile_counter_in_chunk_per_gpu[id] = 0;  // reset chunk counter to zero
    }
  }  // end of for(int i = 0; i < loop_num; i++)

  // process the remaining data(less than a chunk)
  const size_t remain_loop_num = num - loop_num * chunk_size;
  float *value_dst_buf;
  size_t *tensor_index_dst_buf;
  for (size_t i = 0; i < remain_loop_num; i++) {  // process one tile in each loop

    size_t slot_id = slot_id_ptr[loop_num * chunk_size + i];
    size_t gid = slot_id % total_gpu_count;  // global GPU ID
    size_t id = embedding_data_.get_resource_manager().get_gpu_local_id_from_global_id(
        gid);  // local GPU ID (not gpudevice id)
    int dst_rank =
        embedding_data_.get_resource_manager().get_process_id_from_gpu_global_id(gid);  // node id

    if (embedding_data_.get_resource_manager().get_process_id() == dst_rank) {
      TypeHashKey tile_key = key_ptr[loop_num * chunk_size + i];
      size_t tensor_index =
          tile_key - (h_mapping_offsets_per_gpu_tensors[id][local_slot_id[slot_id]]);

      // memcpy hash_table_value to corresponding GPU
      value_dst_buf = h_hash_table_value_chunk_per_gpu[id] +
                      tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
      memcpy(value_dst_buf, embedding_ptr + (loop_num * chunk_size + i) * embedding_vec_size,
             hash_table_value_tile_size_in_B);

      tensor_index_dst_buf =
          h_hash_table_index_chunk_per_gpu[id] + tile_counter_in_chunk_per_gpu[id];
      *tensor_index_dst_buf = tensor_index;
      tile_counter_in_chunk_per_gpu[id] += 1;

    } else {
      continue;
    }
  }

  // memcpy hash_table_slot_id and hash_table_value from CPU to GPU and insert into embedding
  // table
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (tile_counter_in_chunk_per_gpu[id] == 0) {
      continue;
    }

    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

    // Copy value buffer and tensor_index buffer to GPU
    size_t value_chunk_size = tile_counter_in_chunk_per_gpu[id] * hash_table_value_tile_size;
    float *src_buf_value = h_hash_table_value_chunk_per_gpu[id];
    float *dst_buf_value = d_hash_table_value_chunk_per_gpu[id];
    HCTR_LIB_THROW(cudaMemcpyAsync(dst_buf_value, src_buf_value, value_chunk_size * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   embedding_data_.get_local_gpu(id).get_stream()));
    size_t *src_buf_index = h_hash_table_index_chunk_per_gpu[id];
    size_t *dst_buf_index = d_hash_table_index_chunk_per_gpu[id];
    value_chunk_size = tile_counter_in_chunk_per_gpu[id];
    HCTR_LIB_THROW(cudaMemcpyAsync(dst_buf_index, src_buf_index, value_chunk_size * sizeof(size_t),
                                   cudaMemcpyHostToDevice,
                                   embedding_data_.get_local_gpu(id).get_stream()));

    // Call kernel to insert the value into embedding value tensor
    const size_t grid_size = (tile_counter_in_chunk_per_gpu[id] - 1) / 256 + 1;
    upload_value_tensor_kernel<<<grid_size, 256, 0,
                                 embedding_data_.get_local_gpu(id).get_stream()>>>(
        d_hash_table_value_chunk_per_gpu[id], d_hash_table_index_chunk_per_gpu[id],
        hash_table_value_tensors[id].get_ptr(), hash_table_value_tile_size,
        tile_counter_in_chunk_per_gpu[id]);
  }

  // sync wait
  functors_.sync_all_gpus(embedding_data_.get_resource_manager());

  HCTR_LOG(INFO, ROOT, "Done\n");

  // release resources
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());
    HCTR_LIB_THROW(cudaFree(d_hash_table_value_chunk_per_gpu[id]));
    HCTR_LIB_THROW(cudaFree(d_hash_table_index_chunk_per_gpu[id]));
  }
  for (size_t id = 0; id < local_gpu_count; id++) {
    HCTR_LIB_THROW(cudaFreeHost(h_hash_table_value_chunk_per_gpu[id]));
    HCTR_LIB_THROW(cudaFreeHost(h_hash_table_index_chunk_per_gpu[id]));
    HCTR_LIB_THROW(cudaFreeHost(h_mapping_offsets_per_gpu_tensors[id]));
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::dump_parameters(
    std::string sparse_model) const {
  dump_parameters(sparse_model, embedding_data_.embedding_params_.embedding_vec_size,
                  hash_table_value_tensors_, slot_size_array_);
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::dump_parameters(
    BufferBag &buf_bag, size_t *num) const {
  TensorBag2 keys_bag = buf_bag.keys;
  TensorBag2 slot_id_bag = buf_bag.slot_id;
  Tensor2<float> &embeddings = buf_bag.embedding;
  Tensor2<TypeHashKey> keys = Tensor2<TypeHashKey>::stretch_from(keys_bag);
  Tensor2<size_t> slot_id = Tensor2<size_t>::stretch_from(slot_id_bag);

  dump_parameters(keys, slot_id, embeddings, num,
                  embedding_data_.embedding_params_.embedding_vec_size, hash_table_value_tensors_,
                  slot_size_array_);
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::dump_parameters(
    const std::string &sparse_model, size_t embedding_vec_size,
    const Tensors2<float> &hash_table_value_tensors, const std::vector<size_t> &slot_sizes) const {
  CudaDeviceContext context;
  size_t local_gpu_count = embedding_data_.get_resource_manager().get_local_gpu_count();

  auto fs = FileSystemBuilder::build_unique_by_path(sparse_model);
  bool is_local_path = IOUtils::is_local_path(sparse_model);

  const std::string key_file(sparse_model + "/key");
  const std::string slot_file(sparse_model + "/slot_id");
  const std::string vec_file(sparse_model + "/emb_vector");

#ifdef ENABLE_MPI
  HCTR_CHECK_HINT(is_local_path, "Dumping to remote file system in MPI mode is not supported.");
  fs->create_dir(sparse_model);
  MPI_File key_fh, slot_fh, vec_fh;
  HCTR_MPI_THROW(MPI_File_open(MPI_COMM_WORLD, key_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                               MPI_INFO_NULL, &key_fh));
  HCTR_MPI_THROW(MPI_File_open(MPI_COMM_WORLD, slot_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                               MPI_INFO_NULL, &slot_fh));
  HCTR_MPI_THROW(MPI_File_open(MPI_COMM_WORLD, vec_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                               MPI_INFO_NULL, &vec_fh));
#endif

  // memory allocation
  std::unique_ptr<size_t[]> count(new size_t[local_gpu_count]);
  size_t total_count = 0;

  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());
    count[id] = 0;
    for (size_t i = 0; i < slot_sizes.size(); i++) {
      size_t global_id = embedding_data_.get_local_gpu(id).get_global_id();
      if ((i % embedding_data_.get_resource_manager().get_global_gpu_count()) == global_id) {
        count[id] += slot_sizes[i];
      }
    }
    total_count += count[id];
  }

  std::vector<size_t> offset_host(local_gpu_count, 0);
  std::exclusive_scan(count.get(), count.get() + local_gpu_count, offset_host.begin(), 0);

  TypeHashKey *h_hash_table_key;
  size_t *h_hash_table_slot_id;
  float *h_hash_table_value;
  HCTR_LIB_THROW(cudaMallocHost(&h_hash_table_key, total_count * sizeof(TypeHashKey)));
  HCTR_LIB_THROW(cudaMallocHost(&h_hash_table_slot_id, total_count * sizeof(size_t)));
  HCTR_LIB_THROW(
      cudaMallocHost(&h_hash_table_value, total_count * embedding_vec_size * sizeof(float)));

  std::unique_ptr<TypeHashKey *[]> d_hash_table_key(new TypeHashKey *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_hash_table_slot_id(new size_t *[local_gpu_count]);

  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) continue;
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

    HCTR_LIB_THROW(cudaMalloc(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey)));
    HCTR_LIB_THROW(cudaMalloc(&d_hash_table_slot_id[id], count[id] * sizeof(size_t)));
  }

  // Generate key and slot_id tensor, dump value tensor on GPU
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) continue;
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

    HCTR_LOG_S(INFO, WORLD) << "Rank" << embedding_data_.get_resource_manager().get_process_id()
                            << ": Dump embedding table from GPU" << id << std::endl;

    // Loop for each slot
    size_t buffer_offset = 0;
    for (size_t i = 0; i < slot_sizes.size(); i++) {
      size_t global_id = embedding_data_.get_local_gpu(id).get_global_id();
      if ((i % embedding_data_.get_resource_manager().get_global_gpu_count()) == global_id) {
        // Generate key buffer
        size_t key_offset = 0;
        for (size_t j = 0; j < i; j++) {
          key_offset += slot_sizes[j];
        }
        functors_.memset_liner(d_hash_table_key[id] + buffer_offset, (TypeHashKey)key_offset,
                               (TypeHashKey)1, slot_sizes[i],
                               embedding_data_.get_local_gpu(id).get_stream());

        // Generate slot_id
        functors_.memset_const(d_hash_table_slot_id[id] + buffer_offset, i, slot_sizes[i],
                               embedding_data_.get_local_gpu(id).get_stream());

        buffer_offset += slot_sizes[i];
      }
    }
    // Copy key buffer to host
    HCTR_LIB_THROW(cudaMemcpyAsync(h_hash_table_key + offset_host[id], d_hash_table_key[id],
                                   count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost,
                                   embedding_data_.get_local_gpu(id).get_stream()));
    // Copy value buffer to host
    HCTR_LIB_THROW(cudaMemcpyAsync(
        h_hash_table_value + offset_host[id] * embedding_vec_size,
        hash_table_value_tensors[id].get_ptr(), count[id] * embedding_vec_size * sizeof(float),
        cudaMemcpyDeviceToHost, embedding_data_.get_local_gpu(id).get_stream()));
    // Copy slot_id to host
    HCTR_LIB_THROW(cudaMemcpyAsync(h_hash_table_slot_id + offset_host[id], d_hash_table_slot_id[id],
                                   count[id] * sizeof(size_t), cudaMemcpyDeviceToHost,
                                   embedding_data_.get_local_gpu(id).get_stream()));
  }
  functors_.sync_all_gpus(embedding_data_.get_resource_manager());

  long long *h_key_ptr;
  std::vector<long long> i64_key_vec;
  if (std::is_same<TypeHashKey, long long>::value) {
    h_key_ptr = reinterpret_cast<long long *>(h_hash_table_key);
  } else {
    i64_key_vec.resize(total_count);
    std::transform(h_hash_table_key, h_hash_table_key + total_count, i64_key_vec.begin(),
                   [](unsigned key) { return static_cast<long long>(key); });
    h_key_ptr = i64_key_vec.data();
  }

  const size_t key_size = sizeof(long long);
  const size_t slot_size = sizeof(size_t);
  const size_t vec_size = sizeof(float) * embedding_vec_size;

  // write sparse model to file
  HCTR_LOG_S(INFO, WORLD) << "Rank" << embedding_data_.get_resource_manager().get_process_id()
                          << ": Write hash table <key,value> pairs to file" << std::endl;
#ifdef ENABLE_MPI
  MPI_Datatype TYPE_EMB_VECTOR;
  HCTR_MPI_THROW(MPI_Type_contiguous(embedding_vec_size, MPI_FLOAT, &TYPE_EMB_VECTOR));
  HCTR_MPI_THROW(MPI_Type_commit(&TYPE_EMB_VECTOR));

  int my_rank = embedding_data_.get_resource_manager().get_process_id();
  int n_ranks = embedding_data_.get_resource_manager().get_num_process();

  std::vector<size_t> offset_per_rank(n_ranks, 0);
  HCTR_MPI_THROW(MPI_Allgather(&total_count, sizeof(size_t), MPI_CHAR, offset_per_rank.data(),
                               sizeof(size_t), MPI_CHAR, MPI_COMM_WORLD));
  std::exclusive_scan(offset_per_rank.begin(), offset_per_rank.end(), offset_per_rank.begin(), 0);

  size_t key_offset = offset_per_rank[my_rank] * key_size;
  size_t slot_offset = offset_per_rank[my_rank] * slot_size;
  size_t vec_offset = offset_per_rank[my_rank] * vec_size;

  HCTR_MPI_THROW(MPI_Barrier(MPI_COMM_WORLD));
  MPI_Status status;
  HCTR_MPI_THROW(
      MPI_File_write_at(key_fh, key_offset, h_key_ptr, total_count, MPI_LONG_LONG_INT, &status));
  HCTR_MPI_THROW(MPI_File_write_at(slot_fh, slot_offset, h_hash_table_slot_id, total_count,
                                   MPI_SIZE_T, &status));
  HCTR_MPI_THROW(MPI_File_write_at(vec_fh, vec_offset, h_hash_table_value, total_count,
                                   TYPE_EMB_VECTOR, &status));

  HCTR_MPI_THROW(MPI_File_close(&key_fh));
  HCTR_MPI_THROW(MPI_File_close(&slot_fh));
  HCTR_MPI_THROW(MPI_File_close(&vec_fh));
  HCTR_MPI_THROW(MPI_Type_free(&TYPE_EMB_VECTOR));
#else
  fs->write(key_file, reinterpret_cast<char *>(h_key_ptr), total_count * key_size, true);
  fs->write(slot_file, reinterpret_cast<char *>(h_hash_table_slot_id), total_count * slot_size,
            true);
  fs->write(vec_file, reinterpret_cast<char *>(h_hash_table_value), total_count * vec_size, true);
#endif
  HCTR_LOG(INFO, ROOT, "Done\n");

  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) continue;
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

    HCTR_LIB_THROW(cudaFree(d_hash_table_key[id]));
    HCTR_LIB_THROW(cudaFree(d_hash_table_slot_id[id]));
  }
  HCTR_LIB_THROW(cudaFreeHost(h_hash_table_key));
  HCTR_LIB_THROW(cudaFreeHost(h_hash_table_slot_id));
  HCTR_LIB_THROW(cudaFreeHost(h_hash_table_value));
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::dump_parameters(
    Tensor2<TypeHashKey> &keys, Tensor2<size_t> &slot_id, Tensor2<float> &embeddings, size_t *num,
    size_t embedding_vec_size, const Tensors2<float> &hash_table_value_tensors,
    const std::vector<size_t> &slot_sizes) const {
  TypeHashKey *key_ptr = keys.get_ptr();
  size_t *slot_id_ptr = slot_id.get_ptr();
  float *embedding_ptr = embeddings.get_ptr();

  size_t local_gpu_count = embedding_data_.get_resource_manager().get_local_gpu_count();

  // memory allocation
  std::unique_ptr<size_t[]> count(new size_t[local_gpu_count]);
  size_t total_count = 0;

  CudaDeviceContext context;
  for (size_t id = 0; id < local_gpu_count; id++) {
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());
    count[id] = 0;
    for (size_t i = 0; i < slot_sizes.size(); i++) {
      size_t global_id = embedding_data_.get_local_gpu(id).get_global_id();
      if ((i % embedding_data_.get_resource_manager().get_global_gpu_count()) == global_id) {
        count[id] += slot_sizes[i];
      }
    }
    total_count += count[id];
  }

  std::vector<size_t> offset_host(local_gpu_count, 0);
  std::exclusive_scan(count.get(), count.get() + local_gpu_count, offset_host.begin(), 0);
  *num = total_count;

  std::unique_ptr<TypeHashKey *[]> d_hash_table_key(new TypeHashKey *[local_gpu_count]);
  std::unique_ptr<size_t *[]> d_hash_table_slot_id(new size_t *[local_gpu_count]);

  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) continue;
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

    HCTR_LIB_THROW(cudaMalloc(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey)));
    HCTR_LIB_THROW(cudaMalloc(&d_hash_table_slot_id[id], count[id] * sizeof(size_t)));
  }

  // Generate key and slot_id tensor, dump value tensor on GPU
  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) continue;
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

    // Loop for each slot
    size_t buffer_offset = 0;
    for (size_t i = 0; i < slot_sizes.size(); i++) {
      size_t global_id = embedding_data_.get_local_gpu(id).get_global_id();
      if ((i % embedding_data_.get_resource_manager().get_global_gpu_count()) == global_id) {
        // Generate key buffer
        size_t key_offset = 0;
        for (size_t j = 0; j < i; j++) {
          key_offset += slot_sizes[j];
        }
        functors_.memset_liner(d_hash_table_key[id] + buffer_offset,
                               static_cast<TypeHashKey>(key_offset), static_cast<TypeHashKey>(1),
                               slot_sizes[i], embedding_data_.get_local_gpu(id).get_stream());

        // Generate slot_id
        functors_.memset_const(d_hash_table_slot_id[id] + buffer_offset, i, slot_sizes[i],
                               embedding_data_.get_local_gpu(id).get_stream());

        buffer_offset += slot_sizes[i];
      }
    }
    // Copy key buffer to host
    HCTR_LIB_THROW(cudaMemcpyAsync(key_ptr + offset_host[id], d_hash_table_key[id],
                                   count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost,
                                   embedding_data_.get_local_gpu(id).get_stream()));
    // Copy value buffer to host
    HCTR_LIB_THROW(cudaMemcpyAsync(
        embedding_ptr + offset_host[id] * embedding_vec_size,
        hash_table_value_tensors[id].get_ptr(), count[id] * embedding_vec_size * sizeof(float),
        cudaMemcpyDeviceToHost, embedding_data_.get_local_gpu(id).get_stream()));
    // Copy slot_id to host
    HCTR_LIB_THROW(cudaMemcpyAsync(slot_id_ptr + offset_host[id], d_hash_table_slot_id[id],
                                   count[id] * sizeof(size_t), cudaMemcpyDeviceToHost,
                                   embedding_data_.get_local_gpu(id).get_stream()));
  }
  functors_.sync_all_gpus(embedding_data_.get_resource_manager());

  for (size_t id = 0; id < local_gpu_count; id++) {
    if (count[id] == 0) continue;
    context.set_device(embedding_data_.get_local_gpu(id).get_device_id());

    HCTR_LIB_THROW(cudaFree(d_hash_table_key[id]));
    HCTR_LIB_THROW(cudaFree(d_hash_table_slot_id[id]));
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::init_embedding(
    const std::vector<size_t> slot_sizes, size_t embedding_vec_size,
    std::vector<Tensors2<float>> &hash_table_value_tensors,
    Tensors2<size_t> &hash_table_slot_id_tensors) {
  size_t local_gpu_count = embedding_data_.get_resource_manager().get_local_gpu_count();
  size_t total_gpu_count = embedding_data_.get_resource_manager().get_global_gpu_count();

#ifndef NDEBUG
  HCTR_LOG_S(DEBUG, ROOT) << "local_gpu_count=" << local_gpu_count
                          << ", total_gpu_count=" << total_gpu_count << std::endl;
#endif

#pragma omp parallel num_threads(embedding_data_.get_resource_manager().get_local_gpu_count())
  {
    size_t id = omp_get_thread_num();
    size_t device_id = embedding_data_.get_local_gpu(id).get_device_id();
    size_t global_id = embedding_data_.get_local_gpu(id).get_global_id();

#ifndef NDEBUG
    HCTR_LOG_S(DEBUG, ROOT) << "id=" << id << ", device_id=" << device_id
                            << ", global_id=" << global_id << std::endl;
#endif

    functors_.init_embedding_per_gpu(global_id, total_gpu_count, slot_sizes, embedding_vec_size,
                                     hash_table_value_tensors[id], hash_table_slot_id_tensors[id],
                                     embedding_data_.get_local_gpu(id));

    HCTR_LIB_THROW(cudaStreamSynchronize(embedding_data_.get_local_gpu(id).get_stream()));
    HCTR_LOG_S(INFO, ROOT) << "gpu" << id << " init embedding done" << std::endl;
  }

  return;
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void LocalizedSlotSparseEmbeddingOneHot<TypeHashKey, TypeEmbeddingComp>::reset() {
  CudaDeviceContext context;
  for (size_t i = 0; i < embedding_data_.get_resource_manager().get_local_gpu_count(); i++) {
    functors_.init_embedding_per_gpu(
        embedding_data_.get_local_gpu(i).get_global_id(),
        embedding_data_.get_resource_manager().get_global_gpu_count(), slot_size_array_,
        embedding_data_.embedding_params_.embedding_vec_size, value_table_tensors_[i],
        hash_table_slot_id_tensors_[i], embedding_data_.get_local_gpu(i));
  }

  for (size_t i = 0; i < embedding_data_.get_resource_manager().get_local_gpu_count(); i++) {
    HCTR_LIB_THROW(cudaStreamSynchronize(embedding_data_.get_local_gpu(i).get_stream()));
  }
}

template class LocalizedSlotSparseEmbeddingOneHot<unsigned int, float>;
template class LocalizedSlotSparseEmbeddingOneHot<long long, float>;
template class LocalizedSlotSparseEmbeddingOneHot<unsigned int, __half>;
template class LocalizedSlotSparseEmbeddingOneHot<long long, __half>;

}  // namespace HugeCTR
