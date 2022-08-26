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

#include <cuda_runtime.h>

#include <collectives/all_reduce_comm.hpp>
#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/calibration_data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/frequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/indices_container.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/infrequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/statistics.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/embeddings/hybrid_sparse_embedding.hpp"
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {
template <typename dtype, typename emtype>
HybridSparseEmbedding<dtype, emtype>::HybridSparseEmbedding(
    const SparseTensors<dtype> &train_input_tensors,
    const SparseTensors<dtype> &evaluate_input_tensors,
    const HybridSparseEmbeddingParams &embedding_params,
    const std::vector<BuffPtr<emtype>> &grouped_wgrad_buff,
    const GpuLearningRateSchedulers lr_scheds, bool graph_mode,
    const std::shared_ptr<ResourceManager> &resource_manager)
    : embedding_params_(embedding_params),
      resource_manager_(resource_manager),
      grouped_wgrad_buff_(grouped_wgrad_buff),
      grouped_all_reduce_(grouped_wgrad_buff[0] != NULL),
      lr_scheds_(lr_scheds),
      graph_mode_(graph_mode),
      current_train_batch_size_(get_batch_size(true)),
      current_eval_batch_size_(get_batch_size(false)) {
  try {
    // 0. Error check
    if (embedding_params_.train_batch_size < 1 || embedding_params_.evaluate_batch_size < 1 ||
        embedding_params_.slot_num < 1 || embedding_params_.embedding_vec_size < 1) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "batchsize < 1 || slot_num < 1 || embedding_vec_size < 1");
    }

    if (embedding_params_.embedding_vec_size > 1024) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "the embedding_vec_size can not be more than 1024 in embedding layer");
    }

    size_t total_gpu_count = resource_manager_->get_global_gpu_count();
    size_t local_gpu_count = resource_manager_->get_local_gpu_count();

    if (train_input_tensors.size() != local_gpu_count ||
        evaluate_input_tensors.size() != local_gpu_count) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "either train_input_tensors.size() or evaluate_input_tensors.size() isn't "
                     "local_gpu_count_");
    }

    HCTR_LOG_S(INFO, ROOT) << "Using Hybrid Embedding with train batch " << get_batch_size(true)
                           << " and eval batch " << get_batch_size(false) << std::endl;

    // 1. initialize optimizer
    for (size_t id = 0; id < local_gpu_count; id++) {
      OptParams opt_params;
      opt_params.optimizer = embedding_params_.opt_params.optimizer;
      opt_params.lr = embedding_params_.opt_params.lr;
      opt_params.update_type = embedding_params_.opt_params.update_type;
      opt_params.scaler = embedding_params_.opt_params.scaler;
      opt_params_.emplace_back(opt_params);
    }
    // 2. reserve buffers for different tensors
    data_statistics_.reserve(local_gpu_count);
    model_.reserve(local_gpu_count);
    calibration_.reserve(local_gpu_count);
    statistics_.reserve(local_gpu_count);
    train_output_tensors_.reserve(local_gpu_count);
    evaluate_output_tensors_.reserve(local_gpu_count);
    if (embedding_params_.communication_type == CommunicationType::NVLink_SingleNode) {
      frequent_embeddings_single_node_.reserve(local_gpu_count);
    } else {
      frequent_embeddings_multi_node_.reserve(local_gpu_count);
    }

    infrequent_embeddings_single_node_.reserve(local_gpu_count);
    infrequent_embeddings_ib_nvlink_.reserve(local_gpu_count);
    infrequent_embeddings_ib_nvlink_hier_.reserve(local_gpu_count);

    assert(bufs_.empty());
    CudaDeviceContext context;
    // 2.1. construct data
    for (uint32_t i = 0; i < local_gpu_count; i++) {
      int cur_device = get_local_gpu(i).get_device_id();
      context.set_device(cur_device);

      data_statistics_.emplace_back(embedding_params_.slot_size_array, get_batch_size(true),
                                    embedding_params_.num_iterations_statistics);
    }

    // 2.2 construct model
    for (uint32_t i = 0; i < local_gpu_count; i++) {
      int cur_device = get_local_gpu(i).get_device_id();
      context.set_device(cur_device);

      std::vector<uint32_t> num_instances_per_node(resource_manager_->get_num_process(), 0);
      get_num_instances_per_node(num_instances_per_node);
      model_.emplace_back(embedding_params_.communication_type,
                          resource_manager_->get_local_gpu(i)->get_global_id(),
                          num_instances_per_node, get_categories_num());
    }

    // 2.3 construct calibration
    for (uint32_t i = 0; i < local_gpu_count; i++) {
      int cur_device = get_local_gpu(i).get_device_id();
      context.set_device(cur_device);
      calibration_.emplace_back(resource_manager_->get_num_process(), embedding_params_.p_dup_max,
                                embedding_params_.max_all_reduce_bandwidth,
                                embedding_params_.max_all_to_all_bandwidth,
                                embedding_params_.efficiency_bandwidth_ratio);
    }

    // 2.4 construct Statistics
    for (uint32_t i = 0; i < local_gpu_count; i++) {
      int cur_device = get_local_gpu(i).get_device_id();
      context.set_device(cur_device);
      const size_t num_samples_statistics = embedding_params_.num_iterations_statistics *
                                            get_batch_size(true) * embedding_params_.slot_num;
      statistics_.emplace_back((dtype)num_samples_statistics, embedding_params_.slot_num,
                               model_[i].num_instances, get_categories_num());
    }

    for (uint32_t i = 0; i < local_gpu_count; i++) {
      int cur_device = get_local_gpu(i).get_device_id();
      context.set_device(cur_device);
      std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
      bufs_.emplace_back(buf);
      // 2.5. reserve for train output/ evaluate output tensors
      Tensor2<emtype> tensor;
      buf->reserve({get_batch_size_per_gpu(true), get_slot_num(), get_embedding_vec_size()},
                   &tensor);
      train_output_tensors_.emplace_back(tensor);
      buf->reserve({get_batch_size_per_gpu(false), get_slot_num(), get_embedding_vec_size()},
                   &tensor);
      evaluate_output_tensors_.emplace_back(tensor);

      // 2.6 construct frequent embedding
      if (embedding_params_.communication_type == CommunicationType::NVLink_SingleNode) {
        frequent_embeddings_single_node_.emplace_back(
            model_[i], get_local_gpu(i), grouped_wgrad_buff_[i], get_embedding_vec_size(),
            embedding_params_.max_num_frequent_categories);
      } else {
        frequent_embeddings_multi_node_.emplace_back(
            model_[i], get_local_gpu(i), grouped_wgrad_buff_[i], get_embedding_vec_size(),
            embedding_params_.max_num_frequent_categories);
      }

      // 2.7 construct infrequent embedding
      if (embedding_params_.communication_type == CommunicationType::NVLink_SingleNode) {
        infrequent_embeddings_single_node_.emplace_back(model_[i], get_local_gpu(i),
                                                        get_embedding_vec_size());
      }
      if (embedding_params_.communication_type == CommunicationType::IB_NVLink) {
        infrequent_embeddings_ib_nvlink_.emplace_back(model_[i], get_local_gpu(i),
                                                      get_embedding_vec_size());
      }
      if (embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
        infrequent_embeddings_ib_nvlink_hier_.emplace_back(model_[i], get_local_gpu(i),
                                                           get_embedding_vec_size());
      }

      // 2.8 construct communication
      if (embedding_params_.communication_type == CommunicationType::IB_NVLink) {
        size_t max_buf_size = embedding_params_.embedding_vec_size *
                              std::max(get_batch_size(true), get_batch_size(false)) *
                              embedding_params_.slot_num;
        infrequent_embeddings_ib_nvlink_.back().init_comms(
            embedding_params_.embedding_vec_size, &get_local_gpu(i), buf.get(), max_buf_size);
      }

      // Construct comm buffers
      if (embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
        infrequent_embeddings_ib_nvlink_hier_[i].init_comms(
            embedding_params_.max_num_infrequent_samples, embedding_params_.slot_num,
            embedding_params_.embedding_vec_size, buf.get(), get_batch_size(true),
            get_batch_size(false), local_gpu_count);
      }

      // For global barrier in eval
      {
        Tensor2<uint32_t> tensor;
        buf->reserve({1}, &tensor);
        d_barrier_store_.push_back(tensor);
      }
      buf->allocate();
    }

    // Frequent AR comm init
    if ((embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) ||
        (embedding_params_.communication_type == CommunicationType::IB_NVLink)) {
      if (!grouped_all_reduce_) {
        // Do your own all-reduce
        auto ar_comm = resource_manager_->get_ar_comm();
        frequent_embedding_handle_ = ar_comm->register_coll();
        // Frequent all reduce comm
        for (uint32_t i = 0; i < local_gpu_count; i++) {
          frequent_embeddings_multi_node_[i].init_ar_comm(ar_comm, frequent_embedding_handle_, i);
        }
        ar_comm->register_coll_buf(frequent_embedding_handle_);
      }
    }

    // Init after buffer allocation
    if (embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
#ifdef ENABLE_MPI
      ib_comm_ = resource_manager_->get_ib_comm();
      if (!ib_comm_) {
        resource_manager_->init_ib_comm();
        ib_comm_ = resource_manager_->get_ib_comm();
      }
      comm_stream_.resize(local_gpu_count);

      std::vector<size_t *> h_model_indices_sizes_ptrs(local_gpu_count);
      std::vector<size_t *> h_network_indices_sizes_ptrs(local_gpu_count);
      std::vector<emtype *> h_fwd_send_buffer_ptrs(local_gpu_count);
      std::vector<emtype *> h_bwd_send_buffer_ptrs(local_gpu_count);
      for (uint32_t i = 0; i < local_gpu_count; i++) {
        h_model_indices_sizes_ptrs[i] =
            infrequent_embeddings_ib_nvlink_hier_[i].model_indices_sizes_.get_ptr();
        h_network_indices_sizes_ptrs[i] =
            infrequent_embeddings_ib_nvlink_hier_[i].network_indices_sizes_.get_ptr();
        h_fwd_send_buffer_ptrs[i] = infrequent_embeddings_ib_nvlink_hier_[i]
                                        .infrequent_forward_comm_buffers_->send_buffer.get_ptr();
        h_bwd_send_buffer_ptrs[i] = infrequent_embeddings_ib_nvlink_hier_[i]
                                        .infrequent_backward_comm_buffers_->send_buffer.get_ptr();
      }

      // Forward coll init
      auto infrequent_forward_coll_handle = ib_comm_->register_hier_a2a_v_coll(true);
      for (uint32_t i = 0; i < local_gpu_count; i++) {
        int cur_device = get_local_gpu(i).get_device_id();
        context.set_device(cur_device);

        // download pointers
        HCTR_LIB_THROW(cudaMemcpyAsync(
            infrequent_embeddings_ib_nvlink_hier_[i].model_indices_sizes_ptrs_.get_ptr(),
            h_model_indices_sizes_ptrs.data(), sizeof(size_t *) * local_gpu_count,
            cudaMemcpyHostToDevice, get_local_gpu(i).get_stream()));

        HCTR_LIB_THROW(cudaMemcpyAsync(
            infrequent_embeddings_ib_nvlink_hier_[i].network_indices_sizes_ptrs_.get_ptr(),
            h_network_indices_sizes_ptrs.data(), sizeof(size_t *) * local_gpu_count,
            cudaMemcpyHostToDevice, get_local_gpu(i).get_stream()));

        HCTR_LIB_THROW(
            cudaMemcpyAsync(infrequent_embeddings_ib_nvlink_hier_[i]
                                .infrequent_forward_comm_buffers_->send_buffer_ptrs.get_ptr(),
                            h_fwd_send_buffer_ptrs.data(), sizeof(emtype *) * local_gpu_count,
                            cudaMemcpyHostToDevice, get_local_gpu(i).get_stream()));

        HCTR_LIB_THROW(
            cudaMemcpyAsync(infrequent_embeddings_ib_nvlink_hier_[i]
                                .infrequent_backward_comm_buffers_->send_buffer_ptrs.get_ptr(),
                            h_bwd_send_buffer_ptrs.data(), sizeof(emtype *) * local_gpu_count,
                            cudaMemcpyHostToDevice, get_local_gpu(i).get_stream()));

        HCTR_LIB_THROW(cudaStreamSynchronize(get_local_gpu(i).get_stream()));

        // Initialize IB comm
        HCTR_LIB_THROW(cudaStreamCreateWithPriority(&comm_stream_[i], cudaStreamNonBlocking, -100));
        ib_comm_->set_a2a_coll_stream(infrequent_forward_coll_handle, comm_stream_[i], i);

        ib_comm_->set_a2a_coll_buf(
            infrequent_forward_coll_handle,
            infrequent_embeddings_ib_nvlink_hier_[i]
                .infrequent_forward_comm_buffers_->send_buffer.get_ptr(),
            infrequent_embeddings_ib_nvlink_hier_[i]
                .infrequent_forward_comm_buffers_->send_buffer.get_size_in_bytes(),
            infrequent_embeddings_ib_nvlink_hier_[i]
                .infrequent_forward_comm_buffers_->recv_buffer.get_ptr(),
            infrequent_embeddings_ib_nvlink_hier_[i]
                .infrequent_forward_comm_buffers_->recv_buffer.get_size_in_bytes(),
            i);

        infrequent_embeddings_ib_nvlink_hier_[i].infrequent_forward_comms_ =
            std::make_unique<HierAll2Allv_Multi_IB<emtype>>(
                i, infrequent_forward_coll_handle,
                infrequent_embeddings_ib_nvlink_hier_[i].model_indices_sizes_ptrs_.get_ptr(),
                &get_local_gpu(i), ib_comm_, comm_stream_[i]);
      }
      ib_comm_->register_a2a_coll_buf(infrequent_forward_coll_handle);

      // Backward coll init
      auto infrequent_backward_coll_handle = ib_comm_->register_hier_a2a_v_coll(true);
      for (uint32_t i = 0; i < local_gpu_count; i++) {
        int cur_device = get_local_gpu(i).get_device_id();
        context.set_device(cur_device);

        ib_comm_->set_a2a_coll_stream(infrequent_backward_coll_handle, comm_stream_[i], i);
        ib_comm_->set_a2a_coll_buf(
            infrequent_backward_coll_handle,
            infrequent_embeddings_ib_nvlink_hier_[i]
                .infrequent_backward_comm_buffers_->send_buffer.get_ptr(),
            infrequent_embeddings_ib_nvlink_hier_[i]
                .infrequent_backward_comm_buffers_->send_buffer.get_size_in_bytes(),
            infrequent_embeddings_ib_nvlink_hier_[i]
                .infrequent_backward_comm_buffers_->recv_buffer.get_ptr(),
            infrequent_embeddings_ib_nvlink_hier_[i]
                .infrequent_backward_comm_buffers_->recv_buffer.get_size_in_bytes(),
            i);

        infrequent_embeddings_ib_nvlink_hier_[i].infrequent_backward_comms_ =
            std::make_unique<HierAll2Allv_Multi_IB<emtype>>(
                i, infrequent_backward_coll_handle,
                infrequent_embeddings_ib_nvlink_hier_[i].network_indices_sizes_ptrs_.get_ptr(),
                &get_local_gpu(i), ib_comm_, comm_stream_[i]);
      }
      ib_comm_->register_a2a_coll_buf(infrequent_backward_coll_handle);
#else
      HCTR_OWN_THROW(Error_t::WrongInput, "MPI is not enabled but trying to use IB_NVLink_Hier");
#endif
    }

    // 2.9 Single-node: copy some pointers arrays to device
    if (embedding_params_.communication_type == CommunicationType::NVLink_SingleNode) {
      // Initialize GPU barrier
      gpu_barrier_ = std::make_unique<GPUBarrier>(resource_manager_->get_local_gpu_count(),
                                                  resource_manager_->get_local_gpu_device_id_list(),
                                                  graph_mode_);

      std::vector<const emtype *> frequent_vectors_cache_pointers(local_gpu_count);
      std::vector<emtype *> interaction_layer_input_pointers_train(local_gpu_count);
      std::vector<emtype *> interaction_layer_input_pointers_eval(local_gpu_count);
      std::vector<const emtype *> gradients_pointers(local_gpu_count);
      std::vector<const emtype *> frequent_partial_gradients_pointers(local_gpu_count);

      for (uint32_t i = 0; i < local_gpu_count; i++) {
        frequent_vectors_cache_pointers[i] =
            frequent_embeddings_single_node_[i].get_embedding_vectors_cache().get_ptr();
        interaction_layer_input_pointers_train[i] = train_output_tensors_[i].get_ptr();
        gradients_pointers[i] = train_output_tensors_[i].get_ptr();
        interaction_layer_input_pointers_eval[i] = evaluate_output_tensors_[i].get_ptr();
        frequent_partial_gradients_pointers[i] =
            frequent_embeddings_single_node_[i].frequent_data_.get_gradients().get_ptr();
      }

      for (uint32_t i = 0; i < local_gpu_count; i++) {
        int cur_device = get_local_gpu(i).get_device_id();
        context.set_device(cur_device);

        HCTR_LIB_THROW(cudaMemcpyAsync(
            frequent_embeddings_single_node_[i].embedding_vectors_cache_pointers_.get_ptr(),
            frequent_vectors_cache_pointers.data(), local_gpu_count * sizeof(float *),
            cudaMemcpyHostToDevice, get_local_gpu(i).get_stream()));

        infrequent_embeddings_single_node_[i].init_pointers(
            local_gpu_count, get_local_gpu(i).get_stream(), interaction_layer_input_pointers_train,
            interaction_layer_input_pointers_eval, gradients_pointers);
        HCTR_LIB_THROW(cudaMemcpyAsync(
            frequent_embeddings_single_node_[i].partial_gradients_pointers_.get_ptr(),
            frequent_partial_gradients_pointers.data(), local_gpu_count * sizeof(emtype *),
            cudaMemcpyHostToDevice, get_local_gpu(i).get_stream()));
      }
    }

    // Setup default indices
    train_batch_indices_.emplace_back(model_, train_input_tensors, resource_manager_,
                                      get_batch_size(true), embedding_params_.slot_size_array,
                                      embedding_params_.max_num_frequent_categories,
                                      embedding_params_.communication_type);

    eval_batch_indices_.emplace_back(model_, evaluate_input_tensors, resource_manager_,
                                     get_batch_size(false), embedding_params_.slot_size_array,
                                     embedding_params_.max_num_frequent_categories,
                                     embedding_params_.communication_type);

  } catch (const std::runtime_error &rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::init_model(const SparseTensors<dtype> &data,
                                                      size_t &wgrad_offset_in_bytes) {
  size_t local_gpu_count = resource_manager_->get_local_gpu_count();
  HCTR_LOG(INFO, ROOT, "Initializing Hybrid Embedding\n");
#pragma omp parallel for num_threads(local_gpu_count)
  for (size_t id = 0; id < local_gpu_count; ++id) {
    int cur_device = get_local_gpu(id).get_device_id();
    CudaDeviceContext context(cur_device);
    auto stream = get_local_gpu(id).get_stream();
    data_statistics_[id].data_to_unique_categories(data[id].get_value_tensor(), stream);
    model_[id].init_hybrid_model(calibration_[id], statistics_[id], data_statistics_[id], stream);

    get_frequent_embedding_data(id).initialize_embedding_vectors(data_statistics_[id].table_sizes,
                                                                 wgrad_offset_in_bytes);

    if (embedding_params_.communication_type == CommunicationType::NVLink_SingleNode) {
      infrequent_embeddings_single_node_[id].initialize_embedding_vectors(
          data_statistics_[id].table_sizes);
    }
    if (embedding_params_.communication_type == CommunicationType::IB_NVLink) {
      infrequent_embeddings_ib_nvlink_[id].initialize_embedding_vectors(
          data_statistics_[id].table_sizes);
    }
    if (embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
      infrequent_embeddings_ib_nvlink_hier_[id].initialize_embedding_vectors(
          data_statistics_[id].table_sizes);
    }

    if (embedding_params_.max_num_frequent_categories < (size_t)model_[id].num_frequent) {
      HCTR_OWN_THROW(
          Error_t::WrongInput,
          "Found too many frequent categories, please increase 'max_num_frequent_categories'");
    }
  }

  HCTR_LOG_S(INFO, ROOT) << "Initialized hybrid model with " << model_[0].num_frequent
                         << " frequent categories, probability of being frequent is "
                         << model_[0].frequent_probability << std::endl;

  size_t avg_train_infrequent = (1 - model_[0].frequent_probability) *
                                embedding_params_.slot_size_array.size() * get_batch_size(true);
  size_t avg_evaluate_infrequent = (1 - model_[0].frequent_probability) *
                                   embedding_params_.slot_size_array.size() * get_batch_size(false);

  HCTR_LOG_S(INFO, ROOT) << "Estimated number of infrequent categories per train batch: "
                         << avg_train_infrequent << ", eval batch: " << avg_evaluate_infrequent
                         << std::endl;

  if ((embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) ||
      (embedding_params_.communication_type == CommunicationType::IB_NVLink)) {
    size_t wgrad_size =
        model_[0].num_frequent * embedding_params_.embedding_vec_size * sizeof(emtype);

    if (!grouped_all_reduce_) {
      // Manage your own all-reduce
      auto ar_comm = resource_manager_->get_ar_comm();
      ar_comm->update_size(frequent_embedding_handle_, wgrad_size);
    } else {
      wgrad_offset_in_bytes += wgrad_size;
    }
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::setup_buffered_indices(bool is_train,
                                                                  AsyncReader<dtype> *data_reader) {
  if (is_train) {
    // Double buffering for overlapping indices calculation between iterations
    data_reader->set_tensor_buffering(2);
  } else {
    // If get_max_batches_inflight() is > than the number of eval batches in the dataset,
    // this will cause the batch tensors to be cached. We need the tensors to be cached in order
    // for the indices to be cached because the index calculation is done in place in these
    // tensors.
    // TODO: if OOM then eval_data_reader->set_tensor_buffering(2)
    data_reader->set_tensor_buffering(data_reader->get_max_batches_inflight());
  }

  const auto data_tensors = data_reader->get_value_tensor_buffers();
  auto &batch_indices = is_train ? train_batch_indices_ : eval_batch_indices_;
  batch_indices.clear();  // remove default
  for (size_t i = 0; i < data_tensors.size(); ++i) {
    batch_indices.emplace_back(model_, data_tensors.at(i), resource_manager_,
                               get_batch_size(is_train), embedding_params_.slot_size_array,
                               embedding_params_.max_num_frequent_categories,
                               embedding_params_.communication_type);
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::forward(bool is_train) {
  size_t local_gpu_count = resource_manager_->get_local_gpu_count();

// Index calculations
#pragma omp parallel for num_threads(local_gpu_count)
  for (size_t i = 0; i < local_gpu_count; i++) {
    auto &gpu = get_local_gpu(i);
    CudaDeviceContext context(gpu.get_device_id());

    index_calculation(is_train, i);
    infreq_model_forward(i);
    freq_forward(is_train, i, true);
    infreq_network_forward(is_train, i);
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::backward() {
  size_t local_gpu_count = resource_manager_->get_local_gpu_count();

#pragma omp parallel for num_threads(local_gpu_count)
  for (size_t i = 0; i < local_gpu_count; i++) {
    auto cur_device = get_local_gpu(i).get_device_id();
    CudaDeviceContext context(cur_device);

    freq_backward(i);
    infreq_network_backward(i);
    infreq_model_backward(i);
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::update_params() {
  size_t local_gpu_count = resource_manager_->get_local_gpu_count();

#pragma omp parallel for num_threads(local_gpu_count)
  for (size_t i = 0; i < local_gpu_count; i++) {
    auto cur_device = get_local_gpu(i).get_device_id();
    CudaDeviceContext context(cur_device);

    freq_update_params(i);
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::init_params() {
  // TODO: create init_params()
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::load_parameters(
    std::string sparse_model, const DataSourceParams &data_source_params) {
  // TODO: create load_parameters()
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::dump_parameters(
    std::string sparse_model, const DataSourceParams &data_source_params) const {
  // TODO: create dump_parameters()
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::set_learning_rate(float lr) {
  HCTR_OWN_THROW(Error_t::WrongInput, "HybridSparseEmbedding only supports GPU LR scheduler");
}

template <typename dtype, typename emtype>
GpuLearningRateSchedulers HybridSparseEmbedding<dtype, emtype>::get_learning_rate_schedulers()
    const {
  return lr_scheds_;
}

template <typename dtype, typename emtype>
size_t HybridSparseEmbedding<dtype, emtype>::get_params_num() const {
  return 0;
}

template <typename dtype, typename emtype>
size_t HybridSparseEmbedding<dtype, emtype>::get_vocabulary_size() const {
  // TODO: create get_vocabulary_size()
  return 0;
}

template <typename dtype, typename emtype>
size_t HybridSparseEmbedding<dtype, emtype>::get_max_vocabulary_size() const {
  // TODO: create get_max_vocabulary_size()
  return 0;
}

template <typename dtype, typename emtype>
std::vector<TensorBag2> HybridSparseEmbedding<dtype, emtype>::get_train_output_tensors() const {
  return tensors_to_bags(train_output_tensors_);
}

template <typename dtype, typename emtype>
std::vector<TensorBag2> HybridSparseEmbedding<dtype, emtype>::get_evaluate_output_tensors() const {
  return tensors_to_bags(evaluate_output_tensors_);
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::assign_input_tensors(bool is_train, size_t batch_size,
                                                                size_t inflight_id, bool cached) {
  if (is_train) {
    train_inflight_id_ = inflight_id;
    current_train_batch_size_ = batch_size;
    current_train_batch_cached_ = cached;
  } else {
    eval_inflight_id_ = inflight_id;
    current_eval_batch_size_ = batch_size;
    current_eval_batch_cached_ = cached;
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::index_calculation(bool is_train, int i) {
  int cur_device = get_local_gpu(i).get_device_id();
  CudaDeviceContext context(cur_device);
  auto &gpu = get_local_gpu(i);
  cudaStream_t stream = gpu.get_stream();

  auto &batch_indices = is_train ? train_batch_indices_.at(train_inflight_id_)
                                 : eval_batch_indices_.at(eval_inflight_id_);

  if (is_train) {
    if (!current_train_batch_cached_) {
      batch_indices.compute(i, current_train_batch_size_, stream);
    }
  } else {  // eval
    if (!current_eval_batch_cached_) {
      batch_indices.compute(i, current_eval_batch_size_, stream);
    }
  }

  // We don't copy the sparse tensor since all the required data are already in the
  // Data type and indices
  get_frequent_embedding(i).set_current_indices(&batch_indices.get_frequent(i));
  get_infrequent_embedding(i).set_current_indices(&batch_indices.get_infrequent(i));
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::freq_forward(bool is_train, int i,
                                                        bool is_first_eval_batch) {
  int cur_device = get_local_gpu(i).get_device_id();
  auto &gpu = get_local_gpu(i);
  CudaDeviceContext context(cur_device);
  cudaStream_t stream = gpu.get_stream();

  auto &output = (is_train) ? train_output_tensors_[i] : evaluate_output_tensors_[i];
  if (embedding_params_.communication_type == CommunicationType::NVLink_SingleNode) {
    if (is_train) {
      frequent_embeddings_single_node_[i].forward_model(stream);
    } else {
      if (is_first_eval_batch) {
        frequent_embeddings_single_node_[i].forward_model_eval(stream);
      }
    }
    gpu_barrier_->sync_all_gpus(stream, i);

    frequent_embeddings_single_node_[i].forward_network(output.get_ptr(), stream);
  }
  if (embedding_params_.communication_type == CommunicationType::IB_NVLink ||
      embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
    frequent_embeddings_multi_node_[i].forward_network(output.get_ptr(), stream);
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::freq_backward(int i) {
  int cur_device = get_local_gpu(i).get_device_id();
  CudaDeviceContext context(cur_device);
  auto &gpu = get_local_gpu(i);
  cudaStream_t stream = gpu.get_stream();

  if (frequent_embeddings_single_node_.size()) {
    frequent_embeddings_single_node_[i].local_reduce(train_output_tensors_[i].get_ptr(), stream);
  } else {
    frequent_embeddings_multi_node_[i].local_reduce(train_output_tensors_[i].get_ptr(), stream);
    if (!grouped_all_reduce_) {
      frequent_embeddings_multi_node_[i].communicate(stream);
    }
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::freq_update_params(int i) {
  int cur_device = get_local_gpu(i).get_device_id();
  CudaDeviceContext context(cur_device);
  float *dev_lr = lr_scheds_[i]->get_learning_rate();
  float scale = opt_params_[i].scaler;
  auto &gpu = get_local_gpu(i);
  cudaStream_t stream = gpu.get_stream();

  if (embedding_params_.communication_type != CommunicationType::NVLink_SingleNode) {
    frequent_embeddings_multi_node_[i].update_model(dev_lr, scale, stream);
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::infreq_model_forward(int i) {
  int cur_device = get_local_gpu(i).get_device_id();
  auto &gpu = get_local_gpu(i);
  CudaDeviceContext context(cur_device);
  cudaStream_t stream = gpu.get_stream();

  if (embedding_params_.communication_type == CommunicationType::IB_NVLink) {
    infrequent_embeddings_ib_nvlink_[i].forward_model(
        infrequent_embeddings_ib_nvlink_[i].infrequent_forward_comm_buffers_->send_buffer.get_ptr(),
        stream);
  } else if (embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
    infrequent_embeddings_ib_nvlink_hier_[i].calculate_model_indices_sizes_from_offsets(stream);
    infrequent_embeddings_ib_nvlink_hier_[i].calculate_network_indices_sizes_from_offsets(stream);
    infrequent_embeddings_ib_nvlink_hier_[i].infrequent_forward_comms_->update_sizes(stream);
    infrequent_embeddings_ib_nvlink_hier_[i].fused_intra_forward_model(
        infrequent_embeddings_ib_nvlink_hier_[i]
            .infrequent_forward_comm_buffers_->send_buffer_ptrs.get_ptr(),
        stream);
    infrequent_embeddings_ib_nvlink_hier_[i].infrequent_forward_comms_->initiate_communication(
        stream);
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::infreq_network_forward(bool is_train, int i) {
  int cur_device = get_local_gpu(i).get_device_id();
  auto &gpu = get_local_gpu(i);
  CudaDeviceContext context(cur_device);
  cudaStream_t stream = gpu.get_stream();

  auto &output = (is_train) ? train_output_tensors_[i] : evaluate_output_tensors_[i];
  if (embedding_params_.communication_type == CommunicationType::IB_NVLink) {
    infrequent_embeddings_ib_nvlink_[i].infrequent_forward_comms_->communicate(stream);
    infrequent_embeddings_ib_nvlink_[i].forward_network(
        infrequent_embeddings_ib_nvlink_[i].infrequent_forward_comm_buffers_->recv_buffer.get_ptr(),
        output.get_ptr(), stream);
  } else if (embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
    infrequent_embeddings_ib_nvlink_hier_[i].infrequent_forward_comms_->wait_completion(stream);
    infrequent_embeddings_ib_nvlink_hier_[i].hier_forward_network(
        infrequent_embeddings_ib_nvlink_hier_[i]
            .infrequent_forward_comm_buffers_->recv_buffer.get_ptr(),
        output.get_ptr(), stream);
  } else {
    infrequent_embeddings_single_node_[i].forward_network_direct(is_train, stream);
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::global_barrier(bool is_train, int i) {
  int cur_device = get_local_gpu(i).get_device_id();
  auto &gpu = get_local_gpu(i);
  CudaDeviceContext context(cur_device);
  cudaStream_t stream = gpu.get_stream();

  if (embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
    if (!is_train) {
      HCTR_LIB_THROW(ncclAllReduce((const void *)d_barrier_store_[i].get_ptr(),
                                   d_barrier_store_[i].get_ptr(), sizeof(uint32_t),
                                   NcclDataType<uint32_t>::getType(), ncclSum,
                                   get_local_gpu(i).get_nccl(), stream));
    }
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::infreq_network_backward(int i) {
  int cur_device = get_local_gpu(i).get_device_id();
  CudaDeviceContext context(cur_device);
  auto &gpu = get_local_gpu(i);
  cudaStream_t stream = gpu.get_stream();

  if (embedding_params_.communication_type == CommunicationType::IB_NVLink) {
    infrequent_embeddings_ib_nvlink_[i].update_network(
        train_output_tensors_[i].get_ptr(),
        infrequent_embeddings_ib_nvlink_[i]
            .infrequent_backward_comm_buffers_->send_buffer.get_ptr(),
        stream);
  }
  if (embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
    infrequent_embeddings_ib_nvlink_hier_[i].infrequent_backward_comms_->update_sizes(stream);
    infrequent_embeddings_ib_nvlink_hier_[i].fused_intra_update_network(
        train_output_tensors_[i].get_ptr(),
        infrequent_embeddings_ib_nvlink_hier_[i]
            .infrequent_backward_comm_buffers_->send_buffer_ptrs.get_ptr(),
        stream);
  }
}

// Everything that involves network and can be better overlapped with compute
template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::infreq_model_backward(int i) {
  int cur_device = get_local_gpu(i).get_device_id();
  CudaDeviceContext context(cur_device);
  auto &gpu = get_local_gpu(i);
  cudaStream_t stream = gpu.get_stream();
  float *dev_lr = lr_scheds_[i]->get_learning_rate();
  float scale = opt_params_[i].scaler;

  if (embedding_params_.communication_type == CommunicationType::IB_NVLink) {
    infrequent_embeddings_ib_nvlink_[i].infrequent_backward_comms_->communicate(stream);
    infrequent_embeddings_ib_nvlink_[i].update_model(
        infrequent_embeddings_ib_nvlink_[i]
            .infrequent_backward_comm_buffers_->recv_buffer.get_ptr(),
        dev_lr, scale, stream);
  }

  if (embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
    infrequent_embeddings_ib_nvlink_hier_[i].infrequent_backward_comms_->communicate(stream);

    infrequent_embeddings_ib_nvlink_hier_[i].hier_update_model(
        infrequent_embeddings_ib_nvlink_hier_[i]
            .infrequent_backward_comm_buffers_->recv_buffer.get_ptr(),
        dev_lr, scale, stream);
  }
  if (embedding_params_.communication_type == CommunicationType::NVLink_SingleNode) {
    // Synchronize all GPUs before pulling the reduced gradients
    gpu_barrier_->sync_all_gpus(stream, i);

    float *dev_lr = lr_scheds_[i]->get_learning_rate();
    float scale = opt_params_[i].scaler;
    frequent_embeddings_single_node_[i].update_model_direct(dev_lr, scale, stream);

    infrequent_embeddings_single_node_[i].update_model_direct(dev_lr, scale, stream);
  }
}

template class HybridSparseEmbedding<uint32_t, __half>;
template class HybridSparseEmbedding<uint32_t, float>;
template class HybridSparseEmbedding<long long, __half>;
template class HybridSparseEmbedding<long long, float>;
}  // namespace HugeCTR
