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
    const std::shared_ptr<ResourceManager> &resource_manager, bool overlap_ar_a2a,
    bool eval_overlap)
    : embedding_params_(embedding_params),
      resource_manager_(resource_manager),
      stream_manager_(resource_manager->get_local_gpu_count()),
      grouped_wgrad_buff_(grouped_wgrad_buff),
      grouped_all_reduce_(grouped_wgrad_buff[0] != NULL),
      lr_scheds_(lr_scheds),
      graph_mode_(graph_mode),
      overlap_ar_a2a_(overlap_ar_a2a),
      eval_overlap_(eval_overlap),
      current_train_batch_size_(get_batch_size(true)),
      current_eval_batch_size_(get_batch_size(false)),
      use_graph_(graph_mode) {
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
void HybridSparseEmbedding<dtype, emtype>::setup_buffered_indices(
    AsyncReader<dtype> *train_data_reader, AsyncReader<dtype> *eval_data_reader) {
  auto create_buffered_indices = [this](AsyncReader<dtype> *reader, bool is_train) {
    const auto data_tensors = reader->get_value_tensor_buffers();
    auto &batch_indices = is_train ? train_batch_indices_ : eval_batch_indices_;
    batch_indices.clear();  // remove default
    for (size_t i = 0; i < data_tensors.size(); ++i) {
      batch_indices.emplace_back(model_, data_tensors.at(i), resource_manager_,
                                 get_batch_size(is_train), embedding_params_.slot_size_array,
                                 embedding_params_.max_num_frequent_categories,
                                 embedding_params_.communication_type);
    }
    return batch_indices;
  };

  if (embedding_params_.use_eval_precompute_indices) {
    // If get_max_batches_inflight() is > than the number of eval batches in the dataset,
    // this will cause the batch tensors to be cached. We need the tensors to be cached in order
    // for the indices to be cached because the index calculation is done in place in these
    // tensors.
    eval_data_reader->set_tensor_buffering(eval_data_reader->get_max_batches_inflight());
    create_buffered_indices(eval_data_reader, false);
  }

  if (embedding_params_.use_train_precompute_indices) {
    // Double buffering for overlapping indices calculation between iterations
    train_data_reader->set_tensor_buffering(2);
    create_buffered_indices(train_data_reader, true);
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::index_calculation(bool is_train, int i,
                                                             cudaStream_t stream) {
  int cur_device = get_local_gpu(i).get_device_id();
  CudaDeviceContext context(cur_device);

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

  // Only overlap index calculation with next iteration. Prevent next steps in training pipeline
  // from overlapping, so we wait for the previous iteration to end before continuing.
  if (is_train) {
    auto &iteration_end = get_local_gpu(i).get_event("iteration_end");
    HCTR_LIB_THROW(
        cudaStreamWaitEvent(stream, iteration_end, use_graph_ ? cudaEventWaitExternal : 0));
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::forward(bool is_train, bool is_first_batch, int i,
                                                   cudaStream_t stream, cudaEvent_t *evt_ptr) {
  int cur_device = get_local_gpu(i).get_device_id();
  auto &gpu = get_local_gpu(i);
  CudaDeviceContext context(cur_device);

  auto &output = (is_train) ? train_output_tensors_[i] : evaluate_output_tensors_[i];

  if (embedding_params_.communication_type == CommunicationType::IB_NVLink) {
    //// TODO: These copies need to be moved to the index computation
    // TODO, need to split into two parts? before and after frequent_emebedding
    infrequent_embeddings_ib_nvlink_[i].forward(output.get_ptr(), stream);

    frequent_embeddings_multi_node_[i].forward_network(output.get_ptr(), stream);

    evt_ptr = nullptr;

  } else if (embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
    infrequent_embeddings_ib_nvlink_hier_[i].forward_model(stream);
    // Let's initiate the communication as soon as we can and start every other non-urgent work
    // here This is for network
    if (is_train) {
      HCTR_LIB_THROW(cudaEventRecord(*evt_ptr, stream));
    }

    // This is for frequent forward network running in a side stream
    auto &stream_side = stream_manager_.get_stream(i, "stream_side");
    auto &ready_freq_fwd_net = stream_manager_.get_event(i, "ready_freq_fwd_net");
    auto &freq_fwd_net_completion = stream_manager_.get_event(i, "freq_fwd_net_completion");

    if (!is_train) {
      if (eval_overlap_) {
        HCTR_LIB_THROW(cudaStreamWaitEvent(stream, gpu.get_event("eval_comm_wait")));
      }
    }

    HCTR_LIB_THROW(cudaEventRecord(ready_freq_fwd_net, stream));
    HCTR_LIB_THROW(cudaStreamWaitEvent(stream_side, ready_freq_fwd_net));

    infrequent_embeddings_ib_nvlink_hier_[i].infrequent_forward_comms_->wait_completion(stream);
    frequent_embeddings_multi_node_[i].forward_network(output.get_ptr(), stream_side);

    infrequent_embeddings_ib_nvlink_hier_[i].hier_forward_network(
        infrequent_embeddings_ib_nvlink_hier_[i]
            .infrequent_forward_comm_buffers_->recv_buffer.get_ptr(),
        output.get_ptr(), stream);

    // join back frequent forward network
    HCTR_LIB_THROW(cudaEventRecord(freq_fwd_net_completion, stream_side));
    HCTR_LIB_THROW(cudaStreamWaitEvent(stream, freq_fwd_net_completion));

    if (!is_train) {
      if (eval_overlap_) {
        HCTR_LIB_THROW(cudaEventRecord(gpu.get_event("eval_comp_wait"), stream));
      }

      // Global barrier
      HCTR_LIB_THROW(ncclAllReduce((const void *)d_barrier_store_[i].get_ptr(),
                                   d_barrier_store_[i].get_ptr(), sizeof(uint32_t),
                                   NcclDataType<uint32_t>::getType(), ncclSum,
                                   get_local_gpu(i).get_nccl(), stream));
    }
  } else {  // Assuming single node

    infrequent_embeddings_single_node_[i].forward_network_direct(is_train, stream);

    // we just need to update frequent cache once in eval
    if (is_train) {
      frequent_embeddings_single_node_[i].forward_model(stream);
    } else {
      if (is_first_batch) {
        frequent_embeddings_single_node_[i].forward_model_eval(stream);
      }
    }

    // This barrier is needed for two reasons:
    // - Ensure all infrequent vectors have been pushed before mlp
    // - Ensure all frequent vectors have been pushed before forward_network
    gpu_barrier_->sync_all_gpus(stream, i);

    frequent_embeddings_single_node_[i].forward_network(output.get_ptr(), stream);
    evt_ptr = nullptr;
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::forward(bool is_train, bool is_first_batch) {
  size_t local_gpu_count = resource_manager_->get_local_gpu_count();

// Index calculations
#pragma omp parallel for num_threads(local_gpu_count)
  for (size_t i = 0; i < local_gpu_count; i++) {
    auto &gpu = get_local_gpu(i);
    CudaDeviceContext context(gpu.get_device_id());
    auto stream = is_train || !eval_overlap_ ? gpu.get_stream() : gpu.get_stream("eval_comms", -1);
    index_calculation(is_train, i, stream);
    forward(is_train, is_first_batch, i, stream, nullptr);
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::frequent_local_reduce(int i, cudaStream_t stream) {
  int cur_device = get_local_gpu(i).get_device_id();
  CudaDeviceContext context(cur_device);

  if (frequent_embeddings_single_node_.size()) {
    frequent_embeddings_single_node_[i].local_reduce(train_output_tensors_[i].get_ptr(), stream);
  } else {
    frequent_embeddings_multi_node_[i].local_reduce(train_output_tensors_[i].get_ptr(), stream);
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::backward_pre_communication(int i, cudaStream_t stream) {
  int cur_device = get_local_gpu(i).get_device_id();
  CudaDeviceContext context(cur_device);

  if (embedding_params_.communication_type == CommunicationType::IB_NVLink) {
    infrequent_embeddings_ib_nvlink_[i].update_network(
        train_output_tensors_[i].get_ptr(),
        infrequent_embeddings_ib_nvlink_[i]
            .infrequent_backward_comm_buffers_->send_buffer.get_ptr(),
        stream);
  } else if (embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
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
void HybridSparseEmbedding<dtype, emtype>::backward_communications(int i, cudaStream_t stream) {
  int cur_device = get_local_gpu(i).get_device_id();
  CudaDeviceContext context(cur_device);
  if (embedding_params_.communication_type == CommunicationType::NVLink_SingleNode) {
    // Synchronize all GPUs before pulling the reduced gradients
    gpu_barrier_->sync_all_gpus(stream, i);

    float *dev_lr = lr_scheds_[i]->get_learning_rate();
    float scale = opt_params_[i].scaler;
    frequent_embeddings_single_node_[i].update_model_direct(dev_lr, scale, stream);

    infrequent_embeddings_single_node_[i].update_model_direct(dev_lr, scale, stream);
  } else {
    if (!grouped_all_reduce_) {
      frequent_embeddings_multi_node_[i].communicate(stream);
    }

    if (embedding_params_.communication_type == CommunicationType::IB_NVLink) {
      infrequent_embeddings_ib_nvlink_[i].infrequent_backward_comms_->communicate(stream);
    } else {  // IB_NVLink_Hier
      infrequent_embeddings_ib_nvlink_hier_[i].infrequent_backward_comms_->communicate(stream);
    }
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::frequent_update(int i, cudaStream_t stream) {
  int cur_device = get_local_gpu(i).get_device_id();
  CudaDeviceContext context(cur_device);
  float *dev_lr = lr_scheds_[i]->get_learning_rate();
  float scale = opt_params_[i].scaler;

  if (embedding_params_.communication_type != CommunicationType::NVLink_SingleNode) {
    frequent_embeddings_multi_node_[i].update_model(dev_lr, scale, stream);
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::backward_post_communication(int i, cudaStream_t stream) {
  int cur_device = get_local_gpu(i).get_device_id();
  CudaDeviceContext context(cur_device);
  float *dev_lr = lr_scheds_[i]->get_learning_rate();
  float scale = opt_params_[i].scaler;

  if (embedding_params_.communication_type == CommunicationType::IB_NVLink) {
    infrequent_embeddings_ib_nvlink_[i].update_model(
        infrequent_embeddings_ib_nvlink_[i]
            .infrequent_backward_comm_buffers_->recv_buffer.get_ptr(),
        dev_lr, scale, stream);
  }
  if (embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
#ifdef ENABLE_MPI

    infrequent_embeddings_ib_nvlink_hier_[i].hier_update_model(
        infrequent_embeddings_ib_nvlink_hier_[i]
            .infrequent_backward_comm_buffers_->recv_buffer.get_ptr(),
        dev_lr, scale, stream);

    if (graph_mode_) {
      cudaEvent_t update_comm_event = stream_manager_.get_event(i, "update_comm_event");
      HCTR_LIB_THROW(cudaEventRecord(update_comm_event, comm_stream_[i]));
      HCTR_LIB_THROW(cudaStreamWaitEvent(stream, update_comm_event));
    }

#else
    HCTR_OWN_THROW(Error_t::WrongInput, "MPI is not enabled but trying to use IB_NVLink_Hier");
#endif
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::backward() {
  size_t local_gpu_count = resource_manager_->get_local_gpu_count();

#pragma omp parallel for num_threads(local_gpu_count)
  for (size_t i = 0; i < local_gpu_count; i++) {
    auto stream = get_local_gpu(i).get_stream();
    auto cur_device = get_local_gpu(i).get_device_id();
    CudaDeviceContext context(cur_device);
    frequent_local_reduce(i, stream);
    backward_pre_communication(i, stream);
    backward_communications(i, stream);
  }
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::update_params() {
  size_t local_gpu_count = resource_manager_->get_local_gpu_count();

#pragma omp parallel for num_threads(local_gpu_count)
  for (size_t i = 0; i < local_gpu_count; i++) {
    auto stream = get_local_gpu(i).get_stream();
    auto cur_device = get_local_gpu(i).get_device_id();
    CudaDeviceContext context(cur_device);
    frequent_update(i, stream);
    backward_post_communication(i, stream);
  }
}

template <typename dtype, typename emtype>
TrainState HybridSparseEmbedding<dtype, emtype>::train(bool is_train, int i, TrainState state) {
  auto &stream = stream_manager_.get_stream(i, "main_stream");
  auto &ready_bot_mlp_fprop = stream_manager_.get_event(i, "ready_bot_mlp_fprop");
  auto &ready_top_mlp_fprop = stream_manager_.get_event(i, "ready_top_mlp_fprop");
  auto &finish_backward_pre = stream_manager_.get_event(i, "finish_backward_pre");
  auto &finish_iteration = stream_manager_.get_event(i, "finish_iteration");

  auto sync = [&state, &stream]() {
    if (state.event) {
      HCTR_LIB_THROW(cudaStreamWaitEvent(stream, *state.event));
    }
  };

  cudaEvent_t *event_ptr = nullptr;
  switch (state.state) {
    case TrainState_t::Init:
      sync();
      index_calculation(is_train, i, stream);
      forward(is_train, -1, i, stream, &ready_bot_mlp_fprop);
      event_ptr = &ready_bot_mlp_fprop;
      break;
    case TrainState_t::BottomMLPFprop:
      sync();
      break;
    case TrainState_t::TopMLPFprop:
      HCTR_LIB_THROW(cudaEventRecord(ready_top_mlp_fprop, stream));
      event_ptr = &ready_top_mlp_fprop;
      break;
    case TrainState_t::TopMLPBprop:
      break;
    case TrainState_t::BottomMLPBprop:
      if (overlap_ar_a2a_) {
        sync();
        frequent_local_reduce(i, stream);
      }
      break;
    case TrainState_t::MLPExchangeWgrad:
      if (!overlap_ar_a2a_) {
        sync();
        frequent_local_reduce(i, stream);
        backward_pre_communication(i, stream);
      }
      if (grouped_all_reduce_) {
        HCTR_LIB_THROW(cudaEventRecord(finish_backward_pre, stream));
        event_ptr = &finish_backward_pre;
      }
      if (overlap_ar_a2a_) {
        backward_pre_communication(i, stream);
        backward_communications(i, stream);
        backward_post_communication(i, stream);
      }
      break;
    case TrainState_t::MLPUpdate:
      if (!overlap_ar_a2a_) {
        sync();
        backward_communications(i, stream);
        frequent_update(i, stream);
        backward_post_communication(i, stream);
      } else {
        sync();
        frequent_update(i, stream);
      }
      break;
    case TrainState_t::Finalize:
      HCTR_LIB_THROW(cudaEventRecord(finish_iteration, stream));
      event_ptr = &finish_iteration;
      break;
    default:
      HCTR_OWN_THROW(Error_t::InvalidEnv, "hybrid embedding train reach invalid status");
  }
  state.event = event_ptr;
  return state;
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
void HybridSparseEmbedding<dtype, emtype>::compute_indices(
    FrequentEmbeddingCompression<dtype> &compression,
    InfrequentEmbeddingSelection<dtype> &selection, CommunicationType communication_type,
    bool compute_network_cache_indices, cudaStream_t main_stream, StreamManager &manager,
    int raw_device_id, int sm_count) {
  cudaStream_t stream_frequent_sample_indices =
      manager.get_stream(raw_device_id, "stream_frequent_sample_indices");
  cudaStream_t stream_model_indices = manager.get_stream(raw_device_id, "stream_model_indices");
  cudaStream_t stream_network_indices = manager.get_stream(raw_device_id, "stream_network_indices");

  cudaEvent_t event_main = manager.get_event(raw_device_id, "event_main");
  cudaEvent_t event_frequent_sample_indices =
      manager.get_event(raw_device_id, "event_frequent_sample_indices");
  cudaEvent_t event_model_indices = manager.get_event(raw_device_id, "event_model_indices");
  cudaEvent_t event_network_indices = manager.get_event(raw_device_id, "event_network_indices");

  // The new streams can only start after previous work in the main stream has completed
  HCTR_LIB_THROW(cudaEventRecord(event_main, main_stream));
  HCTR_LIB_THROW(cudaStreamWaitEvent(stream_frequent_sample_indices, event_main));
  HCTR_LIB_THROW(cudaStreamWaitEvent(stream_model_indices, event_main));
  HCTR_LIB_THROW(cudaStreamWaitEvent(stream_network_indices, event_main));

  compression.calculate_frequent_sample_indices(stream_frequent_sample_indices);
  HCTR_LIB_THROW(cudaEventRecord(event_frequent_sample_indices, stream_frequent_sample_indices));

  selection.calculate_model_indices(stream_model_indices);
  HCTR_LIB_THROW(cudaEventRecord(event_model_indices, stream_model_indices));

  if (communication_type != CommunicationType::NVLink_SingleNode) {
    selection.calculate_network_indices(sm_count, stream_network_indices);
    HCTR_LIB_THROW(cudaEventRecord(event_network_indices, stream_network_indices));
    HCTR_LIB_THROW(cudaStreamWaitEvent(main_stream, event_network_indices));

  } else {
    cudaStream_t stream_cache_masks = manager.get_stream(raw_device_id, "stream_cache_masks");
    cudaStream_t stream_network_cache_indices =
        manager.get_stream(raw_device_id, "stream_network_cache_indices");
    cudaStream_t stream_model_cache_indices =
        manager.get_stream(raw_device_id, "stream_model_cache_indices");
    cudaEvent_t event_cache_masks = manager.get_event(raw_device_id, "event_cache_masks");
    cudaEvent_t event_network_cache_indices =
        manager.get_event(raw_device_id, "event_network_cache_indices");
    cudaEvent_t event_model_cache_indices =
        manager.get_event(raw_device_id, "event_model_cache_indices");

    HCTR_LIB_THROW(cudaStreamWaitEvent(stream_cache_masks, event_main));

    compression.calculate_cache_masks(stream_cache_masks);
    HCTR_LIB_THROW(cudaEventRecord(event_cache_masks, stream_cache_masks));

    HCTR_LIB_THROW(cudaStreamWaitEvent(stream_network_cache_indices, event_cache_masks));
    HCTR_LIB_THROW(cudaStreamWaitEvent(stream_model_cache_indices, event_cache_masks));

    // we don't need to calculate cache indices during eval
    if (compute_network_cache_indices) {
      compression.calculate_network_cache_indices(stream_network_cache_indices);
    }
    HCTR_LIB_THROW(cudaEventRecord(event_network_cache_indices, stream_network_cache_indices));
    HCTR_LIB_THROW(cudaStreamWaitEvent(main_stream, event_network_cache_indices));

    compression.calculate_model_cache_indices(sm_count, stream_model_cache_indices);
    HCTR_LIB_THROW(cudaEventRecord(event_model_cache_indices, stream_model_cache_indices));
    HCTR_LIB_THROW(cudaStreamWaitEvent(main_stream, event_model_cache_indices));
  }

  // Join streams to the main stream
  HCTR_LIB_THROW(cudaStreamWaitEvent(main_stream, event_frequent_sample_indices));
  HCTR_LIB_THROW(cudaStreamWaitEvent(main_stream, event_model_indices));
}

template <typename dtype, typename emtype>
void HybridSparseEmbedding<dtype, emtype>::assign_input_tensors(bool is_train, size_t batch_size,
                                                                size_t inflight_id, bool cached,
                                                                bool use_graph) {
  use_graph_ = use_graph;
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

template class HybridSparseEmbedding<uint32_t, __half>;
template class HybridSparseEmbedding<uint32_t, float>;
template class HybridSparseEmbedding<long long, __half>;
template class HybridSparseEmbedding<long long, float>;
}  // namespace HugeCTR
