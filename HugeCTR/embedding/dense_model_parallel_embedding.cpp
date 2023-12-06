/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <embedding/dense_model_parallel_embedding.hpp>
#include <utils.hpp>

namespace embedding {

DenseUniformModelParallelEmbeddingMeta::DenseUniformModelParallelEmbeddingMeta(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
    size_t grouped_id)
    : num_lookup_(ebc_param.num_lookup) {
  HugeCTR::CudaDeviceContext context(core->get_device_id());
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::BufferParams buffer_params;
  buffer_params.unitary = false;
  core23::TensorParams params = core23::TensorParams().device(device).buffer_params(buffer_params);

  const auto &lookup_params = ebc_param.lookup_params;
  const auto &group_params = ebc_param.grouped_lookup_params[grouped_id];
  HCTR_CHECK_HINT(
      group_params.embedding_group_type == EmbeddingGroupType::DenseModelParallel ||
          group_params.embedding_group_type == EmbeddingGroupType::DenseModelParallelWithReduction,
      "DenseUniformModelParallelEmbeddingMeta must be initialized by DenseModelParallel or "
      "DenseModelParallelWithReduction");

  size_t num_gpus = core->get_global_gpu_count();
  // int gpu_id = core->get_global_gpu_id();

  HCTR_CHECK_HINT(ebc_param.shard_matrix.size() == num_gpus,
                  "shard matrix should contain num_gpus row.");
  this->num_local_hotness_after_reduction_ = 0;
  this->num_local_hotness_before_reduction_ = 0;
  this->num_local_lookup_ = 0;
  this->global_hotness_ = 0;
  this->global_ev_offset_ = 0;

  this->h_local_hotness_range_.clear();
  this->h_local_hotness_.clear();
  this->h_ev_start_indices_.clear();

  this->h_local_hotness_range_.push_back(0);
  int local_id_in_embedding = 0;
  int local_id_in_output_buffer = 0;
  std::vector<int> lookup_ids_in_embedding;
  std::vector<int> ev_length_in_output_buffer;
  std::vector<int> h_local_table_id_list;
  std::vector<int> h_ev_size_list;
  // iterate over all lookup and calculate embedding output hotness(after reduction) offset
  for (int lookup_id = 0; lookup_id < num_lookup_; ++lookup_id) {
    // if this dense lookup is not in current grouped lookup
    if (std::find(ebc_param.grouped_lookup_params[grouped_id].lookup_ids.begin(),
                  ebc_param.grouped_lookup_params[grouped_id].lookup_ids.end(),
                  lookup_id) == ebc_param.grouped_lookup_params[grouped_id].lookup_ids.end()) {
      int tmp_hotness = lookup_params[lookup_id].combiner == Combiner::Concat
                            ? lookup_params[lookup_id].max_hotness
                            : 1;
      this->global_ev_offset_ += (tmp_hotness * lookup_params[lookup_id].ev_size);
      this->global_hotness_ += tmp_hotness;
      ev_length_in_output_buffer.push_back(tmp_hotness * lookup_params[lookup_id].ev_size);
      local_id_in_output_buffer++;
      continue;
    }
    // else find current dense lookup!!
    // the combiner can be Sum even if the lookup is dense. dense lookup only stands for
    // unique-based lookup
    this->num_local_hotness_before_reduction_ += lookup_params[lookup_id].max_hotness;
    int tmp_hotness = lookup_params[lookup_id].combiner == Combiner::Concat
                          ? lookup_params[lookup_id].max_hotness
                          : 1;
    // TODO this if is redundant?
    if (ebc_param.grouped_lookup_params[grouped_id].embedding_group_type ==
        EmbeddingGroupType::DenseModelParallelWithReduction) {
      tmp_hotness = 1;
    }
    // num_local_hotness_after_reduction_is the output hotness, i.e. after embedding combiner
    this->num_local_hotness_after_reduction_ += tmp_hotness;
    this->global_hotness_ += tmp_hotness;
    this->h_local_hotness_range_.push_back(tmp_hotness);
    this->h_local_hotness_.push_back(tmp_hotness);
    this->global_ev_offset_ += (tmp_hotness * lookup_params[lookup_id].ev_size);
    h_ev_size_list.push_back(lookup_params[lookup_id].ev_size);

    ev_length_in_output_buffer.push_back(tmp_hotness * lookup_params[lookup_id].ev_size);

    lookup_ids_in_embedding.push_back(local_id_in_output_buffer);
    local_id_in_output_buffer++;
    local_id_in_embedding++;
    this->num_local_lookup_++;

    int table_id = lookup_params[lookup_id].table_id;
    h_local_table_id_list.push_back(table_id);
  }

  for (int i = 1; i < this->h_local_hotness_range_.size(); ++i) {
    this->h_local_hotness_range_[i] =
        this->h_local_hotness_range_[i] + this->h_local_hotness_range_[i - 1];
  }

  HCTR_CHECK_HINT(h_ev_size_list.size() > 0, "Dense lookup num should > 0");
  if (std::equal(h_ev_size_list.begin() + 1, h_ev_size_list.end(), h_ev_size_list.begin())) {
    this->ev_size_ = h_ev_size_list[0];
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                   "All lookup in dense lookup should have same ev_size");
  }

  num_local_lookup_ = static_cast<int>(h_local_table_id_list.size());
  for (int local_lookup_id = 0; local_lookup_id < num_local_lookup_; ++local_lookup_id) {
    int map_id = lookup_ids_in_embedding[local_lookup_id];
    int tmp_global_length = 0;
    for (int pre_id = 0; pre_id < map_id; pre_id++) {
      tmp_global_length += ev_length_in_output_buffer[pre_id];
    }
    this->h_ev_start_indices_.push_back(tmp_global_length);
  }

  model_buffer_attr.init(core, ebc_param, grouped_id);
  network_indices.init(core, this->h_local_hotness_range_, this->h_local_hotness_,
                       this->h_ev_start_indices_, this->num_local_lookup_, this->global_ev_offset_);
  network_buffer_attr.init(core, ebc_param, grouped_id, this->num_local_hotness_before_reduction_);
  wgrad_attr.init(core, ebc_param, grouped_id);
}

DenseUniformModelParallelEmbedding::DenseUniformModelParallelEmbedding(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &params,
    size_t grouped_id)
    : core_(core),
      meta_(core, params, grouped_id),
      do_reduction_(params.grouped_lookup_params[grouped_id].embedding_group_type ==
                    EmbeddingGroupType::DenseModelParallelWithReduction) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());

  model_forward_ = ModelForward{core};
  all2all_comm_ = NcclAll2AllComm(core);
  network_forward_ = NetworkForward(core);
  network_backward_ = NetworkBackward(core);

  local_reduce_index_calculation_.init(core);

  local_reduce_.init(core, meta_.ev_size_,
                     meta_.num_local_hotness_after_reduction_ * params.universal_batch_size);
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams tensor_params = core23::TensorParams().device(device);

  embedding_vec_ = core23::init_tensor_list<float>(
      params.universal_batch_size * meta_.num_local_hotness_before_reduction_,
      core->get_device_id());

  model_comm_buffer_.init(core, meta_.model_buffer_attr, params.universal_batch_size);
  network_buffer_.init(core, meta_.network_buffer_attr, params.universal_batch_size);
}
void DenseUniformModelParallelEmbedding::model_forward(const EmbeddingInput &embedding_input,
                                                       ILookup *embedding_table, int batch_size) {
  // (lookup) Results are emb vector addresses
  // output is embedding_vec_
  embedding_table->lookup(
      embedding_input.keys, embedding_input.h_num_keys,
      embedding_input.dense_compression_input.num_keys_per_table_offset,
      (size_t)embedding_input.dense_compression_input.num_keys_per_table_offset.num_elements(),
      embedding_input.dense_compression_input.table_ids, embedding_vec_);
  // (gather) Results are emb embedding vectors stored in model_comm_buffer_,
  // output is model_comm_buffer_
  // num_model_reverse_idx is the number of unique key (the length of embedding_vec_)
  model_forward_.dense_forward(
      embedding_vec_,
      embedding_input.dense_compression_input.model_parallel_compression_input.model_reverse_idx,
      model_comm_buffer_, batch_size,
      embedding_input.dense_compression_input.model_parallel_compression_input
          .num_model_reverse_idx);
}

void DenseUniformModelParallelEmbedding::network_forward(const EmbeddingInput &embedding_input,
                                                         EmbeddingOutput &embedding_output,
                                                         int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  all2all_comm_.dense_communicate(
      model_comm_buffer_.data,
      embedding_input.dense_compression_input.model_parallel_compression_input.h_send_k_per_gpu,
      network_buffer_.data,
      embedding_input.dense_compression_input.model_parallel_compression_input.h_recv_k_per_gpu,
      meta_.ev_size_);
  network_forward_.dense_forward(embedding_input, network_buffer_, meta_.network_indices,
                                 embedding_output, batch_size, do_reduction_);
}

void DenseUniformModelParallelEmbedding::network_backward(const EmbeddingOutput &top_grad,
                                                          const EmbeddingInput &embedding_input,
                                                          Wgrad &wgrad, int batch_size) {
  network_backward_.dense_backward(embedding_input, top_grad, meta_.network_indices,
                                   network_buffer_, batch_size);

  all2all_comm_.dense_communicate(
      network_buffer_.data,
      embedding_input.dense_compression_input.model_parallel_compression_input.h_recv_k_per_gpu,
      model_comm_buffer_.data,
      embedding_input.dense_compression_input.model_parallel_compression_input.h_send_k_per_gpu,
      meta_.ev_size_);
}

void DenseUniformModelParallelEmbedding::backward_index_calculation(
    const EmbeddingInput &embedding_input, Wgrad &wgrad, int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());

  local_reduce_index_calculation_.cal_for_dense_input(embedding_input, reduction_indices_, wgrad,
                                                      meta_.ev_size_);
}

void DenseUniformModelParallelEmbedding::local_reduce(Wgrad &wgrad, int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());

  local_reduce_.local_reduce(reduction_indices_, model_comm_buffer_, wgrad);
}
// model forward(local lookup) is per-gpu, happens after data distributor does keys unique
// network forward is after local lookup, all embedding vectors must be all2all so as to get the dp
// embedding vectors
void DenseUniformModelParallelEmbedding::forward_per_gpu(Stage stage,
                                                         const EmbeddingInput &embedding_input,
                                                         ILookup *embedding_table,
                                                         EmbeddingOutput &embedding_output,
                                                         int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  switch (stage) {
    case Stage::DenseMPModelForward: {
      model_forward(embedding_input, embedding_table, batch_size);
    } break;
    case Stage::DenseMPNetworkForward: {
      network_forward(embedding_input, embedding_output, batch_size);
    } break;
    default:
      HCTR_OWN_THROW(
          HugeCTR::Error_t::IllegalCall,
          "stage is not supported in DenseUniformModelParallelEmbedding::forward_per_gpu");
  }
}

void DenseUniformModelParallelEmbedding::backward_per_gpu(Stage stage,
                                                          const EmbeddingInput &embedding_input,
                                                          const EmbeddingOutput &top_grad,
                                                          Wgrad &wgrad, int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());

  switch (stage) {
    case Stage::DenseMPBackwardIndexCalculation: {
      backward_index_calculation(embedding_input, wgrad, batch_size);
    } break;
    case Stage::DenseMPNetworkBackward: {
      network_backward(top_grad, embedding_input, wgrad, batch_size);
    } break;
    case Stage::DenseMPLocalReduce: {
      local_reduce(wgrad, batch_size);
    } break;
    default:
      HCTR_OWN_THROW(
          HugeCTR::Error_t::IllegalCall,
          "stage is not supported in DenseUniformModelParallelEmbedding::backward_per_gpu");
  }
}

bool DenseUniformModelParallelEmbedding::is_valid_stage(Stage stage) const {
  return (stage == Stage::DenseMPModelForward) || (stage == Stage::DenseMPNetworkForward) ||
         (stage == Stage::DenseMPBackwardIndexCalculation) ||
         (stage == Stage::DenseMPNetworkBackward) || (stage == Stage::DenseMPLocalReduce);
}

}  // namespace embedding
