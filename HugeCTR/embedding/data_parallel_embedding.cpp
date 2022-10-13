/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include "data_parallel_embedding.hpp"

#include "HugeCTR/include/utils.hpp"
namespace embedding {

UniformDPEmbedding::UniformDPEmbedding(std::shared_ptr<CoreResourceManager> core,
                                       const EmbeddingCollectionParam& params, size_t grouped_id)
    : core_(core), meta_(core, params, grouped_id) {
  HugeCTR::CudaDeviceContext context(core->get_device_id());

  int num_gpus = core_->get_global_gpu_count();
  int universal_batch_size = params.universal_batch_size;
  auto key_type = params.key_type;
  auto offset_type = params.offset_type;
  // auto emb_type = params.emb_type;

  // init op
  index_calculation_ =
      DPIndexCalculation(core_, num_gpus, meta_.num_local_lookup_, meta_.num_local_hotness_,
                         meta_.num_hotness_, universal_batch_size, key_type, offset_type);

  dp_local_reduce_index_calculation_ =
      DPLocalReduceIndexCalculation(core_, meta_.num_lookup_, meta_.num_local_lookup_,
                                    meta_.h_local_hotness_list_, universal_batch_size, key_type);

  compress_offset_ = CompressOffset(core_, meta_.num_local_lookup_ + 1);

  dp_model_forward_ = DPModelForward(core_, num_gpus, meta_.num_lookup_, meta_.num_local_lookup_);

  dp_local_reduce_ =
      DPLocalReduce(core_, num_gpus, meta_.num_local_lookup_, meta_.h_local_hotness_list_,
                    meta_.h_local_ev_size_list_, params.universal_batch_size);
  allreduce_comm_ = NcclAllReduceInplaceComm(core_);
  if (std::find(meta_.h_local_combiner_list_.begin(), meta_.h_local_combiner_list_.end(),
                static_cast<char>(Combiner::Average)) != meta_.h_local_combiner_list_.end()) {
    average_combiner_ = AverageCombiner(core, num_gpus, meta_.num_local_lookup_,
                                        meta_.h_ev_size_list_, params.universal_batch_size);
  }

  embedding_vec_ = TensorList(core_.get(), universal_batch_size * meta_.num_local_hotness_,
                              DeviceType::GPU, TensorScalarType::Float32);
}

void UniformDPEmbedding::forward_per_gpu(const Tensor& keys, const Tensor& bucket_range,
                                         size_t num_keys, ILookup* embedding_table,
                                         Tensor& output_buffer, int batch_size) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  batch_size_ = batch_size;
  keys_ = keys;
  num_keys_ = num_keys;
  bucket_range_ = bucket_range;
  int batch_size_per_gpu = batch_size / core_->get_global_gpu_count();

  Tensor dp_key;
  Tensor dp_offset;
  size_t num_dp_key;
  Tensor dp_dst;
  index_calculation_.compute(keys, bucket_range, num_keys, meta_.d_local_lookup_id_list_,
                             batch_size, &dp_key, &dp_offset, &num_dp_key, &dp_dst);

  Tensor num_key_per_lookup_offset;
  compress_offset_.compute(dp_offset, batch_size_per_gpu, &num_key_per_lookup_offset);

  embedding_table->lookup(dp_key, num_dp_key, num_key_per_lookup_offset,
                          meta_.num_local_lookup_ + 1, meta_.d_local_table_id_list_,
                          embedding_vec_);
  dp_model_forward_.compute(bucket_range, meta_.d_combiner_list_, embedding_vec_, dp_offset, dp_dst,
                            output_buffer, meta_.d_local_ev_size_list_, meta_.d_ev_size_offset_,
                            batch_size, meta_.max_ev_size_);
}

void UniformDPEmbedding::backward_per_gpu(const Tensor& top_grad, bool do_allreduce,
                                          Tensor* unique_key, size_t* num_unique_key,
                                          Tensor* num_unique_key_per_table_offset,
                                          size_t* num_table_offset, Tensor* table_id_list,
                                          Tensor* wgrad, Tensor* wgrad_idx_offset) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());

  Tensor sorted_bucket_id_list, sorted_bucket_id_offset;
  dp_local_reduce_index_calculation_.compute(
      keys_, num_keys_, bucket_range_, meta_.d_local_lookup_id_list_, meta_.d_local_table_id_list_,
      meta_.d_local_ev_size_list_, batch_size_, unique_key, num_unique_key, wgrad_idx_offset,
      &sorted_bucket_id_list, &sorted_bucket_id_offset, num_unique_key_per_table_offset);
  *num_table_offset = num_unique_key_per_table_offset->get_num_elements();

  auto d_ev_size_offset = meta_.d_ev_size_offset_;
  if (std::find(meta_.h_local_combiner_list_.begin(), meta_.h_local_combiner_list_.end(),
                static_cast<char>(Combiner::Average)) != meta_.h_local_combiner_list_.end()) {
    average_combiner_.compute(bucket_range_, top_grad, meta_.d_local_lookup_id_list_,
                              meta_.d_combiner_list_, meta_.d_ev_size_offset_, batch_size_,
                              meta_.max_ev_size_);
    dp_local_reduce_.compute(average_combiner_.float_emb_vec_, *wgrad_idx_offset,
                             sorted_bucket_id_list, sorted_bucket_id_offset, *num_unique_key,
                             d_ev_size_offset, batch_size_, meta_.max_ev_size_, wgrad);
  } else {
    dp_local_reduce_.compute(top_grad, *wgrad_idx_offset, sorted_bucket_id_list,
                             sorted_bucket_id_offset, *num_unique_key, d_ev_size_offset,
                             batch_size_, meta_.max_ev_size_, wgrad);
  }

  if (do_allreduce) {
    cudaStream_t stream = core_->get_local_gpu()->get_stream();
    uint32_t num_ev_elements;
    HCTR_LIB_THROW(cudaMemcpyAsync(&num_ev_elements,
                                   wgrad_idx_offset->get<uint32_t>() + *num_unique_key,
                                   sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    allreduce_comm_.communicate(*wgrad, num_ev_elements);
  }

  *table_id_list = meta_.d_local_table_id_list_;
}
}  // namespace embedding