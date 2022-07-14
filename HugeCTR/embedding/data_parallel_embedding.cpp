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

namespace embedding {

UniformDPEmbeddingForward::UniformDPEmbeddingForward(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &params, const GlobalEmbeddingData &global_embedding_data,
    const EmbeddingShardingParam &embedding_sharding_param) : core_(core), global_embedding_data_(global_embedding_data), local_embedding_data_(core, params, embedding_sharding_param){
  CudaDeviceContext context(core->get_device_id());

  int num_gpus = core_->get_global_gpu_count();
  int universal_batch_size = params.universal_batch_size;
  auto key_type = params.key_type;
  auto offset_type = params.offset_type;
  // auto emb_type = params.emb_type;

  embedding_vec_ = TensorList(core_.get(), universal_batch_size * local_embedding_data_.num_local_hotness_, DeviceType::GPU, TensorScalarType::Float32);

  // init op
  index_calculation_ = DPIndexCalculation(core_, num_gpus, local_embedding_data_.num_local_embedding_, local_embedding_data_.num_local_hotness_, global_embedding_data_.num_hotness_, universal_batch_size, key_type, offset_type);
  
  dp_local_reduce_index_calculation_ = DPLocalReduceIndexCalculation(core_, global_embedding_data_.num_embedding_, local_embedding_data_.num_local_embedding_, local_embedding_data_.h_local_hotness_list_, universal_batch_size, key_type);

  compress_offset_ = CompressOffset(core_, local_embedding_data_.num_local_embedding_ + 1);

  dp_model_forward_ = DPModelForward(core_,num_gpus, global_embedding_data_.num_embedding_, local_embedding_data_.num_local_embedding_);

}

void UniformDPEmbeddingForward::forward_per_gpu(const Tensor &keys, const Tensor &bucket_range, size_t num_keys,
                                                const Tensor &sparse_weight,
                                                ILookup *embedding_table, Tensor &output_buffer,
                                                ContextContainer *context_container) {
  CudaDeviceContext context(core_->get_device_id());
  int batch_size = (bucket_range.get_num_elements() - 1) / global_embedding_data_.num_embedding_;
  int batch_size_per_gpu = batch_size / core_->get_global_gpu_count();

  Tensor dp_key;
  Tensor dp_offset;
  size_t num_dp_key;
  Tensor dp_dst;
  index_calculation_.compute(keys, bucket_range, num_keys, local_embedding_data_.d_local_embedding_list_, batch_size, &dp_key, &dp_offset, &num_dp_key, &dp_dst);

  Tensor id_space_offset;
  compress_offset_.compute(dp_offset, batch_size_per_gpu, &id_space_offset);

  Tensor unique_key, unique_dst_idx, sorted_bucket_id_list, sorted_bucket_id_offset, unique_id_space_offset;
  size_t num_unique_key;
  dp_local_reduce_index_calculation_.compute(keys, num_keys, bucket_range, local_embedding_data_.d_local_embedding_list_, local_embedding_data_.d_local_id_space_list_, local_embedding_data_.d_local_ev_size_list_, batch_size, &unique_key, &num_unique_key, &unique_dst_idx, &sorted_bucket_id_list, &sorted_bucket_id_offset, &unique_id_space_offset);

  embedding_table->lookup(dp_key, num_dp_key, id_space_offset, local_embedding_data_.num_local_embedding_ + 1, local_embedding_data_.d_local_id_space_list_, embedding_vec_);

  dp_model_forward_.compute(embedding_vec_, dp_offset, dp_dst, output_buffer, local_embedding_data_.d_local_ev_size_list_, local_embedding_data_.d_local_combiner_list_, global_embedding_data_.d_ev_size_offset_, batch_size);

  context_container->pack("dp_key", dp_key);
  context_container->pack("dp_offset", dp_offset);
  context_container->pack("num_dp_key", num_dp_key);

  context_container->pack("batch_size", batch_size);
  context_container->pack("unique_key", unique_key);
  context_container->pack("num_unique_key", num_unique_key);
  context_container->pack("unique_dst_idx", unique_dst_idx);
  context_container->pack("sorted_bucket_id_list", sorted_bucket_id_list);
  context_container->pack("sorted_bucket_id_offset", sorted_bucket_id_offset);
  context_container->pack("unique_id_space_list", local_embedding_data_.d_local_id_space_list_);
  context_container->pack("unique_id_space_offset", unique_id_space_offset);
}

UniformDPEmbeddingBackward::UniformDPEmbeddingBackward(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &params, const GlobalEmbeddingData &global_embedding_data,
    const EmbeddingShardingParam &embedding_sharding_param) : core_(core), global_embedding_data_(global_embedding_data), local_embedding_data_(core, params, embedding_sharding_param) {
  CudaDeviceContext context(core_->get_device_id());

  int num_gpus = core_->get_global_gpu_count();
  dp_local_reduce_ = DPLocalReduce(core_, num_gpus, local_embedding_data_.num_local_embedding_, local_embedding_data_.h_local_hotness_list_, local_embedding_data_.h_local_ev_size_list_, params.universal_batch_size);
  allreduce_comm_ = NcclAllReduceInplaceComm(core_);
}

void UniformDPEmbeddingBackward::backward_per_gpu(ContextContainer *context_container,
                                                  const Tensor &top_grad, bool do_allreduce,
                                                  Tensor *unique_key, size_t *num_unique_key,
                                                  Tensor *unique_id_space_offset,
                                                  size_t *num_unique_key_id_space_offset,
                                                  Tensor *grad_ev, Tensor *unique_dst_idx) {
  CudaDeviceContext context(core_->get_device_id());
  
  int batch_size = context_container->unpack<int>("batch_size");
  *unique_key = context_container->unpack<Tensor>("unique_key");
  *num_unique_key = context_container->unpack<size_t>("num_unique_key");
  *unique_id_space_offset = context_container->unpack<Tensor>("unique_id_space_offset");
  *num_unique_key_id_space_offset = unique_id_space_offset->get_num_elements();
  *unique_dst_idx = context_container->unpack<Tensor>("unique_dst_idx");
  auto sorted_bucket_id_list = context_container->unpack<Tensor>("sorted_bucket_id_list");
  auto sorted_bucket_id_offset = context_container->unpack<Tensor>("sorted_bucket_id_offset");
  auto d_ev_size_offset = global_embedding_data_.d_ev_size_offset_;

  dp_local_reduce_.compute(top_grad, *unique_dst_idx, sorted_bucket_id_list, sorted_bucket_id_offset, *num_unique_key, d_ev_size_offset, batch_size, grad_ev);

  if (do_allreduce) {
    allreduce_comm_.communicate(*grad_ev, grad_ev->get_num_elements());
  }

}
}  // namespace embedding
