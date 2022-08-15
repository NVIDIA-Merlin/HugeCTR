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
#include <cub/cub.cuh>

#include "HugeCTR/include/utils.hpp"
#include "all2all_embedding_collection.hpp"
namespace embedding {
namespace tf {

namespace {

template <typename offset_t>
__global__ void reorder_row_lengths_kernel(const offset_t *row_lengths, int num_row_lengths,
                                           offset_t *bucket_range, int batch_size_per_gpu,
                                           int num_gpu, int num_embedding) {
  for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < num_row_lengths;
       tid += blockDim.x * gridDim.x) {
    int gpu_id = tid / (batch_size_per_gpu * num_embedding);
    int embedding_id = (tid / batch_size_per_gpu) % num_embedding;
    int batch_id = tid % batch_size_per_gpu;

    int reorder_id =
        embedding_id * batch_size_per_gpu * num_gpu + gpu_id * batch_size_per_gpu + batch_id;
    bucket_range[1 + reorder_id] = row_lengths[tid];
  }
  if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
    bucket_range[0] = 0;
  }
}

template <typename key_t, typename offset_t>
__global__ void reorder_key_kernel(const key_t *key, const offset_t *row_offsets,
                                   int num_row_lengths, const offset_t *bucket_range,
                                   key_t *reorder_key, int batch_size_per_gpu, int num_gpu,
                                   int num_embedding) {
  for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < num_row_lengths;
       tid += blockDim.x * gridDim.x) {
    int gpu_id = tid / (batch_size_per_gpu * num_embedding);
    int embedding_id = (tid / batch_size_per_gpu) % num_embedding;
    int batch_id = tid % batch_size_per_gpu;

    int reorder_id =
        embedding_id * batch_size_per_gpu * num_gpu + gpu_id * batch_size_per_gpu + batch_id;
    offset_t start = (tid == 0) ? 0 : row_offsets[tid];
    offset_t end = row_offsets[tid + 1];
    for (offset_t r = 0; r < (end - start); ++r) {
      reorder_key[bucket_range[reorder_id] + r] = key[start + r];
    }
  }
}

}  // namespace

All2AllEmbeddingCollectionSwizzleKey::All2AllEmbeddingCollectionSwizzleKey(
    std::shared_ptr<CoreResourceManager> core)
    : core_(core) {}

void All2AllEmbeddingCollectionSwizzleKey::sparse_forward_per_gpu(
    const std::vector<Tensor> &keys, const std::vector<Tensor> &row_lengths,
    Tensor &key_all_gather_send_buffer, Tensor &row_lengths_all_gather_send_buffer) {
  size_t key_bytes_offset = 0;
  size_t row_lengths_bytes_offset = 0;
  key_all_gather_send_buffer.get();
  for (size_t i = 0; i < keys.size(); ++i) {
    keys[i].get();
    HCTR_LIB_THROW(cudaMemcpyAsync(
        reinterpret_cast<char *>(key_all_gather_send_buffer.get()) + key_bytes_offset,
        keys[i].get(), keys[i].nbytes(), cudaMemcpyDeviceToDevice,
        core_->get_local_gpu()->get_stream()));
    key_bytes_offset += keys[i].nbytes();

    HCTR_LIB_THROW(
        cudaMemcpyAsync(reinterpret_cast<char *>(row_lengths_all_gather_send_buffer.get()) +
                            row_lengths_bytes_offset,
                        row_lengths[i].get(), row_lengths[i].nbytes(), cudaMemcpyDeviceToDevice,
                        core_->get_local_gpu()->get_stream()));
    row_lengths_bytes_offset += row_lengths[i].nbytes();
  }
}

All2AllEmbeddingCollectionModelForward::All2AllEmbeddingCollectionModelForward(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
    const EmbeddingShardParam &shard_param)
    : core_(core),
      global_embedding_data_(core, ebc_param),
      local_embedding_data_(core, ebc_param, shard_param) {}

std::vector<size_t> All2AllEmbeddingCollectionModelForward::get_model_comm_buffer_size(
    int batch_size) {
  int num_gpus = core_->get_global_gpu_count();
  size_t num_ev_elements = 0;
  int batch_size_per_gpu = batch_size / num_gpus;
  for (int embedding_id : local_embedding_data_.h_local_embedding_list_) {
    int ev_size = global_embedding_data_.h_ev_size_list_[embedding_id];
    num_ev_elements += ev_size * batch_size_per_gpu;
  }
  return std::vector<size_t>(num_gpus, num_ev_elements);
}

void All2AllEmbeddingCollectionModelForward::sparse_forward_per_gpu(
    const Tensor &key_all_gather_recv_buffer, const Tensor &row_lengths_all_gather_recv_buffer,
    ILookup *emb_storage, std::vector<Tensor> &emb_vec_model_buffer, int64_t *num_model_key,
    int64_t *num_model_offsets) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());

  int num_gpus = core_->get_global_gpu_count();
  cudaStream_t stream = core_->get_local_gpu()->get_stream();
  int batch_size =
      row_lengths_all_gather_recv_buffer.get_num_elements() / global_embedding_data_.num_embedding_;

  Tensor keys, bucket_range;
  size_t num_keys = static_cast<size_t>(key_all_gather_recv_buffer.get_num_elements());
  // the shape of key_all_gather is (num_gpus, num_embedding, batch_size_per_gpu)
  // the shape of key is (num_embedding, batch_size)
  auto reorder_from_all_gather_input = [&] {
    Tensor all_gather_row_offsets;

    auto buffer_ptr = GetBuffer(core_);
    keys = buffer_ptr->reserve({key_all_gather_recv_buffer.get_num_elements()},
                               key_all_gather_recv_buffer.device(),
                               key_all_gather_recv_buffer.dtype());
    bucket_range = buffer_ptr->reserve({row_lengths_all_gather_recv_buffer.get_num_elements() + 1},
                                       row_lengths_all_gather_recv_buffer.device(),
                                       row_lengths_all_gather_recv_buffer.dtype());
    all_gather_row_offsets = buffer_ptr->reserve(
        {row_lengths_all_gather_recv_buffer.get_num_elements() + 1},
        row_lengths_all_gather_recv_buffer.device(), row_lengths_all_gather_recv_buffer.dtype());
    buffer_ptr->allocate();

    auto get_bucket_range = [&] {
      DISPATCH_INTEGRAL_FUNCTION(row_lengths_all_gather_recv_buffer.dtype().type(), offset_t, [&] {
        constexpr int block_size = 256;
        int grid_size =
            (row_lengths_all_gather_recv_buffer.get_num_elements() - 1) / block_size + 1;

        reorder_row_lengths_kernel<<<grid_size, block_size, 0, stream>>>(
            row_lengths_all_gather_recv_buffer.get<offset_t>(),
            row_lengths_all_gather_recv_buffer.get_num_elements(), bucket_range.get<offset_t>(),
            batch_size / num_gpus, num_gpus, global_embedding_data_.num_embedding_);

        size_t temp_bytes = 0;
        Tensor temp_scan_storage;
        cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, (offset_t *)nullptr, (offset_t *)nullptr,
                                      bucket_range.get_num_elements());
        temp_scan_storage =
            buffer_ptr->reserve(temp_bytes, DeviceType::GPU, TensorScalarType::Void);
        buffer_ptr->allocate();

        cub::DeviceScan::InclusiveSum(temp_scan_storage.get(), temp_bytes,
                                      bucket_range.get<offset_t>(), bucket_range.get<offset_t>(),
                                      bucket_range.get_num_elements(), stream);

        // HCTR_LIB_THROW(cudaStreamSynchronize(stream));

        // std::vector<offset_t> gpu_bucket_range;
        // bucket_range.to(&gpu_bucket_range);
        // std::cout << "gpu_bucket_range:\n";
        // for (auto i : gpu_bucket_range) {
        //   std::cout << i << " ";
        // }
        // std::cout << "\n";
      });
    };

    auto scan_row_lengths = [&] {
      DISPATCH_INTEGRAL_FUNCTION(row_lengths_all_gather_recv_buffer.dtype().type(), offset_t, [&] {
        size_t temp_bytes = 0;
        Tensor temp_scan_storage;
        cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, (offset_t *)nullptr, (offset_t *)nullptr,
                                      row_lengths_all_gather_recv_buffer.get_num_elements() + 1);
        temp_scan_storage =
            buffer_ptr->reserve(temp_bytes, DeviceType::GPU, TensorScalarType::Void);
        buffer_ptr->allocate();

        cub::DeviceScan::InclusiveSum(
            temp_scan_storage.get(), temp_bytes, row_lengths_all_gather_recv_buffer.get<offset_t>(),
            all_gather_row_offsets.get<offset_t>() + 1,
            row_lengths_all_gather_recv_buffer.get_num_elements(), stream);
        // HCTR_LIB_THROW(cudaStreamSynchronize(stream));

        // std::vector<offset_t> gpu_row_lengths;
        // row_lengths_all_gather_recv_buffer.to(&gpu_row_lengths);
        // std::cout << "gpu_row_lengths:\n";
        // for (auto i : gpu_row_lengths) {
        //   std::cout << i << " ";
        // }
        // std::cout << "\n";

        // std::vector<offset_t> gpu_row_offsets;
        // all_gather_row_offsets.to(&gpu_row_offsets);
        // std::cout << "gpu_row_offsets:\n";
        // for (auto i : gpu_row_offsets) {
        //   std::cout << i << " ";
        // }
        // std::cout << "\n";
      });
    };

    auto reorder_key = [&] {
      DISPATCH_INTEGRAL_FUNCTION(key_all_gather_recv_buffer.dtype().type(), key_t, [&] {
        DISPATCH_INTEGRAL_FUNCTION(
            row_lengths_all_gather_recv_buffer.dtype().type(), offset_t, [&] {
              constexpr int block_size = 256;
              int grid_size =
                  (row_lengths_all_gather_recv_buffer.get_num_elements() - 1) / block_size + 1;
              reorder_key_kernel<<<grid_size, block_size, 0, stream>>>(
                  key_all_gather_recv_buffer.get<key_t>(), all_gather_row_offsets.get<offset_t>(),
                  row_lengths_all_gather_recv_buffer.get_num_elements(),
                  bucket_range.get<offset_t>(), keys.get<key_t>(), batch_size / num_gpus, num_gpus,
                  global_embedding_data_.num_embedding_);
              // HCTR_LIB_THROW(cudaStreamSynchronize(stream));

              // std::vector<key_t> gpu_all_gather_key;
              // key_all_gather_recv_buffer.to(&gpu_all_gather_key);
              // std::cout << "gpu_all_gather_key:\n";
              // for (auto i : gpu_all_gather_key) {
              //   std::cout << i << " ";
              // }
              // std::cout << "\n";

              // std::vector<offset_t> gpu_reorder_key;
              // keys.to(&gpu_reorder_key);
              // std::cout << "gpu_reorder_key:\n";
              // for (auto i : gpu_reorder_key) {
              //   std::cout << i << " ";
              // }
              // std::cout << "\n";
            });
      });
    };

    get_bucket_range();
    scan_row_lengths();
    reorder_key();
  };
  reorder_from_all_gather_input();

  DataType key_type = key_all_gather_recv_buffer.dtype();
  model_index_calculation_ =
      ModelIndexCalculation(core_, local_embedding_data_.num_local_embedding_,
                            local_embedding_data_.h_local_hotness_list_,
                            global_embedding_data_.h_hotness_list_, batch_size, key_type);

  Tensor model_key, model_offsets;
  size_t num_model_key_;
  model_index_calculation_.compute(
      keys, bucket_range, num_keys, local_embedding_data_.d_local_embedding_list_,
      local_embedding_data_.shard_id_, local_embedding_data_.shards_count_, batch_size, &model_key,
      &model_offsets, &num_model_key_);

  compress_offset_ = CompressOffset(core_, local_embedding_data_.num_local_embedding_ + 1);
  Tensor id_space_offset;
  compress_offset_.compute(model_offsets, batch_size, &id_space_offset);

  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  TensorList embedding_vec = TensorList(core_.get(), key_all_gather_recv_buffer.get_num_elements(),
                                        DeviceType::GPU, TensorScalarType::Float32);
  emb_storage->lookup(model_key, num_model_key_, id_space_offset,
                      local_embedding_data_.num_local_embedding_ + 1,
                      local_embedding_data_.d_local_id_space_list_, embedding_vec);

  model_forward_ = ModelForward(core_, num_gpus, local_embedding_data_.h_local_embedding_list_);

  TensorList model_comm_buffer{core_.get(), emb_vec_model_buffer, DeviceType::GPU,
                               emb_vec_model_buffer[0].dtype(), stream};
  model_forward_.compute(embedding_vec, model_offsets, model_comm_buffer,
                         local_embedding_data_.d_local_ev_size_list_,
                         local_embedding_data_.d_local_ev_size_offset_, batch_size,
                         local_embedding_data_.max_ev_size_);

  model_key_ = model_key;
  model_offsets_ = model_offsets;
  *num_model_key = static_cast<int64_t>(num_model_key_);
  *num_model_offsets = model_offsets.get_num_elements();
}

void All2AllEmbeddingCollectionModelForward::copy_model_keys_and_offsets(Tensor &model_key,
                                                                         Tensor &model_offsets) {
  HCTR_LIB_THROW(cudaMemcpyAsync(model_key.get(), model_key_.get(), model_key.nbytes(),
                                 cudaMemcpyDeviceToDevice, core_->get_local_gpu()->get_stream()));
  HCTR_LIB_THROW(cudaMemcpyAsync(model_offsets.get(), model_offsets_.get(), model_offsets.nbytes(),
                                 cudaMemcpyDeviceToDevice, core_->get_local_gpu()->get_stream()));
}

All2AllEmbeddingCollectionNetworkForward::All2AllEmbeddingCollectionNetworkForward(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
    const EmbeddingShardParam &shard_param)
    : core_(core),
      global_embedding_data_(core, ebc_param),
      local_embedding_data_(core, ebc_param, shard_param),
      sharding_param_(ebc_param.num_embedding, shard_param, core->get_global_gpu_id()) {
  int num_gpus = core->get_global_gpu_count();
  network_forward_ = NetworkForward(core, num_gpus);
}

void All2AllEmbeddingCollectionNetworkForward::sparse_forward_per_gpu(
    const std::vector<Tensor> &emb_vec_network_buffer, const std::vector<Tensor> &row_lengths,
    std::vector<Tensor> &forward_emb_vec) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  cudaStream_t stream = core_->get_local_gpu()->get_stream();
  int num_gpus = core_->get_global_gpu_count();
  int batch_size_per_gpu = row_lengths[0].get_num_elements();
  int batch_size = batch_size_per_gpu * num_gpus;
  int global_gpu_id = core_->get_global_gpu_id();

  Tensor bucket_range;
  auto copy_needed_bucket_range = [&] {
    DISPATCH_INTEGRAL_FUNCTION(row_lengths[0].dtype().type(), offset_t, [&] {
      auto buffer_ptr = GetBuffer(core_);
      bucket_range = buffer_ptr->reserve(1 + batch_size * global_embedding_data_.num_embedding_,
                                         DeviceType::GPU, row_lengths[0].dtype());
      buffer_ptr->allocate();

      HCTR_LIB_THROW(
          cudaMemsetAsync(bucket_range.get<offset_t>(), 0, bucket_range.nbytes(), stream));
      for (int embedding_id = 0; embedding_id < global_embedding_data_.num_embedding_;
           ++embedding_id) {
        HCTR_LIB_THROW(cudaMemcpyAsync(bucket_range.get<offset_t>() + batch_size * embedding_id +
                                           global_gpu_id * batch_size_per_gpu + 1,
                                       row_lengths[embedding_id].get<offset_t>(),
                                       row_lengths[embedding_id].nbytes(), cudaMemcpyDeviceToDevice,
                                       stream));
      }

      size_t temp_bytes = 0;
      Tensor temp_scan_storage;
      cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, (offset_t *)nullptr, (offset_t *)nullptr,
                                    bucket_range.get_num_elements());
      temp_scan_storage = buffer_ptr->reserve(temp_bytes, DeviceType::GPU, TensorScalarType::Void);
      buffer_ptr->allocate();

      cub::DeviceScan::InclusiveSum(temp_scan_storage.get(), temp_bytes,
                                    bucket_range.get<offset_t>(), bucket_range.get<offset_t>(),
                                    bucket_range.get_num_elements(), stream);
    });
  };

  Tensor output_buffer;
  auto allocate_continous_output_buffer = [&] {
    int num_output_elements = 0;
    for (auto &t : forward_emb_vec) {
      num_output_elements += t.get_num_elements();
    }
    auto buffer_ptr = GetBuffer(core_);
    output_buffer =
        buffer_ptr->reserve(num_output_elements, DeviceType::GPU, forward_emb_vec[0].dtype());
    buffer_ptr->allocate();
  };

  auto copy_back_fwd_result = [&] {
    size_t nbytes_offset = 0ul;
    for (int embedding_id = 0; embedding_id < global_embedding_data_.num_embedding_;
         ++embedding_id) {
      HCTR_LIB_THROW(cudaMemcpyAsync(forward_emb_vec[embedding_id].get(),
                                     reinterpret_cast<char *>(output_buffer.get()) + nbytes_offset,
                                     forward_emb_vec[embedding_id].nbytes(),
                                     cudaMemcpyDeviceToDevice, stream));
      nbytes_offset += forward_emb_vec[embedding_id].nbytes();
    }
  };
  copy_needed_bucket_range();
  allocate_continous_output_buffer();

  RaggedNetworkIndex ragged_network_index{core_, batch_size, sharding_param_.global_embedding_list,
                                          global_embedding_data_.h_ev_size_list_};

  TensorList network_comm_buffer{core_.get(), emb_vec_network_buffer, DeviceType::GPU,
                                 emb_vec_network_buffer[0].dtype(), stream};
  if (std::find(local_embedding_data_.h_network_combiner_list_.begin(),
                local_embedding_data_.h_network_combiner_list_.end(),
                static_cast<char>(Combiner::Average)) !=
      local_embedding_data_.h_network_combiner_list_.end()) {
    AverageCominber average_combiner{
        core_, num_gpus, static_cast<int>(local_embedding_data_.h_network_embedding_list_.size()),
        global_embedding_data_.h_ev_size_list_, batch_size};
    network_forward_.compute(
        network_comm_buffer, ragged_network_index.gpu_idx_offset_,
        ragged_network_index.global_ev_offset_, ragged_network_index.network_idx_,
        ragged_network_index.network_offset_, ragged_network_index.network_dst_,
        average_combiner.float_emb_vec_, global_embedding_data_.d_ev_size_offset_, batch_size,
        global_embedding_data_.max_ev_size_);

    average_combiner.forward(
        bucket_range, output_buffer, local_embedding_data_.d_network_embedding_list_,
        global_embedding_data_.d_combiner_list_, global_embedding_data_.d_ev_size_offset_,
        batch_size, global_embedding_data_.max_ev_size_);
  } else {
    network_forward_.compute(
        network_comm_buffer, ragged_network_index.gpu_idx_offset_,
        ragged_network_index.global_ev_offset_, ragged_network_index.network_idx_,
        ragged_network_index.network_offset_, ragged_network_index.network_dst_, output_buffer,
        global_embedding_data_.d_ev_size_offset_, batch_size, global_embedding_data_.max_ev_size_);
  }

  copy_back_fwd_result();
}

All2AllEmbeddingCollectionNetworkBackward::All2AllEmbeddingCollectionNetworkBackward(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
    const EmbeddingShardParam &shard_param)
    : core_(core),
      global_embedding_data_(core, ebc_param),
      local_embedding_data_(core, ebc_param, shard_param),
      sharding_param_(ebc_param.num_embedding, shard_param, core->get_global_gpu_id()) {
  int num_gpus = core->get_global_gpu_count();
  network_backward_ = NetworkBackward(core, num_gpus);
}

void All2AllEmbeddingCollectionNetworkBackward::backward_per_gpu(
    const std::vector<Tensor> &top_grad, const std::vector<Tensor> &row_lengths,
    std::vector<Tensor> &emb_vec_network_buffer) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  cudaStream_t stream = core_->get_local_gpu()->get_stream();
  // int batch_size = (top_grad[0].get_num_elements() * core_->get_global_gpu_count()) /
  // global_embedding_data_.h_ev_size_list_[0];
  int num_gpus = core_->get_global_gpu_count();
  int batch_size_per_gpu = row_lengths[0].get_num_elements();
  int batch_size = batch_size_per_gpu * num_gpus;
  int global_gpu_id = core_->get_global_gpu_id();

  Tensor bucket_range;
  auto copy_needed_bucket_range = [&] {
    DISPATCH_INTEGRAL_FUNCTION(row_lengths[0].dtype().type(), offset_t, [&] {
      auto buffer_ptr = GetBuffer(core_);
      bucket_range = buffer_ptr->reserve(1 + batch_size * global_embedding_data_.num_embedding_,
                                         DeviceType::GPU, row_lengths[0].dtype());
      buffer_ptr->allocate();

      HCTR_LIB_THROW(
          cudaMemsetAsync(bucket_range.get<offset_t>(), 0, bucket_range.nbytes(), stream));
      for (int embedding_id = 0; embedding_id < global_embedding_data_.num_embedding_;
           ++embedding_id) {
        HCTR_LIB_THROW(cudaMemcpyAsync(bucket_range.get<offset_t>() + batch_size * embedding_id +
                                           global_gpu_id * batch_size_per_gpu + 1,
                                       row_lengths[embedding_id].get<offset_t>(),
                                       row_lengths[embedding_id].nbytes(), cudaMemcpyDeviceToDevice,
                                       stream));
      }

      size_t temp_bytes = 0;
      Tensor temp_scan_storage;
      cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, (offset_t *)nullptr, (offset_t *)nullptr,
                                    bucket_range.get_num_elements());
      temp_scan_storage = buffer_ptr->reserve(temp_bytes, DeviceType::GPU, TensorScalarType::Void);
      buffer_ptr->allocate();

      cub::DeviceScan::InclusiveSum(temp_scan_storage.get(), temp_bytes,
                                    bucket_range.get<offset_t>(), bucket_range.get<offset_t>(),
                                    bucket_range.get_num_elements(), stream);
    });
  };

  Tensor continous_top_grad;
  auto allocate_and_copy_continous_top_grad = [&] {
    int num_output_elements = 0;
    for (auto &t : top_grad) {
      num_output_elements += t.get_num_elements();
    }
    auto buffer_ptr = GetBuffer(core_);
    continous_top_grad =
        buffer_ptr->reserve(num_output_elements, DeviceType::GPU, top_grad[0].dtype());
    buffer_ptr->allocate();

    size_t nbytes_offset = 0ul;
    for (int embedding_id = 0; embedding_id < global_embedding_data_.num_embedding_;
         ++embedding_id) {
      HCTR_LIB_THROW(
          cudaMemcpyAsync(reinterpret_cast<char *>(continous_top_grad.get()) + nbytes_offset,
                          top_grad[embedding_id].get(), top_grad[embedding_id].nbytes(),
                          cudaMemcpyDeviceToDevice, stream));
      nbytes_offset += top_grad[embedding_id].nbytes();
    }
  };
  copy_needed_bucket_range();
  allocate_and_copy_continous_top_grad();

  RaggedNetworkIndex ragged_network_index{core_, batch_size, sharding_param_.global_embedding_list,
                                          global_embedding_data_.h_ev_size_list_};

  TensorList network_comm_buffer{core_.get(), emb_vec_network_buffer, DeviceType::GPU,
                                 emb_vec_network_buffer[0].dtype(), stream};

  if (std::find(local_embedding_data_.h_network_combiner_list_.begin(),
                local_embedding_data_.h_network_combiner_list_.end(),
                static_cast<char>(Combiner::Average)) !=
      local_embedding_data_.h_network_combiner_list_.end()) {
    AverageCominber average_combiner{
        core_, num_gpus, static_cast<int>(local_embedding_data_.h_network_embedding_list_.size()),
        global_embedding_data_.h_ev_size_list_, batch_size};
    average_combiner.backward(
        bucket_range, continous_top_grad, local_embedding_data_.d_network_embedding_list_,
        global_embedding_data_.d_combiner_list_, global_embedding_data_.d_ev_size_offset_,
        batch_size, global_embedding_data_.max_ev_size_);
    network_backward_.compute(
        average_combiner.float_emb_vec_, global_embedding_data_.d_ev_size_offset_,
        ragged_network_index.gpu_idx_offset_, ragged_network_index.global_ev_offset_,
        ragged_network_index.network_idx_, ragged_network_index.network_offset_,
        ragged_network_index.network_dst_, network_comm_buffer, batch_size,
        global_embedding_data_.max_ev_size_);
  } else {
    network_backward_.compute(
        continous_top_grad, global_embedding_data_.d_ev_size_offset_,
        ragged_network_index.gpu_idx_offset_, ragged_network_index.global_ev_offset_,
        ragged_network_index.network_idx_, ragged_network_index.network_offset_,
        ragged_network_index.network_dst_, network_comm_buffer, batch_size,
        global_embedding_data_.max_ev_size_);
  }
}

All2AllEmbeddingCollectionModelBackward::All2AllEmbeddingCollectionModelBackward(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
    const EmbeddingShardParam &shard_param)
    : core_(core),
      global_embedding_data_(core, ebc_param),
      local_embedding_data_(core, ebc_param, shard_param) {}

void All2AllEmbeddingCollectionModelBackward::sparse_backward_per_gpu(
    const std::vector<Tensor> &emb_vec_model_buffer, const Tensor &model_key,
    const Tensor &model_offsets, std::vector<int> *num_unique_key_per_table,
    std::vector<int> *unique_id_space_list) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  int num_gpus = core_->get_global_gpu_count();
  cudaStream_t stream = core_->get_local_gpu()->get_stream();
  int batch_size =
      (model_offsets.get_num_elements() - 1) / local_embedding_data_.num_local_embedding_;
  size_t num_model_key = static_cast<size_t>(model_key.get_num_elements());

  Tensor id_space_offset;
  CompressOffset compress_offset{core_, local_embedding_data_.num_local_embedding_ + 1};
  compress_offset.compute(model_offsets, batch_size, &id_space_offset);

  model_backward_index_calculation_ = ModelBackwardIndexCalculation(
      core_, num_gpus, local_embedding_data_.num_local_embedding_,
      local_embedding_data_.h_local_hotness_list_, local_embedding_data_.h_local_id_space_list_,
      local_embedding_data_.h_local_ev_size_list_, batch_size, model_key.dtype());

  Tensor continous_unique_key, unique_dst_idx, sorted_bucket_id_list, sorted_bucket_id_offset,
      unique_id_space, unique_id_space_offset, continous_grad_emb_ev;
  size_t num_unique_key;
  model_backward_index_calculation_.compute(
      model_key, num_model_key, model_offsets, id_space_offset,
      local_embedding_data_.d_local_id_space_list_, batch_size, &continous_unique_key,
      &num_unique_key, &unique_dst_idx, &sorted_bucket_id_list, &sorted_bucket_id_offset,
      &unique_id_space, &unique_id_space_offset);

  model_backward_ = ModelBackward(core_, num_gpus, local_embedding_data_.num_local_embedding_,
                                  local_embedding_data_.h_local_hotness_list_,
                                  local_embedding_data_.h_local_ev_size_list_, batch_size);

  TensorList model_comm_buffer{core_.get(), emb_vec_model_buffer, DeviceType::GPU,
                               emb_vec_model_buffer[0].dtype(), stream};
  model_backward_.compute(model_comm_buffer, unique_dst_idx, sorted_bucket_id_list,
                          sorted_bucket_id_offset, num_unique_key,
                          local_embedding_data_.d_local_ev_size_offset_, batch_size,
                          local_embedding_data_.max_ev_size_, &continous_grad_emb_ev);

  unique_id_space.to(unique_id_space_list, stream);
  continous_unique_key_ = continous_unique_key;
  continous_emb_vec_ = continous_grad_emb_ev;
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  std::vector<uint32_t> gpu_unique_id_space_offset;
  unique_id_space_offset.to(&gpu_unique_id_space_offset);

  num_unique_key_per_table->resize(unique_id_space.get_num_elements());
  for (int i = 0; i < unique_id_space.get_num_elements(); ++i) {
    // std::cout << "gpu_unique_id_space_offset " << gpu_unique_id_space_offset[i] << "\n";
    (*num_unique_key_per_table)[i] =
        gpu_unique_id_space_offset[i + 1] - gpu_unique_id_space_offset[i];
  }
}

void All2AllEmbeddingCollectionModelBackward::copy_backward_key_and_emb_vec(
    std::vector<Tensor> &unique_key, std::vector<Tensor> &emb_vec) {
  size_t nbytes_key_offsets = 0ul;
  size_t nbytes_emb_vec_offsets = 0ul;
  for (size_t i = 0; i < unique_key.size(); ++i) {
    HCTR_LIB_THROW(cudaMemcpyAsync(
        unique_key[i].get(),
        reinterpret_cast<char *>(continous_unique_key_.get()) + nbytes_key_offsets,
        unique_key[i].nbytes(), cudaMemcpyDeviceToDevice, core_->get_local_gpu()->get_stream()));
    HCTR_LIB_THROW(cudaMemcpyAsync(
        emb_vec[i].get(),
        reinterpret_cast<char *>(continous_emb_vec_.get()) + nbytes_emb_vec_offsets,
        emb_vec[i].nbytes(), cudaMemcpyDeviceToDevice, core_->get_local_gpu()->get_stream()));
    nbytes_key_offsets += unique_key[i].nbytes();
    nbytes_emb_vec_offsets += emb_vec[i].nbytes();
  }
}

}  // namespace tf
}  // namespace embedding
