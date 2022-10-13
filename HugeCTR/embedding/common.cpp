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
#include "common.hpp"

#include "HugeCTR/include/utils.hpp"
namespace embedding {

std::ostream &operator<<(std::ostream &os, const Combiner &p) {
  switch (p) {
    case Combiner::Sum:
      os << "sum";
      break;
    case Combiner::Average:
      os << "average";
      break;
    case Combiner::Concat:
      os << "concat";
      break;
    default:
      break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const LookupParam &p) {
  os << "lookup_id:" << p.lookup_id << ",";
  os << "table_id:" << p.table_id << ",";
  os << "combiner:" << p.combiner << ",";
  os << "max_hotness:" << p.max_hotness << ",";
  os << "ev_size:" << p.ev_size;
  return os;
}

UniformModelParallelEmbeddingMeta::UniformModelParallelEmbeddingMeta(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
    size_t grouped_id)
    : num_lookup_(ebc_param.num_lookup), h_ev_size_offset_{0}, h_local_ev_size_offset_{0} {
  HugeCTR::CudaDeviceContext context(core->get_device_id());
  const auto &lookup_params = ebc_param.lookup_params;
  auto buffer_ptr = GetBuffer(core);
  const auto &group_params = ebc_param.grouped_emb_params[grouped_id];
  HCTR_CHECK_HINT(group_params.table_placement_strategy == TablePlacementStrategy::ModelParallel,
                  "UniformModelParallelEmbeddingMeta must be initialized by ModelParallel");

  size_t num_gpus = core->get_global_gpu_count();
  int gpu_id = core->get_global_gpu_id();

  HCTR_CHECK_HINT(ebc_param.shard_matrix.size() == num_gpus,
                  "shard matrix should contain num_gpus row.");

  for (int lookup_id = 0; lookup_id < num_lookup_; ++lookup_id) {
    int table_id = lookup_params[lookup_id].table_id;
    int ev_size = lookup_params[lookup_id].ev_size;
    char combiner = static_cast<char>(lookup_params[lookup_id].combiner);

    h_ev_size_list_.push_back(ev_size);
    h_combiner_list_.push_back(combiner);
    if (std::find(group_params.table_ids.begin(), group_params.table_ids.end(), table_id) ==
        group_params.table_ids.end()) {
      continue;
    }

    if (ebc_param.shard_matrix[gpu_id][table_id] == 0) {
      continue;
    }

    std::vector<int> shard_gpus;
    for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
      if (ebc_param.shard_matrix[ggpu_id][table_id] == 1) {
        shard_gpus.push_back(ggpu_id);
      }
    }
    auto find_shard_id_iter = std::find(shard_gpus.begin(), shard_gpus.end(), gpu_id);
    HCTR_CHECK_HINT(find_shard_id_iter != shard_gpus.end(),
                    "ModelParallelEmbeddingMeta does not find shard id");
    int shard_id = std::distance(shard_gpus.begin(), find_shard_id_iter);

    h_local_shard_id_list_.push_back(shard_id);
    h_local_num_shards_list_.push_back(static_cast<int>(shard_gpus.size()));
    h_local_table_id_list_.push_back(table_id);
    h_local_lookup_id_list_.push_back(lookup_id);
    h_local_ev_size_list_.push_back(ev_size);
  }
  std::partial_sum(h_ev_size_list_.begin(), h_ev_size_list_.end(),
                   std::back_inserter(h_ev_size_offset_));
  d_ev_size_offset_ =
      buffer_ptr->reserve({h_ev_size_offset_.size()}, DeviceType::GPU, TensorScalarType::Int32);
  buffer_ptr->allocate();
  d_ev_size_offset_.copy_from(h_ev_size_offset_);
  max_ev_size_ = h_ev_size_list_.size() > 0
                     ? *std::max_element(h_ev_size_list_.begin(), h_ev_size_list_.end())
                     : 0;

  // cudaDeviceProp device_prop;
  // cudaGetDeviceProperties(&device_prop, 0);
  // num_sms_ = device_prop.multiProcessorCount;
  // FIX: cudaGetDeviceProperties get ,cost too much time, need remove it to the start of program ,
  // not use per iteration,for now fix the num_sms_
  num_sms_ = 108;

  d_combiner_list_ =
      buffer_ptr->reserve({h_combiner_list_.size()}, DeviceType::GPU, TensorScalarType::Char);
  buffer_ptr->allocate();
  d_combiner_list_.copy_from(h_combiner_list_);

  num_local_lookup_ = static_cast<int>(h_local_table_id_list_.size());

  d_local_shard_id_list_ = buffer_ptr->reserve({h_local_shard_id_list_.size()}, DeviceType::GPU,
                                               TensorScalarType::Int32);
  buffer_ptr->allocate();
  d_local_shard_id_list_.copy_from(h_local_shard_id_list_);

  d_local_num_shards_list_ = buffer_ptr->reserve({h_local_num_shards_list_.size()}, DeviceType::GPU,
                                                 TensorScalarType::Int32);
  buffer_ptr->allocate();
  d_local_num_shards_list_.copy_from(h_local_num_shards_list_);

  d_local_table_id_list_ = buffer_ptr->reserve({h_local_table_id_list_.size()}, DeviceType::GPU,
                                               TensorScalarType::Int32);
  buffer_ptr->allocate();
  d_local_table_id_list_.copy_from(h_local_table_id_list_);

  d_local_lookup_id_list_ = buffer_ptr->reserve({h_local_lookup_id_list_.size()}, DeviceType::GPU,
                                                TensorScalarType::Int32);
  buffer_ptr->allocate();
  d_local_lookup_id_list_.copy_from(h_local_lookup_id_list_);

  d_local_ev_size_list_ =
      buffer_ptr->reserve({h_local_ev_size_list_.size()}, DeviceType::GPU, TensorScalarType::Int32);
  buffer_ptr->allocate();
  d_local_ev_size_list_.copy_from(h_local_ev_size_list_);

  std::partial_sum(h_local_ev_size_list_.begin(), h_local_ev_size_list_.end(),
                   std::back_inserter(h_local_ev_size_offset_));
  d_local_ev_size_offset_ = buffer_ptr->reserve({h_local_ev_size_offset_.size()}, DeviceType::GPU,
                                                TensorScalarType::Int32);
  buffer_ptr->allocate();
  d_local_ev_size_offset_.copy_from(h_local_ev_size_offset_);

  h_network_lookup_id_list_.clear();

  h_global_lookup_id_list_.resize(num_gpus);
  for (size_t ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
    for (int lookup_id = 0; lookup_id < num_lookup_; ++lookup_id) {
      int table_id = lookup_params[lookup_id].table_id;
      if (std::find(group_params.table_ids.begin(), group_params.table_ids.end(), table_id) ==
          group_params.table_ids.end()) {
        continue;
      }

      if (ebc_param.shard_matrix[ggpu_id][table_id] == 0) {
        continue;
      }
      h_global_lookup_id_list_[ggpu_id].push_back(lookup_id);
    }
  }
  for (auto &vec : h_global_lookup_id_list_) {
    h_network_lookup_id_list_.insert(h_network_lookup_id_list_.end(), vec.begin(), vec.end());
  }
  std::sort(h_network_lookup_id_list_.begin(), h_network_lookup_id_list_.end());
  auto last = std::unique(h_network_lookup_id_list_.begin(), h_network_lookup_id_list_.end());
  h_network_lookup_id_list_.erase(last, h_network_lookup_id_list_.end());
  d_network_lookup_id_list_ = buffer_ptr->reserve({h_network_lookup_id_list_.size()},
                                                  DeviceType::GPU, TensorScalarType::Int32);
  buffer_ptr->allocate();
  d_network_lookup_id_list_.copy_from(h_network_lookup_id_list_);

  for (int lookup_id : h_network_lookup_id_list_) {
    h_network_combiner_list_.push_back(static_cast<char>(lookup_params[lookup_id].combiner));
  }

  h_network_ev_sizes_.resize(num_gpus);
  h_network_ev_offsets_.resize(num_gpus);
  for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
    h_network_ev_offsets_[ggpu_id].push_back(0);
  }

  std::vector<std::tuple<int, int, int>> h_network_buffer_meta_info;
  for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
    int network_id = 0;
    for (int lookup_id : h_global_lookup_id_list_[ggpu_id]) {
      int ev_size = h_ev_size_list_[lookup_id];
      h_network_ev_sizes_[ggpu_id].push_back(ev_size);
      h_network_ev_offsets_[ggpu_id].push_back(ev_size);
      h_network_buffer_meta_info.push_back({ggpu_id, network_id, lookup_id});
      network_id += 1;
    }
  }
  for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
    std::partial_sum(h_network_ev_offsets_[ggpu_id].begin(), h_network_ev_offsets_[ggpu_id].end(),
                     h_network_ev_offsets_[ggpu_id].begin());
  }

  for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
    network_ev_size_list_.push_back(buffer_ptr->reserve({h_network_ev_sizes_[ggpu_id].size()},
                                                        DeviceType::GPU, TensorScalarType::Int32));
    network_ev_offset_list_.push_back(buffer_ptr->reserve(
        {h_network_ev_offsets_[ggpu_id].size()}, DeviceType::GPU, TensorScalarType::Int32));
  }
  buffer_ptr->allocate();
  for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
    network_ev_size_list_[ggpu_id].copy_from(h_network_ev_sizes_[ggpu_id]);
    network_ev_offset_list_[ggpu_id].copy_from(h_network_ev_offsets_[ggpu_id]);
  }
  network_ev_sizes_ =
      TensorList(core.get(), network_ev_size_list_, DeviceType::GPU, TensorScalarType::Int32);
  network_ev_offsets_ =
      TensorList(core.get(), network_ev_offset_list_, DeviceType::GPU, TensorScalarType::Int32);

  std::sort(h_network_buffer_meta_info.begin(), h_network_buffer_meta_info.end(),
            [](const auto &lhs, const auto &rhs) { return std::get<2>(lhs) <= std::get<2>(rhs); });

  for (size_t i = 0; i < h_network_buffer_meta_info.size(); ++i) {
    const auto &meta_info = h_network_buffer_meta_info[i];
    int network_gpu_id = std::get<0>(meta_info);
    int network_id = std::get<1>(meta_info);
    h_network_ids_.push_back(network_id);
    h_network_gpu_ids_.push_back(network_gpu_id);
  }
  network_ids_ =
      buffer_ptr->reserve({h_network_ids_.size()}, DeviceType::GPU, TensorScalarType::Int32);
  buffer_ptr->allocate();
  network_ids_.copy_from(h_network_ids_);
  network_gpu_ids_ =
      buffer_ptr->reserve({h_network_gpu_ids_.size()}, DeviceType::GPU, TensorScalarType::Int32);
  buffer_ptr->allocate();
  network_gpu_ids_.copy_from(h_network_gpu_ids_);

  int network_offset = 0;
  for (size_t i = 0; i < h_network_buffer_meta_info.size(); ++i) {
    const auto &meta_info = h_network_buffer_meta_info[i];
    int lookup_id = std::get<2>(meta_info);
    if (i == 0 || lookup_id != std::get<2>(h_network_buffer_meta_info[i - 1])) {
      h_network_offsets_.push_back(network_offset);
    }
    network_offset += 1;
  }
  h_network_offsets_.push_back(network_offset);
  network_offsets_ =
      buffer_ptr->reserve({h_network_offsets_.size()}, DeviceType::GPU, TensorScalarType::Int32);
  buffer_ptr->allocate();
  network_offsets_.copy_from(h_network_offsets_);

  for (size_t i = 0; i < h_network_buffer_meta_info.size(); ++i) {
    const auto &meta_info = h_network_buffer_meta_info[i];
    int lookup_id = std::get<2>(meta_info);
    if (i == 0 || lookup_id != std::get<2>(h_network_buffer_meta_info[i - 1])) {
      h_network_dst_lookup_ids_.push_back(lookup_id);
    }
  }
  network_dst_lookup_ids_ = buffer_ptr->reserve({h_network_dst_lookup_ids_.size()}, DeviceType::GPU,
                                                TensorScalarType::Int32);
  buffer_ptr->allocate();
  network_dst_lookup_ids_.copy_from(h_network_dst_lookup_ids_);

  update_mutable_meta(core, ebc_param, grouped_id);
}

void UniformModelParallelEmbeddingMeta::update_mutable_meta(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
    size_t grouped_id) const {
  h_hotness_list_.clear();
  h_local_hotness_list_.clear();

  HugeCTR::CudaDeviceContext context(core->get_device_id());
  const auto &lookup_params = ebc_param.lookup_params;
  const auto &group_params = ebc_param.grouped_emb_params[grouped_id];
  HCTR_CHECK_HINT(group_params.table_placement_strategy == TablePlacementStrategy::ModelParallel,
                  "UniformModelParallelEmbeddingMeta must be initialized by ModelParallel");

  size_t num_gpus = core->get_global_gpu_count();
  int gpu_id = core->get_global_gpu_id();

  HCTR_CHECK_HINT(ebc_param.shard_matrix.size() == num_gpus,
                  "shard matrix should contain num_gpus row.");

  for (int lookup_id = 0; lookup_id < num_lookup_; ++lookup_id) {
    int table_id = lookup_params[lookup_id].table_id;
    int max_hotness = lookup_params[lookup_id].max_hotness;

    h_hotness_list_.push_back(max_hotness);
    if (std::find(group_params.table_ids.begin(), group_params.table_ids.end(), table_id) ==
        group_params.table_ids.end()) {
      continue;
    }

    if (ebc_param.shard_matrix[gpu_id][table_id] == 0) {
      continue;
    }

    h_local_hotness_list_.push_back(max_hotness);
  }
  num_local_hotness_ =
      std::accumulate(h_local_hotness_list_.begin(), h_local_hotness_list_.end(), 0);
  hotness_sum_ = std::accumulate(h_hotness_list_.begin(), h_hotness_list_.end(), 0);
}

UniformDataParallelEmbeddingMeta::UniformDataParallelEmbeddingMeta(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
    size_t grouped_id)
    : num_lookup_(ebc_param.num_lookup), h_ev_size_offset_{0} {
  HugeCTR::CudaDeviceContext context(core->get_device_id());
  const auto &lookup_params = ebc_param.lookup_params;
  auto buffer_ptr = GetBuffer(core);
  const auto &group_params = ebc_param.grouped_emb_params[grouped_id];
  HCTR_CHECK_HINT(group_params.table_placement_strategy == TablePlacementStrategy::DataParallel,
                  "UniformDataParallelEmbeddingMeta must be initialized by DataParallel");

  size_t num_gpus = core->get_global_gpu_count();
  int gpu_id = core->get_global_gpu_id();

  HCTR_CHECK_HINT(ebc_param.shard_matrix.size() == num_gpus,
                  "shard matrix should contain num_gpus row.");

  for (int lookup_id = 0; lookup_id < num_lookup_; ++lookup_id) {
    int table_id = lookup_params[lookup_id].table_id;
    int ev_size = lookup_params[lookup_id].ev_size;
    char combiner = static_cast<char>(lookup_params[lookup_id].combiner);

    h_ev_size_list_.push_back(ev_size);
    h_combiner_list_.push_back(combiner);
    if (std::find(group_params.table_ids.begin(), group_params.table_ids.end(), table_id) ==
        group_params.table_ids.end()) {
      continue;
    }
    HCTR_CHECK_HINT(ebc_param.shard_matrix[gpu_id][table_id] == 1,
                    "dp table must be shared on all gpus");
    h_local_combiner_list_.push_back(combiner);
    h_local_lookup_id_list_.push_back(lookup_id);
    h_local_ev_size_list_.push_back(ev_size);
    h_local_table_id_list_.push_back(table_id);
  }

  max_ev_size_ = h_ev_size_list_.size() > 0
                     ? *std::max_element(h_ev_size_list_.begin(), h_ev_size_list_.end())
                     : 0;
  std::partial_sum(h_ev_size_list_.begin(), h_ev_size_list_.end(),
                   std::back_inserter(h_ev_size_offset_));
  d_ev_size_offset_ =
      buffer_ptr->reserve({h_ev_size_offset_.size()}, DeviceType::GPU, TensorScalarType::Int32);
  buffer_ptr->allocate();
  d_ev_size_offset_.copy_from(h_ev_size_offset_);

  d_combiner_list_ =
      buffer_ptr->reserve({h_combiner_list_.size()}, DeviceType::GPU, TensorScalarType::Char);
  buffer_ptr->allocate();
  d_combiner_list_.copy_from(h_combiner_list_);

  num_local_lookup_ = static_cast<int>(h_local_table_id_list_.size());

  d_local_lookup_id_list_ = buffer_ptr->reserve({h_local_lookup_id_list_.size()}, DeviceType::GPU,
                                                TensorScalarType::Int32);
  buffer_ptr->allocate();
  d_local_lookup_id_list_.copy_from(h_local_lookup_id_list_);

  d_local_ev_size_list_ =
      buffer_ptr->reserve({h_local_ev_size_list_.size()}, DeviceType::GPU, TensorScalarType::Int32);
  buffer_ptr->allocate();
  d_local_ev_size_list_.copy_from(h_local_ev_size_list_);

  d_local_table_id_list_ = buffer_ptr->reserve({h_local_table_id_list_.size()}, DeviceType::GPU,
                                               TensorScalarType::Int32);
  buffer_ptr->allocate();
  d_local_table_id_list_.copy_from(h_local_table_id_list_);

  update_mutable_meta(core, ebc_param, grouped_id);
}

void UniformDataParallelEmbeddingMeta::update_mutable_meta(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param,
    size_t grouped_id) const {
  h_hotness_list_.clear();
  h_local_hotness_list_.clear();

  HugeCTR::CudaDeviceContext context(core->get_device_id());
  const auto &lookup_params = ebc_param.lookup_params;
  const auto &group_params = ebc_param.grouped_emb_params[grouped_id];
  HCTR_CHECK_HINT(group_params.table_placement_strategy == TablePlacementStrategy::DataParallel,
                  "UniformDataParallelEmbeddingMeta must be initialized by DataParallel");

  size_t num_gpus = core->get_global_gpu_count();
  int gpu_id = core->get_global_gpu_id();

  HCTR_CHECK_HINT(ebc_param.shard_matrix.size() == num_gpus,
                  "shard matrix should contain num_gpus row.");

  for (int lookup_id = 0; lookup_id < num_lookup_; ++lookup_id) {
    int table_id = lookup_params[lookup_id].table_id;
    int max_hotness = lookup_params[lookup_id].max_hotness;

    h_hotness_list_.push_back(max_hotness);
    if (std::find(group_params.table_ids.begin(), group_params.table_ids.end(), table_id) ==
        group_params.table_ids.end()) {
      continue;
    }
    HCTR_CHECK_HINT(ebc_param.shard_matrix[gpu_id][table_id] == 1,
                    "dp table must be shared on all gpus");
    h_local_hotness_list_.push_back(max_hotness);
  }
  num_hotness_ = std::accumulate(h_hotness_list_.begin(), h_hotness_list_.end(), 0);

  num_local_hotness_ =
      std::accumulate(h_local_hotness_list_.begin(), h_local_hotness_list_.end(), 0);
}

}  // namespace embedding
