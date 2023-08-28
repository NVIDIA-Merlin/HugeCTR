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

#include <cuda_runtime.h>

#include <core/hctr_impl/hctr_backend.hpp>
#include <embedding/common.hpp>
#include <utils.hpp>

namespace HugeCTR {
DataDistributionInput::DataDistributionInput(std::shared_ptr<core::CoreResourceManager> core,
                                             int num_lookup, core23::DataType key_type,
                                             core23::DataType offset_type)
    : num_lookup_(num_lookup), key_type(key_type), offset_type(offset_type) {
  CudaDeviceContext ctx(core->get_device_id());

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  // 2x for both keys & bucket_range
  this->d_ptrs_ = core23::Tensor(
      params.shape({static_cast<int64_t>(num_lookup_ * 2)}).data_type(core23::ScalarType::Pointer));
  this->h_ptrs_ = core23::Tensor(core23::TensorParams()
                                     .device(core23::DeviceType::CPU)
                                     .shape({static_cast<int64_t>(num_lookup_ * 2)})
                                     .data_type(core23::ScalarType::Pointer));
}

void DataDistributionInput::copy_tensor_vec(const std::vector<core23::Tensor> &dp_keys,
                                            const std::vector<core23::Tensor> &dp_bucket_range,
                                            cudaStream_t stream) {
  int num_lookup = dp_keys.size();
  HCTR_CHECK(num_lookup == num_lookup_);

  // concat both arrays so we only need to copy up one array of pointers
  for (size_t i = 0; i < num_lookup; ++i) {
    h_ptrs_.data<void *>()[i] = dp_keys[i].data();
    h_ptrs_.data<void *>()[num_lookup + i] = dp_bucket_range[i].data();
  }

  core23::copy_async(d_ptrs_, h_ptrs_, stream);
}
}  // namespace HugeCTR

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

std::ostream &operator<<(std::ostream &os, const EmbeddingLayout &p) {
  switch (p) {
    case EmbeddingLayout::FeatureMajor:
      os << "FeatureMajor";
      break;
    case EmbeddingLayout::BatchMajor:
      os << "BatchMajor";
      break;
    default:
      HCTR_OWN_THROW(HugeCTR::Error_t::NotInitialized, "EmbeddingLayout is not initialized");
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const CommunicationStrategy &p) {
  switch (p) {
    case CommunicationStrategy::Uniform:
      os << "Uniform";
      break;
    case CommunicationStrategy::Hierarchical:
      os << "Hierarchical";
      break;
    default:
      HCTR_OWN_THROW(HugeCTR::Error_t::NotInitialized, "CommunicationStrategy is not initialized");
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const SortStrategy &p) {
  switch (p) {
    case SortStrategy::Radix:
      os << "Radix";
      break;
    case SortStrategy::Segmented:
      os << "Segmented";
      break;
    default:
      HCTR_OWN_THROW(HugeCTR::Error_t::NotInitialized, "SortStrategy is not initialized");
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const KeysPreprocessStrategy &p) {
  switch (p) {
    case KeysPreprocessStrategy::None:
      os << "None";
      break;
    case KeysPreprocessStrategy::AddOffset:
      os << "AddOffset";
      break;
    default:
      HCTR_OWN_THROW(HugeCTR::Error_t::NotInitialized, "KeysPreprocessStrategy is not initialized");
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const AllreduceStrategy &p) {
  switch (p) {
    case AllreduceStrategy::Dense:
      os << "Dense";
      break;
    case AllreduceStrategy::GroupDense:
      os << "GroupDense";
      break;
    case AllreduceStrategy::Sparse:
      os << "Sparse";
      break;
    default:
      HCTR_OWN_THROW(HugeCTR::Error_t::NotInitialized, "AllreduceStrategy is not initialized");
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

void EmbeddingOutputAttr::init(std::shared_ptr<CoreResourceManager> core,
                               const EmbeddingCollectionParam &ebc_param) {
  this->num_lookup = ebc_param.num_lookup;
  const auto &lookup_params = ebc_param.lookup_params;
  h_id_to_ev_size.clear();
  h_id_to_combiner.clear();
  h_id_to_ev_start_indices = {0};

  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    int ev_size = lookup_params[lookup_id].ev_size;
    h_id_to_ev_size.push_back(ev_size);
    char combiner = static_cast<char>(lookup_params[lookup_id].combiner);
    h_id_to_combiner.push_back(combiner);

    h_id_to_ev_start_indices.push_back((combiner == static_cast<char>(Combiner::Concat))
                                           ? ebc_param.lookup_params[lookup_id].max_hotness *
                                                 ev_size
                                           : ev_size);
  }

  std::partial_sum(h_id_to_ev_start_indices.begin(), h_id_to_ev_start_indices.end(),
                   h_id_to_ev_start_indices.begin());

  HugeCTR::CudaDeviceContext context(core->get_device_id());
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::BufferParams buffer_params;
  buffer_params.unitary = false;
  core23::TensorParams params = core23::TensorParams().device(device).buffer_params(buffer_params);

  this->id_to_ev_size = core23::Tensor(params.shape({static_cast<int64_t>(h_id_to_ev_size.size())})
                                           .data_type(core23::ScalarType::Int32));
  core23::copy_sync(this->id_to_ev_size, h_id_to_ev_size);

  this->id_to_ev_start_indices =
      core23::Tensor(params.shape({static_cast<int64_t>(h_id_to_ev_start_indices.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(this->id_to_ev_start_indices, h_id_to_ev_start_indices);

  this->id_to_combiner =
      core23::Tensor(params.shape({static_cast<int64_t>(h_id_to_combiner.size())})
                         .data_type(core23::ScalarType::Char));
  core23::copy_sync(this->id_to_combiner, h_id_to_combiner);

  this->num_elements_per_sample = h_id_to_ev_start_indices.back();

  this->layout = ebc_param.output_layout_;
  this->max_ev_size = !h_id_to_ev_size.empty()
                          ? *std::max_element(h_id_to_ev_size.begin(), h_id_to_ev_size.end())
                          : 0;
  this->is_ragged =
      (h_id_to_ev_size.size() == 0)
          ? false
          : std::equal(h_id_to_ev_size.begin() + 1, h_id_to_ev_size.end(), h_id_to_ev_size.begin());

  bool is_aligned = true;
  for (auto ev_size : h_id_to_ev_size) {
    if (ev_size % 4 != 0) is_aligned = false;
  }
  this->is_aligned = is_aligned;

  this->type = ebc_param.emb_type;
}

void EmbeddingOutputAttr::update_mutable_data(std::shared_ptr<CoreResourceManager> core,
                                              const EmbeddingCollectionParam &ebc_param) const {
  h_id_to_hotness.clear();

  HugeCTR::CudaDeviceContext context(core->get_device_id());
  const auto &lookup_params = ebc_param.lookup_params;

  size_t num_gpus = core->get_global_gpu_count();

  HCTR_CHECK_HINT(ebc_param.shard_matrix.size() == num_gpus,
                  "shard matrix should contain num_gpus row.");

  for (int lookup_id = 0; lookup_id < this->num_lookup; ++lookup_id) {
    int max_hotness = lookup_params[lookup_id].max_hotness;

    h_id_to_hotness.push_back(max_hotness);
  }
  hotness_sum = std::accumulate(h_id_to_hotness.begin(), h_id_to_hotness.end(), 0);
}

std::vector<int> get_lookup_id_table_id(const EmbeddingCollectionParam &ebc_param,
                                        size_t grouped_id, int gpu_id) {
  const auto &lookup_params = ebc_param.lookup_params;

  std::vector<int> lookup_id_to_table_id;
  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    if (!ebc_param.has_table_shard(gpu_id, grouped_id, lookup_id)) continue;

    int table_id = lookup_params[lookup_id].table_id;
    lookup_id_to_table_id.push_back(table_id);
  }
  return lookup_id_to_table_id;
}

std::vector<int> sort_lookup_ids(const std::vector<int> &lookup_id_to_table_id) {
  int num_lookup = lookup_id_to_table_id.size();

  std::vector<int> sorted_lookup_ids(num_lookup);
  std::iota(sorted_lookup_ids.begin(), sorted_lookup_ids.end(), 0);
  std::sort(sorted_lookup_ids.begin(), sorted_lookup_ids.end(),
            [&](int l, int r) { return lookup_id_to_table_id[l] < lookup_id_to_table_id[r]; });

  return sorted_lookup_ids;
}

std::vector<int> get_sorted_table_ids(const std::vector<int> &sorted_lookup_ids,
                                      const std::vector<int> &lookup_id_to_table_id) {
  std::vector<int> sorted_table_ids;
  std::transform(sorted_lookup_ids.begin(), sorted_lookup_ids.end(),
                 std::back_inserter(sorted_table_ids),
                 [&](int idx) { return lookup_id_to_table_id[idx]; });

  return sorted_table_ids;
}

std::vector<int> deduplicate_sorted_table_ids(const std::vector<int> &sorted_table_ids) {
  std::vector<int> unique_table_ids{sorted_table_ids.begin(), sorted_table_ids.end()};
  auto last = std::unique(unique_table_ids.begin(), unique_table_ids.end());
  unique_table_ids.erase(last, unique_table_ids.end());
  return unique_table_ids;
}

std::vector<int> get_table_id_to_ev_size(std::shared_ptr<CoreResourceManager> core,
                                         const EmbeddingCollectionParam &ebc_param) {
  const auto &lookup_params = ebc_param.lookup_params;

  std::vector<int> h_table_id_to_ev_size;
  h_table_id_to_ev_size.resize(ebc_param.num_table);
  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    int table_id = lookup_params[lookup_id].table_id;
    h_table_id_to_ev_size[table_id] = lookup_params[lookup_id].ev_size;
  }
  return h_table_id_to_ev_size;
}

void WgradAttr::init(std::shared_ptr<CoreResourceManager> core,
                     const EmbeddingCollectionParam &ebc_param, size_t grouped_id) {
  int gpu_id = core->get_global_gpu_id();
  auto h_lookup_id_to_table_id = get_lookup_id_table_id(ebc_param, grouped_id, gpu_id);
  auto h_sorted_lookup_ids = sort_lookup_ids(h_lookup_id_to_table_id);
  auto h_sorted_table_ids = get_sorted_table_ids(h_sorted_lookup_ids, h_lookup_id_to_table_id);
  this->h_sorted_unique_table_ids = deduplicate_sorted_table_ids(h_sorted_table_ids);
  auto h_table_id_to_ev_size = get_table_id_to_ev_size(core, ebc_param);

  this->num_table = static_cast<int>(this->h_sorted_unique_table_ids.size());
  this->num_lookup = h_lookup_id_to_table_id.size();
  this->type = ebc_param.wgrad_type_;

  HugeCTR::CudaDeviceContext context(core->get_device_id());
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::BufferParams buffer_params;
  buffer_params.unitary = false;
  core23::TensorParams params = core23::TensorParams().device(device).buffer_params(buffer_params);

  const auto &lookup_params = ebc_param.lookup_params;
  std::vector<int> h_ev_size_list;
  h_ev_size_list.clear();

  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    h_ev_size_list.push_back(lookup_params[lookup_id].ev_size);
  }

  // FIX:global ev size same or local ev size same? for now is global ev size same
  // I think is include dp and mp.
  if (h_ev_size_list.size() > 0) {
    same_ev_size = h_ev_size_list[0];
    is_same_ev_size = true;
    for (size_t i = 1; i < h_ev_size_list.size(); ++i) {
      if (h_ev_size_list[i] != same_ev_size) {
        is_same_ev_size = false;
        same_ev_size = 0;
        break;
      }
    }
  }

  this->lookup_id_to_table_ids =
      core23::Tensor(params.shape({static_cast<int64_t>(h_lookup_id_to_table_id.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(this->lookup_id_to_table_ids, h_lookup_id_to_table_id);

  this->sorted_lookup_ids =
      core23::Tensor(params.shape({static_cast<int64_t>(h_sorted_lookup_ids.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(this->sorted_lookup_ids, h_sorted_lookup_ids);

  this->sorted_table_ids =
      core23::Tensor(params.shape({static_cast<int64_t>(h_sorted_table_ids.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(this->sorted_table_ids, h_sorted_table_ids);

  this->sorted_unique_table_ids =
      core23::Tensor(params.shape({static_cast<int64_t>(h_sorted_unique_table_ids.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(this->sorted_unique_table_ids, h_sorted_unique_table_ids);

  this->table_id_to_ev_size =
      core23::Tensor(params.shape({static_cast<int64_t>(h_table_id_to_ev_size.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(this->table_id_to_ev_size, h_table_id_to_ev_size);
}

std::vector<int> get_wgrad_max_num_keys(const EmbeddingCollectionParam &ebc_param,
                                        size_t grouped_id, int gpu_id) {
  const auto &lookup_params = ebc_param.lookup_params;

  std::vector<int> local_max_hotness_list;
  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    if (!ebc_param.has_table_shard(gpu_id, grouped_id, lookup_id)) continue;

    int max_hotness = lookup_params[lookup_id].max_hotness;
    local_max_hotness_list.push_back(max_hotness);
  }
  return local_max_hotness_list;
}

std::vector<int> get_wgrad_ev_size(const EmbeddingCollectionParam &ebc_param, size_t grouped_id,
                                   int gpu_id) {
  const auto &lookup_params = ebc_param.lookup_params;
  std::vector<int> local_ev_size_list;
  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    if (!ebc_param.has_table_shard(gpu_id, grouped_id, lookup_id)) continue;

    int ev_size = lookup_params[lookup_id].ev_size;
    local_ev_size_list.push_back(ev_size);
  }
  return local_ev_size_list;
}

std::vector<int> get_allreduce_buffer_num_keys(
    const std::vector<int> &unique_table_ids, const std::vector<int> &table_id_to_vocabulary_size) {
  std::vector<int> vocabulary_size_list;
  for (auto table_id : unique_table_ids) {
    vocabulary_size_list.push_back(table_id_to_vocabulary_size[table_id]);
  }
  return vocabulary_size_list;
}

void Wgrad::bind_data_ptr(void *ptr) {
  data = core23::Tensor::bind(ptr, {static_cast<int64_t>(this->max_buffer_size)}, this->attr.type,
                              this->unique_keys.device());
}

WgradInitializer &WgradInitializer::init(Wgrad &other) {
  this->wgrad = &other;
  wgrad->attr = wgrad_attr;
  return *this;
}

WgradInitializer &WgradInitializer::init_indices() {
  int gpu_id = core->get_global_gpu_id();

  auto local_max_hotness_list = get_wgrad_max_num_keys(ebc_param, grouped_id, gpu_id);
  auto local_ev_size_list = get_wgrad_ev_size(ebc_param, grouped_id, gpu_id);

  int max_num_keys =
      std::accumulate(local_max_hotness_list.begin(), local_max_hotness_list.end(), 0);

  auto key_type = ebc_param.key_type;
  int batch_size = ebc_param.universal_batch_size;

  HugeCTR::CudaDeviceContext context(core->get_device_id());

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);
  wgrad->attr.type = ebc_param.wgrad_type_;
  wgrad->unique_keys =
      core23::Tensor(params.shape({batch_size * max_num_keys}).data_type(key_type));
  wgrad->num_unique_keys = core23::Tensor(params.shape({1}).data_type(core23::ScalarType::UInt64));
  wgrad->table_ids = core23::Tensor(
      params.shape({batch_size * max_num_keys}).data_type(core23::ScalarType::Int32));
  wgrad->ev_start_indices = core23::Tensor(
      params.shape({batch_size * max_num_keys + 1}).data_type(core23::ScalarType::UInt32));
  return *this;
}

WgradInitializer &WgradInitializer::init_data() {
  int gpu_id = core->get_global_gpu_id();

  auto local_max_hotness_list = get_wgrad_max_num_keys(ebc_param, grouped_id, gpu_id);
  auto local_ev_size_list = get_wgrad_ev_size(ebc_param, grouped_id, gpu_id);

  int batch_size = ebc_param.universal_batch_size;

  HugeCTR::CudaDeviceContext context(core->get_device_id());

  int64_t max_buffer_size = 0;
  for (size_t i = 0; i < local_max_hotness_list.size(); ++i) {
    max_buffer_size += local_max_hotness_list[i] * local_ev_size_list[i];
  }
  max_buffer_size *= batch_size;
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);
  wgrad->data = core23::Tensor(params.shape({max_buffer_size}).data_type(wgrad->attr.type));
  return *this;
}

AllreduceWgradInitializer &AllreduceWgradInitializer::init(Wgrad &other) {
  this->wgrad = &other;
  wgrad->attr = wgrad_attr;
  return *this;
}

AllreduceWgradInitializer &AllreduceWgradInitializer::init_indices() {
  int gpu_id = core->get_global_gpu_id();

  std::vector<int> h_local_ev_size_list = get_wgrad_ev_size(ebc_param, grouped_id, gpu_id);
  std::vector<int> h_unique_table_ids(wgrad_attr.sorted_unique_table_ids.num_elements());
  core23::copy_sync(h_unique_table_ids, wgrad_attr.sorted_unique_table_ids);

  std::vector<int> h_local_num_keys_list =
      get_allreduce_buffer_num_keys(h_unique_table_ids, table_id_to_vocabulary_size);
  auto key_type = ebc_param.key_type;

  int max_num_keys = std::accumulate(h_local_num_keys_list.begin(), h_local_num_keys_list.end(), 0);

  HugeCTR::CudaDeviceContext context(core->get_device_id());

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  wgrad->unique_keys = core23::Tensor(params.shape({max_num_keys}).data_type(key_type));
  wgrad->num_unique_keys = core23::Tensor(params.shape({1}).data_type(core23::ScalarType::UInt64));
  wgrad->table_ids =
      core23::Tensor(params.shape({max_num_keys}).data_type(core23::ScalarType::Int32));
  wgrad->ev_start_indices =
      core23::Tensor(params.shape({max_num_keys + 1}).data_type(core23::ScalarType::UInt32));

  // TODO: move those initialization on GPU
  DISPATCH_INTEGRAL_FUNCTION_CORE23(key_type.type(), key_t, [&] {
    std::vector<key_t> h_unique_keys;
    size_t num_keys = 0;
    for (size_t i = 0; i < h_local_num_keys_list.size(); ++i) {
      num_keys += h_local_num_keys_list[i];
    }
    h_unique_keys.resize(num_keys);
    std::iota(h_unique_keys.begin(), h_unique_keys.end(), 0);
    core23::copy_sync(wgrad->unique_keys, h_unique_keys);
    std::vector<size_t> h_num_unqiue_keys{h_unique_keys.size()};
    core23::copy_sync(wgrad->num_unique_keys.data(), h_num_unqiue_keys.data(),
                      wgrad->num_unique_keys.num_bytes(), wgrad->num_unique_keys.device(),
                      core23::DeviceType::CPU);
  });

  std::vector<int> h_table_ids;
  for (size_t i = 0; i < h_local_num_keys_list.size(); ++i) {
    h_table_ids.insert(h_table_ids.end(), h_local_num_keys_list[i], h_unique_table_ids[i]);
  }
  core23::copy_sync(wgrad->table_ids.data(), h_table_ids.data(), wgrad->table_ids.num_bytes(),
                    wgrad->table_ids.device(), core23::DeviceType::CPU);
  {
    std::vector<uint32_t> h_ev_start_indices;
    uint32_t cnt = 0;
    for (size_t i = 0; i < h_local_num_keys_list.size(); ++i) {
      int ev_size = h_local_ev_size_list[i];
      int num_keys = h_local_num_keys_list[i];
      for (int ik = 0; ik < num_keys; ++ik) {
        h_ev_start_indices.push_back(cnt);
        cnt += ev_size;
      }
    }
    h_ev_start_indices.push_back(cnt);
    core23::copy_sync(wgrad->ev_start_indices.data(), h_ev_start_indices.data(),
                      wgrad->ev_start_indices.num_bytes(), wgrad->ev_start_indices.device(),
                      core23::DeviceType::CPU);
  }
  return *this;
}

AllreduceWgradInitializer &AllreduceWgradInitializer::init_data(bool not_grouped) {
  wgrad->attr = wgrad_attr;
  int gpu_id = core->get_global_gpu_id();
  std::vector<int> h_local_ev_size_list = get_wgrad_ev_size(ebc_param, grouped_id, gpu_id);
  std::vector<int> h_unique_table_ids(wgrad_attr.sorted_unique_table_ids.num_elements());
  core23::copy_sync(h_unique_table_ids, wgrad_attr.sorted_unique_table_ids);
  std::vector<int> h_local_num_keys_list =
      get_allreduce_buffer_num_keys(h_unique_table_ids, table_id_to_vocabulary_size);

  HugeCTR::CudaDeviceContext context(core->get_device_id());

  int64_t max_buffer_size = 0;
  for (size_t i = 0; i < h_local_num_keys_list.size(); ++i) {
    max_buffer_size += h_local_num_keys_list[i] * h_local_ev_size_list[i];
  }
  wgrad->max_buffer_size = max_buffer_size;
  if (not_grouped) {
    core23::Device device(core23::DeviceType::GPU, core->get_device_id());
    core23::TensorParams params = core23::TensorParams().device(device);
    wgrad->data = core23::Tensor(params.shape({max_buffer_size}).data_type(wgrad->attr.type));
  }

  return *this;
}

AllreduceWgradInitializer &AllreduceWgradInitializer::init_data(
    bool grouped, const core23::BufferChannel &buffer_channel) {
  wgrad->attr = wgrad_attr;
  int gpu_id = core->get_global_gpu_id();
  std::vector<int> h_local_ev_size_list = get_wgrad_ev_size(ebc_param, grouped_id, gpu_id);
  std::vector<int> h_unique_table_ids(wgrad_attr.sorted_unique_table_ids.num_elements());
  core23::copy_sync(h_unique_table_ids, wgrad_attr.sorted_unique_table_ids);
  std::vector<int> h_local_num_keys_list =
      get_allreduce_buffer_num_keys(h_unique_table_ids, table_id_to_vocabulary_size);

  HugeCTR::CudaDeviceContext context(core->get_device_id());

  int64_t max_buffer_size = 0;
  for (size_t i = 0; i < h_local_num_keys_list.size(); ++i) {
    max_buffer_size += h_local_num_keys_list[i] * h_local_ev_size_list[i];
  }
  wgrad->max_buffer_size = max_buffer_size;

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams wgrads_params = core23::TensorParams().device(device);
  int alignment_num = 32;
  if (grouped) {
    // out-of-place modifications
    wgrads_params = wgrads_params.alignment(alignment_num).buffer_channel(buffer_channel);
  }
  wgrad->data = core23::Tensor(wgrads_params.shape({max_buffer_size}).data_type(wgrad->attr.type));

  return *this;
}
}  // namespace embedding
