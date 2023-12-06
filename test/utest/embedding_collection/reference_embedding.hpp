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
#pragma once

#include <gtest/gtest.h>

#include <embedding/embedding.hpp>
#include <embedding/operators/communication.hpp>
#include <embedding/operators/keys_to_indices.hpp>
#include <iterator>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utest/embedding_collection/embedding_collection_utils.hpp>
#include <utest/embedding_collection/embedding_table_cpu.hpp>

using namespace embedding;

template <typename key_t, typename offset_t, typename index_t, typename emb_t>
class EmbeddingReferenceCPU {
 public:
  int num_gpus_;
  int num_lookup_;
  std::vector<LookupParam> lookup_params_;
  int num_table_;
  EmbeddingTableCPU<key_t, index_t> emb_table_cpu_;

  std::vector<int> ev_offset_list_;

  std::vector<std::vector<emb_t>> embedding_vec_;
  std::vector<std::unordered_map<key_t, std::vector<float>>> accumulate_grad_map_;

  EmbeddingLayout output_layout_;

  EmbeddingReferenceCPU(int num_gpus, const EmbeddingCollectionParam &ebc_param, int num_table,
                        const std::vector<EmbeddingTableParam> &table_param_list,
                        std::vector<std::vector<IGroupedEmbeddingTable *>> emb_storages,
                        EmbeddingLayout output_layout)
      : num_gpus_(num_gpus),
        num_lookup_(ebc_param.num_lookup),
        lookup_params_(ebc_param.lookup_params),
        num_table_(num_table),
        emb_table_cpu_{num_table, emb_storages, table_param_list},
        ev_offset_list_{0},
        output_layout_{output_layout} {
    HCTR_CHECK_HINT(num_gpus == static_cast<int>(emb_storages.size()),
                    "EmbeddingReferenceCPU emb_storages size not match with num_gpus");
    for (int i = 0; i < num_lookup_; ++i) {
      if (ebc_param.lookup_params[i].combiner == Combiner::Concat) {
        ev_offset_list_.push_back(ebc_param.lookup_params[i].max_hotness *
                                  ebc_param.lookup_params[i].ev_size);
      } else {
        ev_offset_list_.push_back(ebc_param.lookup_params[i].ev_size);
      }
    }
    std::partial_sum(ev_offset_list_.begin(), ev_offset_list_.end(), ev_offset_list_.begin());
  }

  void embedding_forward_cpu(const std::vector<key_t> &keys,
                             const std::vector<offset_t> &bucket_range) {
    int batch_size = (static_cast<int>(bucket_range.size()) - 1) / num_lookup_;
    int batch_size_per_gpu = batch_size / num_gpus_;

    embedding_vec_.resize(num_gpus_);
    for (int gpu_id = 0; gpu_id < num_gpus_; ++gpu_id) {
      embedding_vec_[gpu_id].clear();
      embedding_vec_[gpu_id].resize(ev_offset_list_.back() * batch_size_per_gpu);
    }

    for (int lookup_id = 0; lookup_id < num_lookup_; ++lookup_id) {
      int table_id = lookup_params_[lookup_id].table_id;
      int ev_size = lookup_params_[lookup_id].ev_size;
      int ev_offset = ev_offset_list_[lookup_id];
      Combiner combiner = lookup_params_[lookup_id].combiner;
      int max_hotness = lookup_params_[lookup_id].max_hotness;

      for (int b = 0; b < batch_size; ++b) {
        int bucket_id = lookup_id * batch_size + b;
        int gpu_id = b / batch_size_per_gpu;
        int local_b = b % batch_size_per_gpu;
        uint32_t start = static_cast<uint32_t>(bucket_range[bucket_id]);
        uint32_t end = static_cast<uint32_t>(bucket_range[bucket_id + 1]);

        std::vector<std::vector<float>> lookup_evs;
        for (int r = 0; r < static_cast<int>(end - start); ++r) {
          key_t k = keys[start + r];
          ASSERT_TRUE(table_id < emb_table_cpu_.emb_table_list_.size());
          ASSERT_TRUE(emb_table_cpu_.emb_table_list_[table_id].find(k) !=
                      emb_table_cpu_.emb_table_list_[table_id].end());
          lookup_evs.push_back(emb_table_cpu_.emb_table_list_[table_id][k]);
        }
        if (combiner == Combiner::Sum || combiner == Combiner::Average) {
          for (int e = 0; e < ev_size; ++e) {
            float v = 0.f;

            for (int r = 0; r < static_cast<int>(end - start); ++r) {
              v += lookup_evs[r][e];
            }
            if (combiner == Combiner::Average && (end - start > 0)) {
              v /= (end - start);
            }
            int dst;
            if (output_layout_ == embedding::EmbeddingLayout::FeatureMajor) {
              dst = ev_offset * batch_size_per_gpu + local_b * ev_size + e;
            } else {
              ASSERT_TRUE(output_layout_ == embedding::EmbeddingLayout::BatchMajor);
              dst = ev_offset_list_.back() * local_b + ev_offset + e;
            }
            embedding_vec_[gpu_id][dst] = HugeCTR::TypeConvert<emb_t, float>::convert(v);
          }
        } else {
          for (int r = 0; r < static_cast<int>(end - start); ++r) {
            for (int e = 0; e < ev_size; ++e) {
              int dst;
              if (output_layout_ == embedding::EmbeddingLayout::FeatureMajor) {
                dst = ev_offset * batch_size_per_gpu + local_b * max_hotness * ev_size +
                      r * ev_size + e;
              } else {
                ASSERT_TRUE(output_layout_ == embedding::EmbeddingLayout::BatchMajor);
                dst = ev_offset_list_.back() * local_b + ev_offset + r * ev_size + e;
              }
              embedding_vec_[gpu_id][dst] =
                  HugeCTR::TypeConvert<emb_t, float>::convert(lookup_evs[r][e]);
            }
          }
        }
      }
    }
  }

  void embedding_backward_cpu(const std::vector<std::vector<emb_t>> &top_grad,
                              const std::vector<key_t> &keys,
                              const std::vector<offset_t> &bucket_range) {
    int batch_size = (static_cast<int>(bucket_range.size()) - 1) / num_lookup_;
    int batch_size_per_gpu = batch_size / num_gpus_;

    std::vector<std::unordered_map<key_t, std::vector<std::vector<float>>>> grouped_grads;
    grouped_grads.resize(num_table_);

    for (int lookup_id = 0; lookup_id < num_lookup_; ++lookup_id) {
      int table_id = lookup_params_[lookup_id].table_id;
      int ev_size = lookup_params_[lookup_id].ev_size;
      int ev_offset = ev_offset_list_[lookup_id];
      Combiner combiner = lookup_params_[lookup_id].combiner;
      int max_hotness = lookup_params_[lookup_id].max_hotness;

      for (int b = 0; b < batch_size; ++b) {
        int bucket_id = lookup_id * batch_size + b;
        int gpu_id = b / batch_size_per_gpu;
        int local_b = b % batch_size_per_gpu;
        uint32_t start = static_cast<uint32_t>(bucket_range[bucket_id]);
        uint32_t end = static_cast<uint32_t>(bucket_range[bucket_id + 1]);

        std::vector<float> grad_ev;

        if (combiner == Combiner::Sum || combiner == Combiner::Average) {
          for (int e = 0; e < ev_size; ++e) {
            int src_id;
            if (output_layout_ == embedding::EmbeddingLayout::FeatureMajor) {
              src_id = ev_offset * batch_size_per_gpu + local_b * ev_size + e;
            } else {
              src_id = local_b * ev_offset_list_.back() + ev_offset + e;
            }
            float gi = HugeCTR::TypeConvert<float, emb_t>::convert(top_grad[gpu_id][src_id]);
            grad_ev.push_back(gi);
          }
        } else {
          for (int e = 0; e < max_hotness * ev_size; ++e) {
            int src_id;
            if (output_layout_ == embedding::EmbeddingLayout::FeatureMajor) {
              src_id = ev_offset * batch_size_per_gpu + local_b * max_hotness * ev_size + e;
            } else {
              int r = e / ev_size;
              src_id = local_b * ev_offset_list_.back() + ev_offset + r * ev_size + e;
            }
            float gi = HugeCTR::TypeConvert<float, emb_t>::convert(top_grad[gpu_id][src_id]);
            grad_ev.push_back(gi);
          }
        }

        if (combiner == Combiner::Average) {
          for (size_t i = 0; i < grad_ev.size(); ++i) {
            grad_ev[i] /= (end - start);
          }
        }
        for (int r = 0; r < static_cast<int>(end - start); ++r) {
          key_t k = keys[start + r];
          std::vector<float> ev;
          for (int e = 0; e < ev_size; ++e) {
            float gi;
            if (combiner == Combiner::Sum || combiner == Combiner::Average) {
              gi = grad_ev[e];
            } else {
              gi = grad_ev[r * ev_size + e];
            }
            ev.push_back(gi);
          }
          grouped_grads[table_id][k].push_back(ev);
        }
      }
    }

    accumulate_grad_map_.clear();
    accumulate_grad_map_.resize(num_table_);
    for (size_t table_id = 0; table_id < grouped_grads.size(); ++table_id) {
      auto &table_grad = grouped_grads[table_id];
      for (auto &[k, evs] : table_grad) {
        HCTR_CHECK(evs.size() >= 1);
        int ev_size = evs[0].size();
        std::vector<float> ev_sum;
        for (int e = 0; e < ev_size; ++e) {
          std::vector<float> arr;
          for (size_t i = 0; i < evs.size(); ++i) {
            arr.push_back(evs[i][e]);
          }
          float summation = kahanSum(arr);
          ev_sum.push_back(summation);
        }
        accumulate_grad_map_[table_id][k] = ev_sum;
      }
    }
  }

  void embedding_update_cpu() { emb_table_cpu_.update(accumulate_grad_map_); }
};

void copy_float_emb_vec_ptrs_data(const core23::Tensor &float_emb_vec_ptrs, core23::Tensor &data,
                                  size_t num_keys, int vec_size, cudaStream_t stream);

struct CombinerData {
  core23::Tensor combiner_division;  // int
  core23::Tensor ev_start_indices;   // int
  core23::Tensor ev_size;            // int
  size_t num_bucket;
  int max_ev_size;
};

void combiner_func(const CombinerData &combiner_data, core23::Tensor &data, cudaStream_t stream);

void copy_to_float_bufer(const core23::Tensor &data, core23::Tensor &float_buffer,
                         int ev_start_indices, cudaStream_t stream);

template <typename KeyType>
struct WgradKey {
  KeyType key;
  int table_id;
};

template <typename KeyType>
bool operator==(const WgradKey<KeyType> &lhs, const WgradKey<KeyType> &rhs) {
  return (lhs.key == rhs.key) && (lhs.table_id == rhs.table_id);
}

namespace std {
template <>
struct hash<WgradKey<uint32_t>> {
  size_t operator()(const WgradKey<uint32_t> &k) const { return std::hash<uint32_t>{}(k.key); }
};
template <>
struct hash<WgradKey<int32_t>> {
  size_t operator()(const WgradKey<int32_t> &k) const { return std::hash<uint32_t>{}(k.key); }
};

template <>
struct hash<WgradKey<uint64_t>> {
  size_t operator()(const WgradKey<uint64_t> &k) const { return std::hash<uint32_t>{}(k.key); }
};

template <>
struct hash<WgradKey<int64_t>> {
  size_t operator()(const WgradKey<int64_t> &k) const { return std::hash<uint32_t>{}(k.key); }
};
}  // namespace std

template <typename KeyType>
using WgradDict = std::unordered_map<WgradKey<KeyType>, std::vector<float>>;

template <typename KeyType>
void compare_wgard_dict(const WgradDict<KeyType> &gpu_wgrad, const WgradDict<KeyType> &ref_wgrad,
                        float threshold) {
  ASSERT_GE(gpu_wgrad.size(), ref_wgrad.size());
  for (auto &[key, ev] : ref_wgrad) {
    ASSERT_TRUE(gpu_wgrad.find(key) != gpu_wgrad.end());
  }
  for (auto &[key, ev] : gpu_wgrad) {
    if (ref_wgrad.find(key) == ref_wgrad.end()) {
      // in dp dense allreduce, there will be grad for every key in the table.
      // and the ref_wgrad only has keys that have non-zero grad
      for (auto i : ev) {
        ASSERT_LE(i, threshold) << ",lhs:" << ev[i] << "\n";
      }
    } else {
      auto &ref_ev = ref_wgrad.at(key);
      ASSERT_EQ(ev.size(), ref_ev.size());

      for (size_t i = 0; i < ev.size(); ++i) {
        float lhs = ev[i];
        float rhs = ref_ev[i];
        while (std::abs(lhs) > 10.f) {
          lhs /= 10.f;
          rhs /= 10.f;
        }
        float error = std::abs(lhs - rhs);
        ASSERT_LE(error, threshold) << ",lhs:" << lhs << "rhs:" << rhs << " at " << i << "\n";
      }
    }
  }
}

template <typename KeyType, typename OffsetType, typename EmbType>
class EmbeddingReferenceGPU {
 private:
  std::shared_ptr<core::CoreResourceManager> core_;
  EmbeddingCollectionParam ebc_param_;
  std::vector<EmbeddingTableParam> emb_table_param_;
  std::vector<IGroupedEmbeddingTable *> emb_storages_;

  int batch_size_;
  int num_gpus_;
  int num_local_gpus_;

  int batch_size_per_gpu_;
  int num_table_;
  int num_lookup_;
  int vec_size_per_sample_;
  int max_ev_size_;
  int max_hotness_in_all_lookup_;

  std::vector<int> lookup_id_to_table_id;
  std::vector<int> lookup_id_to_ev_start_indices;

  std::vector<KeysToIndicesConverter> keys_to_indices_converters;

  struct LocalBatchInput {
    core23::Tensor dp_keys;
    core23::Tensor feature_ids;
    core23::Tensor ev_start_indices;  // int
    core23::Tensor d_num_keys;
  } local_batch_input;

  struct GlobalBatchInput {
    core23::Tensor d_k_per_gpu;  // uint64_t
    core23::Tensor h_k_per_gpu;  // uint64_t

    core23::Tensor allgather_keys;
    core23::Tensor allgather_feature_ids;
    core23::Tensor allgather_ev_start_indices;
    size_t num_allgather_keys;

    std::vector<KeyType> gpu_allgather_keys;
    std::vector<int> gpu_allgather_feature_ids;
    std::vector<int> gpu_allgather_ev_start_indices;
  } global_batch_input;

  struct LookupData {
    // lookup
    core23::Tensor keys;  // key_type
    size_t num_keys;
    core23::Tensor table_range;    // offset_type
    core23::Tensor table_id_list;  // int
    std::vector<int> ev_start_indices;

    core23::Tensor d_float_emb_vec_ptrs;
    core23::Tensor d_float_emb_vec_data;
  } lookup_data;

  CombinerData combiner_data;

  struct EmbVecBuffer {
    core23::Tensor h_global_batch_float_emb_output;
    core23::Tensor d_global_batch_float_emb_output;
    core23::Tensor float_emb_output;

    core23::Tensor d_local_batch_float_top_grad;
    core23::Tensor d_global_batch_float_top_grad;
  } buffer;

  std::tuple<int, int, int> parse_feature_id(const int &feature_id) {
    int gpu_id = feature_id / (batch_size_per_gpu_ * num_lookup_);
    int lookup_id = (feature_id % (batch_size_per_gpu_ * num_lookup_)) / batch_size_per_gpu_;
    int bid = (feature_id % (batch_size_per_gpu_ * num_lookup_)) % batch_size_per_gpu_;
    return {gpu_id, lookup_id, bid};
  }

  void fwd_idx_cal(const std::vector<core23::Tensor> &sparse_dp_bucket_range,
                   const std::vector<core23::Tensor> &sparse_dp_tensors);

  void broadcast_input();

  void copy_broadcast_input() {
    global_batch_input.gpu_allgather_keys.clear();
    global_batch_input.gpu_allgather_keys.resize(global_batch_input.allgather_keys.num_elements());
    auto &gpu_allgather_keys = global_batch_input.gpu_allgather_keys;
    core23::copy_sync(gpu_allgather_keys, global_batch_input.allgather_keys);

    global_batch_input.gpu_allgather_feature_ids.clear();
    global_batch_input.gpu_allgather_feature_ids.resize(
        global_batch_input.allgather_feature_ids.num_elements());
    auto &gpu_allgather_feature_ids = global_batch_input.gpu_allgather_feature_ids;
    core23::copy_sync(gpu_allgather_feature_ids, global_batch_input.allgather_feature_ids);

    global_batch_input.gpu_allgather_ev_start_indices.clear();
    global_batch_input.gpu_allgather_ev_start_indices.resize(
        global_batch_input.allgather_ev_start_indices.num_elements());
    auto &gpu_allgather_ev_start_indices = global_batch_input.gpu_allgather_ev_start_indices;
    core23::copy_sync(gpu_allgather_ev_start_indices,
                      global_batch_input.allgather_ev_start_indices);
  }

  bool filter_keys(int group_id, int lookup_id, bool is_forward) {
    auto stream = core_->get_local_gpu()->get_stream();

    auto &gpu_allgather_keys = global_batch_input.gpu_allgather_keys;
    auto &gpu_allgather_feature_ids = global_batch_input.gpu_allgather_feature_ids;
    auto &gpu_allgather_ev_start_indices = global_batch_input.gpu_allgather_ev_start_indices;

    auto &grouped_lookup_param = ebc_param_.grouped_lookup_params[group_id];

    auto &lookup_param = ebc_param_.lookup_params[lookup_id];
    int table_id = lookup_param.table_id;

    auto embedding_group_type = grouped_lookup_param.embedding_group_type;

    std::vector<KeyType> keys;
    lookup_data.ev_start_indices.clear();
    lookup_data.num_keys = 0ul;
    for (size_t i = 0; i < global_batch_input.num_allgather_keys; ++i) {
      auto key = gpu_allgather_keys[i];
      int feature_id = gpu_allgather_feature_ids[i];
      auto [gid, lid, bid] = parse_feature_id(feature_id);
      int gpu_id = core_->get_global_gpu_id();
      // compare with forward, in backward we need to accumulate dp keys for all gpus
      bool dp_valid_key = is_forward ? (embedding_group_type == EmbeddingGroupType::DataParallel &&
                                        lid == lookup_id && gid == gpu_id)
                                     : (embedding_group_type == EmbeddingGroupType::DataParallel &&
                                        lid == lookup_id);

      bool mp_valid_key =
          ((embedding_group_type == EmbeddingGroupType::SparseModelParallel ||
            embedding_group_type == EmbeddingGroupType::DenseModelParallel) &&
           lid == lookup_id && ebc_param_.has_table_shard(gpu_id, group_id, lookup_id));
      if (mp_valid_key) {
        int shard_id, num_shard;
        ebc_param_.get_table_shard_id(gpu_id, table_id, &shard_id, &num_shard);
        mp_valid_key &= ((static_cast<int>(key) % num_shard) == shard_id);
      }
      if (dp_valid_key || mp_valid_key) {
        keys.push_back(key);
        lookup_data.ev_start_indices.push_back(gpu_allgather_ev_start_indices[i]);
        lookup_data.num_keys += 1;
      }
    }
    if (lookup_data.num_keys == 0) return false;

    std::vector<OffsetType> table_range{0, static_cast<OffsetType>(lookup_data.num_keys)};
    std::vector<int> table_id_list{table_id};
    HCTR_LIB_THROW(cudaMemcpy(lookup_data.keys.data(), keys.data(),
                              lookup_data.num_keys * sizeof(KeyType), cudaMemcpyHostToDevice));
    core23::copy_sync(lookup_data.table_range, table_range);
    core23::copy_sync(lookup_data.table_id_list, table_id_list);

    if (!keys_to_indices_converters.empty()) {
      keys_to_indices_converters[group_id].convert(lookup_data.keys, lookup_data.num_keys,
                                                   lookup_data.table_range,
                                                   lookup_data.table_id_list);
    }
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
    return true;
  }

  void table_lookup();

  void forward_communication() {
    auto stream = core_->get_local_gpu()->get_stream();
    auto comm = core_->get_nccl();

    core23::copy_async(buffer.d_global_batch_float_emb_output,
                       buffer.h_global_batch_float_emb_output, stream);

    HCTR_LIB_THROW(ncclReduceScatter(buffer.d_global_batch_float_emb_output.data(),
                                     buffer.float_emb_output.data(),
                                     batch_size_per_gpu_ * vec_size_per_sample_, ncclFloat32,
                                     ncclRedOp_t::ncclSum, comm, stream));
    combiner_func(combiner_data, buffer.float_emb_output, stream);
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  }

  void backward_communication(const core23::Tensor &top_grad) {
    auto stream = core_->get_local_gpu()->get_stream();
    auto comm = core_->get_nccl();
    int gpu_id = core_->get_global_gpu_id();

    copy_to_float_bufer(top_grad, buffer.d_local_batch_float_top_grad, 0, stream);
    combiner_func(combiner_data, buffer.d_local_batch_float_top_grad, stream);

    core23::zeros_async(buffer.d_global_batch_float_top_grad, stream);
    copy_to_float_bufer(buffer.d_local_batch_float_top_grad, buffer.d_global_batch_float_top_grad,
                        gpu_id * batch_size_per_gpu_ * vec_size_per_sample_, stream);

    HCTR_LIB_THROW(ncclAllReduce(buffer.d_global_batch_float_top_grad.data(),
                                 buffer.d_global_batch_float_top_grad.data(),
                                 batch_size_per_gpu_ * vec_size_per_sample_ * num_gpus_,
                                 ncclFloat32, ncclRedOp_t::ncclSum, comm, stream));
    HCTR_LIB_THROW(cudaStreamSynchronize(stream));
  }

  WgradDict<KeyType> get_gpu_wgrad(const Wgrad &wgrad) {
    std::vector<uint64_t> gpu_num_unique_keys(wgrad.num_unique_keys.num_elements());
    core23::copy_sync(gpu_num_unique_keys, wgrad.num_unique_keys);

    WgradDict<KeyType> wgrad_dict;
    if (gpu_num_unique_keys[0] == 0) return wgrad_dict;

    std::vector<KeyType> gpu_unique_keys(wgrad.unique_keys.num_elements());
    core23::copy_sync(gpu_unique_keys, wgrad.unique_keys);

    std::vector<int> gpu_table_ids(wgrad.table_ids.num_elements());
    core23::copy_sync(gpu_table_ids, wgrad.table_ids);

    std::vector<uint32_t> gpu_ev_start_indices(wgrad.ev_start_indices.num_elements());
    core23::copy_sync(gpu_ev_start_indices, wgrad.ev_start_indices);

    std::vector<EmbType> gpu_data(wgrad.data.num_elements());
    core23::copy_sync(gpu_data, wgrad.data);

    for (uint64_t i = 0; i < gpu_num_unique_keys[0]; ++i) {
      KeyType key = gpu_unique_keys[i];
      int table_id = gpu_table_ids[i];
      int ev_size = emb_table_param_[table_id].ev_size;
      uint32_t ev_start_indices = gpu_ev_start_indices[i];
      std::vector<float> ev(ev_size, 0.f);

      for (int ev_id = 0; ev_id < ev_size; ++ev_id) {
        ev[ev_id] =
            HugeCTR::TypeConvert<float, EmbType>::convert(gpu_data[ev_start_indices + ev_id]);
      }
      WgradKey<KeyType> wgrad_key{key, table_id};
      wgrad_dict.insert({wgrad_key, ev});
    }
    return wgrad_dict;
  }

 public:
  EmbeddingReferenceGPU(const std::shared_ptr<core::CoreResourceManager> &core,
                        const EmbeddingCollectionParam &ebc_param,
                        const std::vector<EmbeddingTableParam> &table_param_list,
                        std::vector<IGroupedEmbeddingTable *> emb_storages)
      : core_(core),
        ebc_param_(ebc_param),
        emb_table_param_(table_param_list),
        emb_storages_(emb_storages),
        batch_size_(ebc_param.universal_batch_size),
        num_gpus_(core->get_global_gpu_count()),
        num_local_gpus_(core->get_local_gpu_count()),
        batch_size_per_gpu_(batch_size_ / num_gpus_),
        num_table_(ebc_param.num_table),
        num_lookup_(ebc_param.num_lookup) {
    HugeCTR::CudaDeviceContext context(core_->get_device_id());

    auto key_type = ebc_param.key_type;
    auto offset_type = ebc_param.offset_type;
    //    auto emb_type = ebc_param.emb_type;
    this->lookup_id_to_table_id.resize(num_lookup_);
    this->lookup_id_to_ev_start_indices.resize(num_lookup_ + 1);

    int64_t num_features_per_sample = 0;
    this->vec_size_per_sample_ = 0;
    this->max_ev_size_ = 0;
    this->max_hotness_in_all_lookup_ = 0;
    for (int lookup_id = 0; lookup_id < num_lookup_; ++lookup_id) {
      num_features_per_sample += ebc_param.lookup_params[lookup_id].max_hotness;

      this->lookup_id_to_table_id[lookup_id] = ebc_param.lookup_params[lookup_id].table_id;

      this->lookup_id_to_ev_start_indices[lookup_id] = this->vec_size_per_sample_;
      int ev_size = ebc_param.lookup_params[lookup_id].ev_size;
      int max_hotness = ebc_param.lookup_params[lookup_id].max_hotness;
      auto combiner = ebc_param.lookup_params[lookup_id].combiner;
      this->vec_size_per_sample_ +=
          (combiner == Combiner::Concat) ? ev_size * max_hotness : ev_size;

      this->max_ev_size_ = std::max(this->max_ev_size_, ev_size);
      this->max_hotness_in_all_lookup_ = std::max(this->max_hotness_in_all_lookup_, max_hotness);
    }
    this->lookup_id_to_ev_start_indices[num_lookup_] = this->vec_size_per_sample_;

    core23::Device device(core23::DeviceType::GPU, core_->get_device_id());
    core23::TensorParams params = core23::TensorParams().device(device);

    if (ebc_param.keys_preprocess_strategy_ == KeysPreprocessStrategy::AddOffset) {
      for (size_t group_id = 0; group_id < ebc_param.grouped_lookup_params.size(); ++group_id) {
        keys_to_indices_converters.emplace_back(core, table_param_list, ebc_param, group_id);
      }
    }

    this->local_batch_input.dp_keys = core23::Tensor(
        params.shape({num_features_per_sample * batch_size_per_gpu_}).data_type(key_type));
    this->local_batch_input.feature_ids =
        core23::Tensor(params.shape({num_features_per_sample * batch_size_per_gpu_})
                           .data_type(core23::ScalarType::Int32));
    this->local_batch_input.ev_start_indices =
        core23::Tensor(params.shape({num_features_per_sample * batch_size_per_gpu_})
                           .data_type(core23::ScalarType::Int32));
    this->local_batch_input.d_num_keys =
        core23::Tensor(params.shape({1}).data_type(core23::ScalarType::UInt64));

    this->global_batch_input.d_k_per_gpu =
        core23::Tensor(params.shape({num_gpus_}).data_type(core23::ScalarType::UInt64));
    this->global_batch_input.h_k_per_gpu = core23::Tensor(params.shape({num_gpus_})
                                                              .data_type(core23::ScalarType::UInt64)
                                                              .device(core23::DeviceType::CPU));
    this->global_batch_input.allgather_keys =
        core23::Tensor(params.shape({num_features_per_sample * batch_size_}).data_type(key_type));
    this->global_batch_input.allgather_feature_ids = core23::Tensor(
        params.shape({num_features_per_sample * batch_size_}).data_type(core23::ScalarType::Int32));
    this->global_batch_input.allgather_ev_start_indices = core23::Tensor(
        params.shape({num_features_per_sample * batch_size_}).data_type(core23::ScalarType::Int32));

    this->lookup_data.keys = core23::Tensor(
        params.shape({max_hotness_in_all_lookup_ * batch_size_}).data_type(key_type));
    this->lookup_data.table_range = core23::Tensor(params.shape({2}).data_type(offset_type));
    this->lookup_data.table_id_list =
        core23::Tensor(params.shape({1}).data_type(core23::ScalarType::Int32));
    this->lookup_data.d_float_emb_vec_ptrs =
        core23::Tensor(params.shape({max_hotness_in_all_lookup_ * batch_size_})
                           .data_type(core23::ScalarType::Pointer));
    this->lookup_data.d_float_emb_vec_data =
        core23::Tensor(params.shape({max_ev_size_ * batch_size_ * max_hotness_in_all_lookup_})
                           .data_type(core23::ScalarType::Float));

    this->combiner_data.combiner_division =
        core23::Tensor(params.shape({num_features_per_sample * batch_size_per_gpu_})
                           .data_type(core23::ScalarType::Int32));
    this->combiner_data.ev_start_indices =
        core23::Tensor(params.shape({num_features_per_sample * batch_size_per_gpu_})
                           .data_type(core23::ScalarType::Int32));
    this->combiner_data.ev_size =
        core23::Tensor(params.shape({num_features_per_sample * batch_size_per_gpu_})
                           .data_type(core23::ScalarType::Int32));
    this->combiner_data.num_bucket = 0ul;
    this->combiner_data.max_ev_size = max_ev_size_;

    this->buffer.h_global_batch_float_emb_output =
        core23::Tensor(params.shape({batch_size_ * vec_size_per_sample_})
                           .data_type(core23::ScalarType::Float)
                           .device(core23::DeviceType::CPU));
    this->buffer.d_global_batch_float_emb_output = core23::Tensor(
        params.shape({batch_size_ * vec_size_per_sample_}).data_type(core23::ScalarType::Float));
    this->buffer.float_emb_output =
        core23::Tensor(params.shape({batch_size_per_gpu_ * vec_size_per_sample_})
                           .data_type(core23::ScalarType::Float));
    this->buffer.d_local_batch_float_top_grad =
        core23::Tensor(params.shape({batch_size_per_gpu_ * vec_size_per_sample_})
                           .data_type(core23::ScalarType::Float));
    this->buffer.d_global_batch_float_top_grad =
        core23::Tensor(params.shape({num_gpus_ * batch_size_per_gpu_ * vec_size_per_sample_})
                           .data_type(core23::ScalarType::Float));
  }

  void embedding_forward_cpu(const std::vector<core23::Tensor> &sparse_dp_bucket_range,
                             const std::vector<core23::Tensor> &sparse_dp_tensors);

  void embedding_backward_cpu(const core23::Tensor &top_grad);

  void compare_forward_result(const core23::Tensor &ebc_outptut) {
    HugeCTR::CudaDeviceContext context(core_->get_device_id());

    std::vector<EmbType> gpu_ebc_output(ebc_outptut.num_elements());
    core23::copy_sync(gpu_ebc_output, ebc_outptut);

    std::vector<float> reference_ebc_output(buffer.float_emb_output.num_elements());
    core23::copy_sync(reference_ebc_output, buffer.float_emb_output);
    //  std::cout << "forward ref output:\n";
    //  print_array(reference_ebc_output.size(), reference_ebc_output);
    //  std::cout << "forward gpu output:\n";
    //  print_array(gpu_ebc_output.size(), gpu_ebc_output);

    std::cout << "gpu_id:" << core_->get_global_gpu_id()
              << " compare ebc gpu emb output vs. emb reference emb output.\n";

    ASSERT_GE(gpu_ebc_output.size(), reference_ebc_output.size());
    float threshold = 1e-5;
    for (size_t i = 0; i < gpu_ebc_output.size(); ++i) {
      float lhs = HugeCTR::TypeConvert<float, EmbType>::convert(gpu_ebc_output[i]);
      float rhs = reference_ebc_output[i];
      while (std::abs(lhs) > 10.f) {
        lhs /= 10.f;
        rhs /= 10.f;
      }
      float error = std::abs(lhs - rhs);
      ASSERT_LE(error, threshold) << ",lhs:" << lhs << ",rhs:" << rhs << " at " << i << "\n";
    }
  }

  void compare_backward_result(const std::vector<Wgrad> &wgrads) {
    HugeCTR::CudaDeviceContext context(core_->get_device_id());

    std::vector<float> gpu_global_batch_float_top_grad(
        buffer.d_global_batch_float_top_grad.num_elements());
    core23::copy_sync(gpu_global_batch_float_top_grad, buffer.d_global_batch_float_top_grad);

    std::cout << "gpu_id:" << core_->get_global_gpu_id()
              << " compare ebc gpu wgrad vs. emb reference wgrad.\n";

    for (size_t group_id = 0; group_id < ebc_param_.grouped_lookup_params.size(); ++group_id) {
      std::cout << "compare group id:" << group_id << " wgrad" << std::endl;

      auto &wgrad = wgrads[group_id];
      auto gpu_wgrad_dict = get_gpu_wgrad(wgrad);

      auto &grouped_lookup_param = ebc_param_.grouped_lookup_params[group_id];

      std::unordered_map<WgradKey<KeyType>, std::vector<std::vector<float>>> wgrad_accum_dict;

      for (auto lookup_id : grouped_lookup_param.lookup_ids) {
        if (!filter_keys(group_id, lookup_id, false)) continue;

        std::vector<KeyType> gpu_filtered_keys(lookup_data.keys.num_elements());
        core23::copy_sync(gpu_filtered_keys, lookup_data.keys);
        std::vector<int> gpu_filtered_table_ids(lookup_data.table_id_list.num_elements());
        core23::copy_sync(gpu_filtered_table_ids, lookup_data.table_id_list);

        int table_id = gpu_filtered_table_ids[0];
        for (size_t i = 0; i < lookup_data.num_keys; ++i) {
          int ev_size = ebc_param_.lookup_params[lookup_id].ev_size;
          std::vector<float> ev(ev_size, 0.f);

          int ev_start_indices = lookup_data.ev_start_indices[i];
          for (int ev_id = 0; ev_id < ev_size; ++ev_id) {
            ev[ev_id] = gpu_global_batch_float_top_grad[ev_start_indices + ev_id];
          }
          KeyType key = gpu_filtered_keys[i];
          WgradKey<KeyType> wgrad_key{key, table_id};
          wgrad_accum_dict[wgrad_key].push_back(ev);
        }
      }

      WgradDict<KeyType> ref_wgrad_dict;
      for (auto &[key, evs] : wgrad_accum_dict) {
        int ev_size = static_cast<int>(evs[0].size());
        for (int i = 0; i < ev_size; ++i) {
          std::vector<float> accum;
          for (auto &ev : evs) {
            accum.push_back(ev[i]);
          }
          ref_wgrad_dict[key].push_back(kahanSum(accum));
        }
      }
      compare_wgard_dict(gpu_wgrad_dict, ref_wgrad_dict, 1e-4);
      // std::cout << "gpu <key, wgrad>:" << std::endl;
      // for (auto it = gpu_wgrad_dict.begin(); it != gpu_wgrad_dict.end(); it++) {
      //   std::cout << it->first.key << " size " << it->second.size() << " [0][0]" <<
      //   (it->second)[0]
      //             << std::endl;
      // }
      // std::cout << "ref <key, wgrad>:" << std::endl;
      // for (auto it = ref_wgrad_dict.begin(); it != ref_wgrad_dict.end(); it++) {
      //   std::cout << it->first.key << " size " << it->second.size() << " [0][0]" <<
      //   (it->second)[0]
      //             << std::endl;
      // }
    }
  }
};

template <typename KeyType, typename OffsetType, typename EmbType>
void EmbeddingReferenceGPU<KeyType, OffsetType, EmbType>::broadcast_input() {
  auto stream = core_->get_local_gpu()->get_stream();
  auto comm = core_->get_nccl();

  HCTR_LIB_THROW(ncclAllGather(local_batch_input.d_num_keys.data(),
                               global_batch_input.d_k_per_gpu.data(), 1ul, ncclUint64, comm,
                               stream));
  core23::copy_async(global_batch_input.h_k_per_gpu, global_batch_input.d_k_per_gpu, stream);
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  HCTR_LIB_THROW(ncclGroupStart());
  const uint64_t *h_k_per_gpu = global_batch_input.h_k_per_gpu.template data<uint64_t>();
  uint64_t count_offset = 0;
  auto nccl_key_type =
      core23::get_nccl_dtype_from_tensor_scalar_type_core23(ebc_param_.key_type.type());

  for (int dst_gpu_id = 0; dst_gpu_id < num_gpus_; ++dst_gpu_id) {
    uint64_t num_keys = h_k_per_gpu[dst_gpu_id];

    HCTR_LIB_THROW(
        ncclBroadcast(local_batch_input.dp_keys.data(),
                      global_batch_input.allgather_keys.template data<KeyType>() + count_offset,
                      num_keys, nccl_key_type, dst_gpu_id, comm, stream));
    HCTR_LIB_THROW(
        ncclBroadcast(local_batch_input.feature_ids.template data<int>(),
                      global_batch_input.allgather_feature_ids.template data<int>() + count_offset,
                      num_keys, ncclInt32, dst_gpu_id, comm, stream));
    HCTR_LIB_THROW(ncclBroadcast(
        local_batch_input.ev_start_indices.template data<int>(),
        global_batch_input.allgather_ev_start_indices.template data<int>() + count_offset, num_keys,
        ncclInt32, dst_gpu_id, comm, stream));

    count_offset += num_keys;
  }
  HCTR_LIB_THROW(ncclGroupEnd());
  global_batch_input.num_allgather_keys = count_offset;
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
}

template <typename KeyType, typename OffsetType, typename EmbType>
void EmbeddingReferenceGPU<KeyType, OffsetType, EmbType>::table_lookup() {
  auto stream = core_->get_local_gpu()->get_stream();
  HCTR_LIB_THROW(cudaMemset(buffer.h_global_batch_float_emb_output.data(), 0,
                            buffer.h_global_batch_float_emb_output.num_bytes()));

  for (size_t group_id = 0; group_id < ebc_param_.grouped_lookup_params.size(); ++group_id) {
    auto &grouped_lookup_param = ebc_param_.grouped_lookup_params[group_id];

    for (auto lookup_id : grouped_lookup_param.lookup_ids) {
      if (!filter_keys(group_id, lookup_id, true)) continue;

      auto grouped_table_idx = grouped_lookup_param.grouped_table_idx;
      auto emb_table = emb_storages_[grouped_table_idx];

      emb_table->lookup(lookup_data.keys, lookup_data.num_keys, lookup_data.table_range,
                        lookup_data.table_range.num_elements(), lookup_data.table_id_list,
                        lookup_data.d_float_emb_vec_ptrs);
      int ev_size = ebc_param_.lookup_params[lookup_id].ev_size;
      copy_float_emb_vec_ptrs_data(lookup_data.d_float_emb_vec_ptrs,
                                   lookup_data.d_float_emb_vec_data, lookup_data.num_keys, ev_size,
                                   stream);
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));

      std::vector<float> gpu_float_emb_vec_data(lookup_data.d_float_emb_vec_data.num_elements());
      core23::copy_sync(gpu_float_emb_vec_data, lookup_data.d_float_emb_vec_data);

      float *h_global_batch_float_emb_output_ptr =
          buffer.h_global_batch_float_emb_output.template data<float>();
      for (size_t i = 0; i < lookup_data.num_keys; ++i) {
        for (int ev_id = 0; ev_id < ev_size; ++ev_id) {
          h_global_batch_float_emb_output_ptr[lookup_data.ev_start_indices[i] + ev_id] +=
              gpu_float_emb_vec_data[i * ev_size + ev_id];
        }
      }
    }
  }
}

template <typename KeyType, typename OffsetType, typename EmbType>
void EmbeddingReferenceGPU<KeyType, OffsetType, EmbType>::fwd_idx_cal(
    const std::vector<core23::Tensor> &sparse_dp_bucket_range,
    const std::vector<core23::Tensor> &sparse_dp_tensors) {
  size_t num_keys_offset = 0;
  combiner_data.num_bucket = 0;
  std::vector<int> combiner_division;
  std::vector<int> combiner_ev_start_indices;
  std::vector<int> combiner_ev_size;
  for (int lookup_id = 0; lookup_id < num_lookup_; ++lookup_id) {
    auto &k = sparse_dp_tensors[lookup_id];
    auto &b = sparse_dp_bucket_range[lookup_id];
    std::vector<OffsetType> gpu_sparse_dp_bucket_range(b.num_elements());
    core23::copy_sync(gpu_sparse_dp_bucket_range, b);
    std::vector<int> gpu_feature_ids;
    std::vector<int> gpu_ev_start_indices;
    for (int bid = 0; bid < batch_size_per_gpu_; ++bid) {
      auto num_key_in_bucket =
          gpu_sparse_dp_bucket_range[bid + 1] - gpu_sparse_dp_bucket_range[bid];
      int bucket_id_offset = lookup_id * batch_size_per_gpu_ +
                             core_->get_local_gpu_id() * batch_size_per_gpu_ * num_lookup_;
      for (int _ = 0; _ < static_cast<int>(num_key_in_bucket); ++_) {
        gpu_feature_ids.push_back(bid + bucket_id_offset);
      }

      int max_hotness = ebc_param_.lookup_params[lookup_id].max_hotness;
      int ev_size = ebc_param_.lookup_params[lookup_id].ev_size;

      int ev_start_indices_for_current_gpu =
          vec_size_per_sample_ * batch_size_per_gpu_ * core_->get_global_gpu_id();

      auto combiner = ebc_param_.lookup_params[lookup_id].combiner;
      auto output_layout = ebc_param_.output_layout_;
      for (int i = 0; i < static_cast<int>(num_key_in_bucket); ++i) {
        if (output_layout == EmbeddingLayout::FeatureMajor && combiner == Combiner::Concat) {
          int ev_start_indices = lookup_id_to_ev_start_indices[lookup_id] * batch_size_per_gpu_ +
                                 bid * max_hotness * ev_size + i * ev_size;
          gpu_ev_start_indices.push_back(ev_start_indices_for_current_gpu + ev_start_indices);
        } else if (output_layout == EmbeddingLayout::FeatureMajor && combiner != Combiner::Concat) {
          int ev_start_indices =
              lookup_id_to_ev_start_indices[lookup_id] * batch_size_per_gpu_ + bid * ev_size;
          gpu_ev_start_indices.push_back(ev_start_indices_for_current_gpu + ev_start_indices);
        } else if (output_layout == EmbeddingLayout::BatchMajor && combiner == Combiner::Concat) {
          int ev_start_indices =
              bid * vec_size_per_sample_ + lookup_id_to_ev_start_indices[lookup_id] + i * ev_size;
          gpu_ev_start_indices.push_back(ev_start_indices_for_current_gpu + ev_start_indices);
        } else if (output_layout == EmbeddingLayout::BatchMajor && combiner != Combiner::Concat) {
          int ev_start_indices =
              bid * vec_size_per_sample_ + lookup_id_to_ev_start_indices[lookup_id];
          gpu_ev_start_indices.push_back(ev_start_indices_for_current_gpu + ev_start_indices);
        } else {
          HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError, "not supported");
        }
      }

      if (combiner == Combiner::Average) {
        combiner_division.push_back(num_key_in_bucket);
        combiner_ev_size.push_back(ev_size);
        combiner_data.num_bucket += 1;
        if (output_layout == EmbeddingLayout::BatchMajor) {
          combiner_ev_start_indices.push_back(bid * vec_size_per_sample_ +
                                              lookup_id_to_ev_start_indices[lookup_id]);
        }
        if (output_layout == EmbeddingLayout::FeatureMajor) {
          combiner_ev_start_indices.push_back(
              lookup_id_to_ev_start_indices[lookup_id] * batch_size_per_gpu_ + bid * ev_size);
        }
      }
    }

    auto num_keys = gpu_sparse_dp_bucket_range[batch_size_per_gpu_];
    HCTR_LIB_THROW(cudaMemcpy(local_batch_input.dp_keys.template data<KeyType>() + num_keys_offset,
                              k.data(), num_keys * sizeof(KeyType), cudaMemcpyDeviceToDevice));
    HCTR_LIB_THROW(cudaMemcpy(local_batch_input.feature_ids.template data<int>() + num_keys_offset,
                              gpu_feature_ids.data(), num_keys * sizeof(int),
                              cudaMemcpyHostToDevice));
    HCTR_LIB_THROW(
        cudaMemcpy(local_batch_input.ev_start_indices.template data<int>() + num_keys_offset,
                   gpu_ev_start_indices.data(), num_keys * sizeof(int), cudaMemcpyHostToDevice));

    num_keys_offset += num_keys;
  }
  HCTR_LIB_THROW(cudaMemcpy(local_batch_input.d_num_keys.data(), &num_keys_offset, sizeof(uint64_t),
                            cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaMemcpy(combiner_data.combiner_division.data(), combiner_division.data(),
                            combiner_data.num_bucket * sizeof(int), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaMemcpy(combiner_data.ev_start_indices.data(), combiner_ev_start_indices.data(),
                            combiner_data.num_bucket * sizeof(int), cudaMemcpyHostToDevice));
  HCTR_LIB_THROW(cudaMemcpy(combiner_data.ev_size.data(), combiner_ev_size.data(),
                            combiner_data.num_bucket * sizeof(int), cudaMemcpyHostToDevice));
}

template <typename KeyType, typename OffsetType, typename EmbType>
void EmbeddingReferenceGPU<KeyType, OffsetType, EmbType>::embedding_forward_cpu(
    const std::vector<core23::Tensor> &sparse_dp_bucket_range,
    const std::vector<core23::Tensor> &sparse_dp_tensors) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  auto stream = core_->get_local_gpu()->get_stream();

  fwd_idx_cal(sparse_dp_bucket_range, sparse_dp_tensors);
  broadcast_input();
  copy_broadcast_input();
  table_lookup();
  forward_communication();
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
}

template <typename KeyType, typename OffsetType, typename EmbType>
void EmbeddingReferenceGPU<KeyType, OffsetType, EmbType>::embedding_backward_cpu(
    const core23::Tensor &top_grad) {
  HugeCTR::CudaDeviceContext context(core_->get_device_id());
  auto stream = core_->get_local_gpu()->get_stream();

  backward_communication(top_grad);
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));
}
