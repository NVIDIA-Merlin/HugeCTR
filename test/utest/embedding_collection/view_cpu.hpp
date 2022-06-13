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
#pragma once
#include <iterator>
#include <numeric>

#include "HugeCTR/embedding/view.hpp"

using namespace embedding;

template <typename emb_t>
class RaggedEmbForwardResultViewCPU
    : public std::iterator<std::random_access_iterator_tag, ArrayView<emb_t>> {
 private:
  std::vector<emb_t> *forward_result_;
  const std::vector<int> *ev_size_list_;
  const std::vector<int> *ev_offset_list_;

  int batch_size_per_gpu_;

 public:
  using base = typename std::iterator<std::random_access_iterator_tag, ArrayView<emb_t>>;
  using value_type = typename base::value_type;
  using difference_type = typename base::difference_type;

  RaggedEmbForwardResultViewCPU(std::vector<emb_t>* forward_result, const std::vector<int> *ev_size_list,
                                const std::vector<int> *ev_offset_list, int batch_size_per_gpu)
      : forward_result_(forward_result),
        ev_size_list_(ev_size_list),
        ev_offset_list_(ev_offset_list),
        batch_size_per_gpu_(batch_size_per_gpu) {}

  // RaggedEmbForwardResultViewCPU(std::vector<emb_t> *forward_result, const std::vector<int> *ev_size_list,
  //                               const std::vector<int> *ev_offset_list, int batch_size_per_gpu)
  //     : forward_result_(forward_result),
  //       ev_size_list_(ev_size_list),
  //       ev_offset_list_(ev_offset_list),
  //       batch_size_per_gpu_(batch_size_per_gpu) {}

  value_type operator[](difference_type idx) {
    int embedding_id = idx / batch_size_per_gpu_;
    int batch_id = idx % batch_size_per_gpu_;

    int ev_size = ev_size_list_->at(embedding_id);
    int start = ev_offset_list_->at(embedding_id) * batch_size_per_gpu_ + ev_size * batch_id;

    assert(forward_result_->size() >= (start + ev_size));
    return ArrayView<emb_t>{forward_result_->data() + start, ev_size};
  }
};

template <typename emb_t>
class RaggedModelBufferViewCPU
    : public std::iterator<std::random_access_iterator_tag, ArrayView<emb_t>> {
 private:
  std::vector<std::vector<emb_t>> *data_;
  const std::vector<int> *local_ev_offset_list_;

  int num_gpus_;
  int batch_size_;
  int batch_size_per_gpu_;

 public:
  using base = typename std::iterator<std::random_access_iterator_tag, ArrayView<emb_t>>;
  using value_type = typename base::value_type;
  using difference_type = typename base::difference_type;

  RaggedModelBufferViewCPU(std::vector<std::vector<emb_t>> *data,
                           const std::vector<int> *local_ev_offset_list, int num_gpus,
                           int batch_size)
      : data_(data),
        local_ev_offset_list_(local_ev_offset_list),
        num_gpus_(num_gpus),
        batch_size_(batch_size),
        batch_size_per_gpu_(batch_size / num_gpus) {
  }

  value_type operator[](difference_type idx) {
    int embedding_id = idx / batch_size_;
    int batch_id = idx % batch_size_;
    int gpu_id = batch_id / batch_size_per_gpu_;
    int local_batch_id = batch_id % batch_size_per_gpu_;
    int ev_size =
        local_ev_offset_list_->at(embedding_id + 1) - local_ev_offset_list_->at(embedding_id);
    int start =
        batch_size_per_gpu_ * local_ev_offset_list_->at(embedding_id) + local_batch_id * ev_size;

    assert(data_->at(gpu_id).size() >= (start + ev_size));
    return ArrayView<emb_t>{data_->at(gpu_id).data() + start, ev_size};
  }
};

template <typename emb_t>
class RaggedNetworkBufferViewCPU
    : public std::iterator<std::random_access_iterator_tag, ArrayView<emb_t>> {
 private:
  std::vector<std::vector<emb_t>> *data_;
  std::vector<int> *gpu_idx_offset_;
  std::vector<std::vector<int>> *global_ev_offset_;

  int num_gpus_;
  int batch_size_;
  int batch_size_per_gpu_;

 public:
  using base = typename std::iterator<std::random_access_iterator_tag, ArrayView<emb_t>>;
  using value_type = typename base::value_type;
  using difference_type = typename base::difference_type;

  RaggedNetworkBufferViewCPU(std::vector<std::vector<emb_t>> *data,
                             std::vector<int> *gpu_idx_offset,
                             std::vector<std::vector<int>> *global_ev_offset, int num_gpus,
                             int batch_size)
      : data_(data),
        gpu_idx_offset_(gpu_idx_offset),
        global_ev_offset_(global_ev_offset),
        num_gpus_(num_gpus),
        batch_size_(batch_size) {
    batch_size_per_gpu_ = batch_size_ / num_gpus_;
  }

  value_type operator[](difference_type idx) {
    int gpu_id = std::upper_bound(gpu_idx_offset_->begin(), gpu_idx_offset_->end(), idx) -
                 gpu_idx_offset_->begin() - 1;

    int local_bucket_id = idx - gpu_idx_offset_->at(gpu_id);
    int local_embedding_id = local_bucket_id / batch_size_per_gpu_;
    int local_batch_id = local_bucket_id % batch_size_per_gpu_;
    int ev_size = global_ev_offset_->at(gpu_id).at(local_embedding_id + 1) -
                  global_ev_offset_->at(gpu_id).at(local_embedding_id);
    int start = global_ev_offset_->at(gpu_id).at(local_embedding_id) * batch_size_per_gpu_ +
                local_batch_id * ev_size;
    assert(data_->at(gpu_id).size() >= (start + ev_size));
    return ArrayView<emb_t>{data_->at(gpu_id).data() + start, ev_size};
  }
};

template <typename emb_t>
class RaggedGradBufferViewCPU :
  public std::iterator<std::random_access_iterator_tag, ArrayView<emb_t>>{
private:
  const std::vector<int> *ev_size_scan_list_;
  std::vector<emb_t> *grad_;

 public:
  using base = typename std::iterator<std::random_access_iterator_tag, ArrayView<emb_t>>;
  using value_type = typename base::value_type;
  using difference_type = typename base::difference_type;

  RaggedGradBufferViewCPU(
    const std::vector<int> *ev_size_scan_list,
    std::vector<emb_t> *grad
  )
      : ev_size_scan_list_(ev_size_scan_list),
        grad_(grad) {
  }

  value_type operator[](difference_type idx) {
    int ev_size = ev_size_scan_list_->at(idx + 1) - ev_size_scan_list_->at(idx);
    int start = ev_size_scan_list_->at(idx);

    assert((start + ev_size) <= grad_->size());
    return ArrayView<emb_t>{grad_->data() + start, ev_size};
  }
};

template <typename emb_t>
struct RaggedGradBufferCPU {
  std::vector<int> unique_id_space_list_;
  std::vector<int> unique_ev_size_list_;
  std::vector<emb_t> grad_;

  RaggedGradBufferCPU(int batch_size, const std::vector<int> &local_id_space_list, const std::vector<int> &local_ev_size_list, const std::vector<int> &local_hotness_list) {
    for (size_t i = 0; i < local_id_space_list.size(); ++i) {
      int id_space = local_id_space_list[i];
      if ((i == 0) || (id_space > unique_id_space_list_.back())) {
        unique_id_space_list_.push_back(id_space);
        unique_ev_size_list_.push_back(local_ev_size_list[i]);
      }
    }

    int max_num_elems = 0;
    for (size_t i = 0; i < local_id_space_list.size(); ++i) {
      max_num_elems += batch_size * local_hotness_list[i] * local_ev_size_list[i];
    }
    grad_.resize(max_num_elems);
  }
};

