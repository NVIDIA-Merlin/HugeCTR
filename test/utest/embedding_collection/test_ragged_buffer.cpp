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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "embedding_collection_cpu.hpp"


template <typename emb_t>
void test_ragged_model_buffer_cpu() {
  int num_gpus = 8;
  int batch_size = 16;
  std::vector<int> ev_size_list = {8, 16, 32, 8, 16, 8};
  
  RaggedModelBufferCPU<emb_t> ragged_model_buffer_cpu{num_gpus, batch_size, ev_size_list};

  int count = batch_size * ev_size_list.size();

  RaggedModelBufferViewCPU<emb_t> view_cpu {
    &ragged_model_buffer_cpu.data_,
    &ragged_model_buffer_cpu.local_ev_offset_list_,
    ragged_model_buffer_cpu.num_gpus_,
    ragged_model_buffer_cpu.batch_size_
  };
  
  for (int i = 0; i < count; ++i) {
    auto ev = view_cpu[i];
    std::cout << ev.size() << " ";
    for (int j = 0; j < ev.size(); ++j) {
      ev[j] = static_cast<emb_t>(i);
    }
  }
  std::cout << "\n";
  for (size_t i = 0; i < ragged_model_buffer_cpu.data_.size(); ++i) {
    for (size_t j = 0; j < ragged_model_buffer_cpu.data_[i].size(); ++j) {
      std::cout << ragged_model_buffer_cpu.data_[i][j] << " ";
    }
  }
  std::cout << "\n";
}

template <typename emb_t>
void test_ragged_network_buffer_cpu() {
  std::vector<std::vector<int>> global_embedding_list = {
    {0, 1, 2},
    {2, 3, 5},
    {1, 2},
    {3}
  };
  std::vector<int> ev_size_list = {8, 16, 32, 8, 16, 8};
  int batch_size = 12;

  RaggedNetworkBufferCPU<emb_t> ragged_network_buffer_cpu{batch_size, global_embedding_list, ev_size_list};

  int count = batch_size * 9 / global_embedding_list.size();

  RaggedNetworkBufferViewCPU<emb_t> view_cpu{
    &ragged_network_buffer_cpu.data_,
    &ragged_network_buffer_cpu.gpu_idx_offset_,
    &ragged_network_buffer_cpu.global_ev_offset_,
    ragged_network_buffer_cpu.num_gpus_,
    ragged_network_buffer_cpu.batch_size_
  };
  for (int i = 0; i < count; ++i) {
    auto ev = view_cpu[i];
    std::cout << ev.size() << " ";
    for (int j = 0; j < ev.size(); ++j) {
      ev[j] = static_cast<emb_t>(i);
    }
  }
  std::cout << "\n";
  for (int i = 0; i < count; ++i) {
    auto ev = view_cpu[i];
    std::cout << ev.size() << " ";
    for (int j = 0; j < ev.size(); ++j) {
      std::cout << ev[j] << " ";
    }
  }
}


template <typename emb_t>
void test_ragged_embedding_forward_result_cpu() {
  int batch_size_per_gpu = 10;
  int num_embedding = 5;
  std::vector<int> ev_size_list = {8, 16, 32, 8, 16};
  int ev_size_sum = std::accumulate(ev_size_list.begin(), ev_size_list.end(), 0);
  std::cout << "ev_size_sum:" << ev_size_sum << "\n";

  std::vector<int> ev_offset_list = {0};
  for (int i: ev_size_list) {
    ev_offset_list.push_back(i);
  }
  std::partial_sum(ev_offset_list.begin(), ev_offset_list.end(), ev_offset_list.begin());
  for (auto i: ev_offset_list) {
    std::cout << i << " ";
  }
  std::cout << "\n";
  std::vector<emb_t> forward_result;
  forward_result.resize(ev_size_sum * batch_size_per_gpu);

  RaggedEmbForwardResultViewCPU<emb_t> result_view_cpu{&forward_result, &ev_size_list, &ev_offset_list, batch_size_per_gpu};
  
  for (int i = 0; i < batch_size_per_gpu * num_embedding; ++i) {
    auto ev = result_view_cpu[i];
    std::cout << ev.size() << " ";
    for (int j = 0; j < ev.size(); ++j) {
      ev[j] = static_cast<emb_t>(i);
    }
  }

  for (size_t i = 0; i < forward_result.size(); ++i) {
    std::cout << forward_result[i] << " ";
  }
  
}

template <typename emb_t>
void test_ragged_grad_buffer_cpu() {
  std::vector<emb_t> grad;
  std::vector<int> offset_list = {0, 1, 3, 6, 10, 11};
  std::vector<int> ev_size_list = {8, 16, 32, 8, 16};
  
  std::vector<int> ev_size_scan_list{0};
  for (size_t idx = 0; idx < ev_size_list.size(); ++idx) {
    int start = offset_list[idx];
    int end = offset_list[idx + 1];
    for (int i = start; i < end; ++i) {
      ev_size_scan_list.push_back(ev_size_list[idx]);
    }
  }
  std::partial_sum(ev_size_scan_list.begin(), ev_size_scan_list.end(), ev_size_scan_list.begin());
  grad.resize(ev_size_scan_list.back());

  RaggedGradBufferViewCPU<emb_t> view_cpu{
    &ev_size_scan_list,
    &grad
  };
  
  for (int i = 0; i < offset_list.back(); ++i) {
    auto ev = view_cpu[i];
    std::cout << ev.size() << " ";
    for (int j = 0; j < ev.size(); ++j) {
      ev[j] = static_cast<emb_t>(i);
    }
  }
  std::cout << "\n";

  for (size_t i = 0; i < grad.size(); ++i) {
    std::cout << grad[i] << " ";
  }
  
}

TEST(test_ragged_buffer, test_ragged_model_buffer_cpu) {
  test_ragged_model_buffer_cpu<float>();
}

TEST(test_ragged_buffer, test_ragged_network_buffer_cpu) {
  test_ragged_network_buffer_cpu<float>();
}


TEST(test_ragged_buffer, test_ragged_embedding_forward_result_cpu) {
  test_ragged_embedding_forward_result_cpu<float>();
}


TEST(test_ragged_buffer, test_ragged_grad_buffer_cpu) {
  test_ragged_grad_buffer_cpu<float>();
}