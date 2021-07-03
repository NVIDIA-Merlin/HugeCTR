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
#include <deque>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
namespace HugeCTR {
template <typename T>
void convert_parquet_dense_columns(std::vector<T*>& dense_column_data_ptr,
                                   const int label_dense_dim, int batch_size, int num_dense_buffers,
                                   std::vector<rmm::device_buffer>& dense_data_buffers,
                                   int64_t* dev_ptr_staging,
                                   std::deque<rmm::device_buffer>& rmm_resources,
                                   rmm::mr::device_memory_resource* mr, cudaStream_t task_stream);

template <typename T>
size_t convert_parquet_cat_columns(std::vector<T*>& cat_column_data_ptr, int num_params,
                                   int param_id, int num_slots, int batch_size, int num_csr_buffers,
                                   int num_devices, bool distributed_slot, int pid_,
                                   const std::shared_ptr<ResourceManager> resource_manager,
                                   std::vector<rmm::device_buffer>& csr_value_buffers,
                                   std::vector<rmm::device_buffer>& csr_row_offset_buffers,
                                   int64_t* dev_ptr_staging, uint32_t* dev_embed_param_offset_buf,
                                   T* dev_slot_offset_ptr,
                                   std::deque<rmm::device_buffer>& rmm_resources,
                                   rmm::mr::device_memory_resource *mr,
                                   cudaStream_t task_stream);
}  // namespace HugeCTR
