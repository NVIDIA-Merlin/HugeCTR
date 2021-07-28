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

#include <cuda_runtime.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <iostream>
#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/statistics.hpp"
#include "HugeCTR/include/tensor2.hpp"
#include "HugeCTR/include/utils.cuh"

namespace HugeCTR {
namespace hybrid_embedding {

namespace statistics_kernels {

/** Compute keys to sort the frequent embedding tables.
 * The categories are organized:
 *  - per instance (round-robin)
 *  - then per slot
 *  - and finally in decreasing order of frequency
 *
 * The sort is stable, so the keys only need to be: instance_id * num_tables + table_id
 */
template <typename dtype>
static __global__ void category_to_frequent_section(const dtype *__restrict__ categories_sorted,
                                                    uint32_t *keys,
                                                    const dtype *__restrict__ table_offsets,
                                                    size_t num_frequent, size_t num_tables,
                                                    size_t num_instances) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < num_frequent) {
    dtype category = categories_sorted[tid];

    uint32_t table_id = 0;
    for (table_id = 0; table_id < num_tables - 1 && category >= table_offsets[table_id + 1];
         ++table_id) {
    }

    uint32_t instance_id = tid % num_instances;

    keys[tid] = instance_id * num_tables + table_id;
  }
}

template <typename T, typename IdxT>
static __global__ void fill(T *__restrict__ array, T val, IdxT n_elem) {
  IdxT tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n_elem) array[tid] = val;
}

template <typename dtype>
static __global__ void calculate_category_frequent_index(
    const dtype *__restrict__ frequent_categories, dtype *category_frequent_index,
    size_t num_frequent) {
  dtype tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_frequent) {
    category_frequent_index[frequent_categories[tid]] = tid;
  }
}

template <typename dtype>
static __global__ void calculate_category_location(const dtype *__restrict__ infrequent_categories,
                                                   dtype *category_location, dtype num_infrequent,
                                                   size_t num_models) {
  dtype tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_infrequent) {
    dtype category = infrequent_categories[tid];
    category_location[2 * category] = tid % num_models;
    category_location[2 * category + 1] = tid / num_models;
  }
}

template <typename dtype>
static __global__ void calculate_infrequent_model_table_offsets(
    const dtype *__restrict__ categories, const dtype *__restrict__ category_location,
    const dtype *__restrict__ table_offsets, dtype *offsets, size_t n_tables, dtype n_elem,
    dtype n_model_elem, uint32_t global_instance_id) {
  const size_t table_id = threadIdx.x;

  // Find first category id belonging to that table (not necessarily in this model!)
  dtype category = table_offsets[table_id];

  // Step 1: binary search of the category
  dtype start = 0;
  dtype end = n_elem;
  while (start < end) {
    dtype mid = (start + end) / 2;
    dtype value = categories[mid];

    if (value < category)
      start = mid + 1;
    else
      end = mid;
  }

  // Step 2: increment until the model id matches
  while (start < n_elem && category_location[2 * categories[start]] != global_instance_id) {
    start++;
  }

  // Step 3: lookup location and write the offset
  if (start == n_elem) {
    // If we are at the end of the array, write the number of elements belonging to this model
    offsets[table_id] = n_model_elem;
  } else {
    // Else, write the location of the first category from this table belonging to this model
    offsets[table_id] = category_location[2 * categories[start] + 1];
  }
}

template <typename dtype>
static __global__ void calculate_frequent_model_table_offsets(
    const dtype *__restrict__ categories, const dtype *__restrict__ table_offsets, dtype *offsets,
    size_t n_divs, size_t n_tables, dtype n_elem) {
  const size_t div_id = blockIdx.x;
  const size_t table_id = threadIdx.x;

  const dtype n_elem_per_div = n_elem / n_divs;  // Note: num_instances divides num_frequent

  // Find first category id belonging to that table
  dtype category = table_offsets[table_id];

  // Setup start and end to the bounds of this division
  dtype start = div_id * n_elem_per_div;
  dtype end = (div_id + 1) * n_elem_per_div;

  // Binary search
  while (start < end) {
    dtype mid = (start + end) / 2;
    dtype value = categories[mid];

    if (value < category)
      start = mid + 1;
    else
      end = mid;
  }

  // Write offset
  offsets[div_id * (n_tables + 1) + table_id] = start;
}

}  // namespace statistics_kernels

///
/// Perform count of categories within the samples and sort the categories by count
///
template <typename dtype>
void Statistics<dtype>::sort_categories_by_count(const Tensor2<dtype> &samples,
                                                 cudaStream_t stream) {
  const dtype *d_samples = samples.get_ptr();
  size_t num_samples = samples.get_size_in_bytes() / sizeof(dtype);
  dtype *d_categories = categories_sorted.get_ptr();
  uint32_t *d_counts = counts_sorted.get_ptr();
  sort_categories_by_count(d_samples, num_samples, d_categories, d_counts, num_unique_categories,
                           stream);  // Kefengs' function
  categories_sorted.reset_shape({num_unique_categories, 1});
  counts_sorted.reset_shape({num_unique_categories, 1});
}

template <typename dtype>
struct InfrequentSelectOp {
  const dtype *category_frequent_index;
  const dtype num_categories;
  __host__ __device__ __forceinline__ InfrequentSelectOp(const dtype *category_frequent_index,
                                                         const dtype num_categories)
      : category_frequent_index(category_frequent_index), num_categories(num_categories) {}
  __device__ __forceinline__ bool operator()(const dtype &category) const {
    return category_frequent_index[category] == num_categories;
  }
};

template <typename dtype>
void Statistics<dtype>::reserve_temp_storage(std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf) {
  size_t size_sort_keys_temp = 0;
  sort_categories_by_count_temp_storages_.resize(7);
  CK_CUDA_THROW_(cub::DeviceRadixSort::SortKeys((void *)nullptr, size_sort_keys_temp,
                                                (dtype *)nullptr, (dtype *)nullptr,
                                                (int)num_samples, 0, sizeof(dtype) * 8, 0));
  buf->reserve({size_sort_keys_temp, 1}, &sort_categories_by_count_temp_storages_[0]);
  buf->reserve({num_samples * sizeof(dtype), 1}, &sort_categories_by_count_temp_storages_[1]);
  size_t size_unique_categories_temp = 0;
  CK_CUDA_THROW_(cub::DeviceRunLengthEncode::Encode(
      (void *)nullptr, size_unique_categories_temp, (dtype *)nullptr, (dtype *)nullptr,
      (uint32_t *)nullptr, (uint32_t *)nullptr, (int)num_samples, 0));

  buf->reserve({size_unique_categories_temp, 1}, &sort_categories_by_count_temp_storages_[2]);
  buf->reserve({num_samples * sizeof(dtype), 1}, &sort_categories_by_count_temp_storages_[3]);
  buf->reserve({num_samples * sizeof(uint32_t), 1}, &sort_categories_by_count_temp_storages_[4]);
  buf->reserve({sizeof(uint32_t), 1}, &sort_categories_by_count_temp_storages_[5]);

  size_t size_sort_pairs_temp = 0;
  CK_CUDA_THROW_(cub::DeviceRadixSort::SortPairsDescending(
      (void *)nullptr, size_sort_pairs_temp, (uint32_t *)nullptr, (uint32_t *)nullptr,
      (dtype *)nullptr, (dtype *)nullptr, (int)num_samples, 0, sizeof(uint32_t) * 8, 0));
  buf->reserve({size_sort_pairs_temp, 1}, &sort_categories_by_count_temp_storages_[6]);

  /// TODO: reuse temp storage for operations that can't run concurrently!

  calculate_frequent_categories_temp_storages_.resize(3);
  size_t size_sort_temp = 0;
  int bit_width = 1;
  for (uint32_t i = num_instances * num_tables - 1; i >>= 1;) bit_width++;
  CK_CUDA_THROW_(cub::DeviceRadixSort::SortPairs(
      (void *)nullptr, size_sort_temp, (uint32_t *)nullptr, (uint32_t *)nullptr, (dtype *)nullptr,
      (dtype *)nullptr, (int)num_samples, 0, bit_width, 0));

  buf->reserve({num_samples * sizeof(uint32_t), 1},
               &calculate_frequent_categories_temp_storages_[0]);
  buf->reserve({num_samples * sizeof(uint32_t), 1},
               &calculate_frequent_categories_temp_storages_[1]);
  buf->reserve({size_sort_temp, 1}, &calculate_frequent_categories_temp_storages_[2]);

  calculate_infrequent_categories_temp_storages_.resize(2);
  size_t size_select_temp = 0;
  cub::CountingInputIterator<dtype> counting(0);
  InfrequentSelectOp<dtype> select_op(nullptr, 0);
  CK_CUDA_THROW_(cub::DeviceSelect::If((void *)nullptr, size_select_temp, counting,
                                       (dtype *)nullptr, (dtype *)nullptr, num_categories,
                                       select_op, 0));
  buf->reserve({size_select_temp, 1}, &calculate_infrequent_categories_temp_storages_[0]);
  buf->reserve({sizeof(dtype), 1}, &calculate_infrequent_categories_temp_storages_[1]);
};

template <typename dtype>
void Statistics<dtype>::sort_categories_by_count(const dtype *samples, size_t num_samples,
                                                 dtype *categories_sorted, uint32_t *counts_sorted,
                                                 uint32_t &num_unique_categories,
                                                 cudaStream_t stream) {
  if (num_samples > 0x7fffffff) {
    std::cout << "Num samples: " << std::hex << num_samples << std::dec << std::endl;
    CK_THROW_(Error_t::WrongInput, "num_samples is too large, overflow for int type");
  }
  void *p_sort_keys_temp =
      reinterpret_cast<void *>(sort_categories_by_count_temp_storages_[0].get_ptr());  // void*
  dtype *p_sort_keys_out =
      reinterpret_cast<dtype *>(sort_categories_by_count_temp_storages_[1].get_ptr());  // dtype*
  void *p_unique_categories_temp =
      reinterpret_cast<void *>(sort_categories_by_count_temp_storages_[2].get_ptr());  // void*
  dtype *p_unique_categories_out =
      reinterpret_cast<dtype *>(sort_categories_by_count_temp_storages_[3].get_ptr());  // dtype*
  uint32_t *p_unique_categories_counts = reinterpret_cast<uint32_t *>(
      sort_categories_by_count_temp_storages_[4].get_ptr());  // uint32_t*
  uint32_t *p_num_unique_categories = reinterpret_cast<uint32_t *>(
      sort_categories_by_count_temp_storages_[5].get_ptr());  // uint32*
  void *p_sort_pairs_temp =
      reinterpret_cast<void *>(sort_categories_by_count_temp_storages_[6].get_ptr());  // void*

  size_t temp_size = sort_categories_by_count_temp_storages_[0].get_size_in_bytes();
  CK_CUDA_THROW_(cub::DeviceRadixSort::SortKeys(p_sort_keys_temp, temp_size, samples,
                                                p_sort_keys_out, (int)num_samples, 0,
                                                sizeof(dtype) * 8, stream));

  temp_size = sort_categories_by_count_temp_storages_[2].get_size_in_bytes();
  CK_CUDA_THROW_(cub::DeviceRunLengthEncode::Encode(
      p_unique_categories_temp, temp_size, p_sort_keys_out, p_unique_categories_out,
      p_unique_categories_counts, p_num_unique_categories, (int)num_samples, stream));
  CK_CUDA_THROW_(cudaMemcpyAsync((void *)&num_unique_categories, (void *)p_num_unique_categories,
                                 sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  CK_CUDA_THROW_(cudaPeekAtLastError());

  temp_size = sort_categories_by_count_temp_storages_[6].get_size_in_bytes();
  CK_CUDA_THROW_(cub::DeviceRadixSort::SortPairsDescending(
      p_sort_pairs_temp, temp_size, p_unique_categories_counts, counts_sorted,
      p_unique_categories_out, categories_sorted, (int)num_unique_categories, 0,
      sizeof(uint32_t) * 8, stream));
}

template <typename dtype>
void Statistics<dtype>::calculate_frequent_categories(dtype *frequent_categories,
                                                      dtype *category_frequent_index,
                                                      const size_t num_frequent,
                                                      cudaStream_t stream) {
  uint32_t *p_keys_in = reinterpret_cast<uint32_t *>(
      calculate_frequent_categories_temp_storages_[0].get_ptr());  // uint32_t*
  uint32_t *p_keys_out = reinterpret_cast<uint32_t *>(
      calculate_frequent_categories_temp_storages_[1].get_ptr());  // uint32_t*
  void *p_sort_temp =
      reinterpret_cast<void *>(calculate_frequent_categories_temp_storages_[2].get_ptr());  // void*
  size_t sort_temp_size = calculate_frequent_categories_temp_storages_[2].get_size_in_bytes();

  if (num_frequent > 0) {
    /* Step 1: generate keys (table "sections") */
    constexpr size_t TPB_keys = 256;
    const size_t n_blocks_keys = ceildiv<size_t>(num_frequent, TPB_keys);
    statistics_kernels::category_to_frequent_section<<<n_blocks_keys, TPB_keys, 0, stream>>>(
        categories_sorted.get_ptr(), p_keys_in, table_offsets.get_ptr(), num_frequent, num_tables,
        num_instances);
    CK_CUDA_THROW_(cudaPeekAtLastError());

    /* Step 2: sort */
    int bit_width = 1;
    for (uint32_t i = num_instances * num_tables - 1; i >>= 1;) bit_width++;
    CK_CUDA_THROW_(cub::DeviceRadixSort::SortPairs(
        p_sort_temp, sort_temp_size, p_keys_in, p_keys_out, categories_sorted.get_ptr(),
        frequent_categories, (int)num_frequent, 0, bit_width, stream));
  }

  /* Step 3: construct category_frequent_index */
  constexpr size_t TPB_fill = 256;
  const size_t n_blocks_fill = ceildiv<size_t>(num_categories, TPB_fill);
  statistics_kernels::fill<<<n_blocks_fill, TPB_fill, 0, stream>>>(
      category_frequent_index, (dtype)num_categories, num_categories);
  CK_CUDA_THROW_(cudaPeekAtLastError());

  if (num_frequent > 0) {
    constexpr size_t TPB_inversion = 256;
    const size_t n_blocks_inversion = ceildiv<size_t>(num_frequent, TPB_inversion);
    statistics_kernels::
        calculate_category_frequent_index<<<n_blocks_inversion, TPB_inversion, 0, stream>>>(
            frequent_categories, category_frequent_index, num_frequent);
    CK_CUDA_THROW_(cudaPeekAtLastError());
  }
}

template <typename dtype>
void Statistics<dtype>::calculate_infrequent_categories(dtype *infrequent_categories,
                                                        const dtype *category_frequent_index,
                                                        dtype *category_location,
                                                        const dtype num_infrequent,
                                                        cudaStream_t stream) {
  void *p_select_temp = reinterpret_cast<void *>(
      calculate_infrequent_categories_temp_storages_[0].get_ptr());  // void*
  dtype *p_num_selected = reinterpret_cast<dtype *>(
      calculate_infrequent_categories_temp_storages_[1].get_ptr());  // dtype*
  size_t select_temp_size = calculate_infrequent_categories_temp_storages_[0].get_size_in_bytes();

  /* Fill with default value */
  constexpr size_t TPB_fill = 256;
  const size_t n_blocks_fill = ceildiv<size_t>(2 * num_categories, TPB_fill);
  statistics_kernels::fill<<<n_blocks_fill, TPB_fill, 0, stream>>>(
      category_location, (dtype)num_categories, 2 * num_categories);
  CK_CUDA_THROW_(cudaPeekAtLastError());

  /// TODO: combine select and writing to category_location with a custom output iterator

  /* Select the infrequent categories */
  cub::CountingInputIterator<dtype> counting(0);
  InfrequentSelectOp<dtype> select_op(category_frequent_index, num_categories);
  CK_CUDA_THROW_(cub::DeviceSelect::If(p_select_temp, select_temp_size, counting,
                                       infrequent_categories, p_num_selected, num_categories,
                                       select_op, stream));

  /* Write to category_location */
  if (num_infrequent > 0) {
    constexpr size_t TPB_loc = 256;
    const size_t n_blocks_loc = (size_t)ceildiv<dtype>(num_infrequent, TPB_loc);
    statistics_kernels::calculate_category_location<<<n_blocks_loc, TPB_loc, 0, stream>>>(
        infrequent_categories, category_location, num_infrequent, num_instances);
    CK_CUDA_THROW_(cudaPeekAtLastError());
  }
}

template <typename dtype>
void Statistics<dtype>::calculate_infrequent_model_table_offsets(
    std::vector<dtype> &h_infrequent_model_table_offsets,
    const Tensor2<dtype> &infrequent_categories, const Tensor2<dtype> &category_location,
    uint32_t global_instance_id, const dtype num_infrequent, cudaStream_t stream) {
  dtype num_model_infrequent = num_infrequent / num_instances +
                               (global_instance_id < num_infrequent % num_instances ? 1 : 0);

  statistics_kernels::calculate_infrequent_model_table_offsets<<<1, num_tables + 1, 0, stream>>>(
      infrequent_categories.get_ptr(), category_location.get_ptr(), table_offsets.get_ptr(),
      infrequent_model_table_offsets.get_ptr(), num_tables, num_infrequent, num_model_infrequent,
      global_instance_id);
  CK_CUDA_THROW_(cudaPeekAtLastError());

  h_infrequent_model_table_offsets.resize(num_tables + 1);
  CK_CUDA_THROW_(cudaMemcpyAsync(h_infrequent_model_table_offsets.data(),
                                 infrequent_model_table_offsets.get_ptr(),
                                 (num_tables + 1) * sizeof(dtype), cudaMemcpyDeviceToHost, stream));
}

template <typename dtype>
void Statistics<dtype>::calculate_frequent_model_table_offsets(
    std::vector<dtype> &h_frequent_model_table_offsets, const Tensor2<dtype> &frequent_categories,
    const dtype num_frequent, cudaStream_t stream) {
  statistics_kernels::
      calculate_frequent_model_table_offsets<<<num_instances, num_tables + 1, 0, stream>>>(
          frequent_categories.get_ptr(), table_offsets.get_ptr(),
          frequent_model_table_offsets.get_ptr(), num_instances, num_tables, num_frequent);
  CK_CUDA_THROW_(cudaPeekAtLastError());

  h_frequent_model_table_offsets.resize(num_instances * (num_tables + 1));
  CK_CUDA_THROW_(cudaMemcpyAsync(
      h_frequent_model_table_offsets.data(), frequent_model_table_offsets.get_ptr(),
      num_instances * (num_tables + 1) * sizeof(dtype), cudaMemcpyDeviceToHost, stream));
}

template class Statistics<uint32_t>;
template class Statistics<long long>;
template class Statistics<unsigned long>;
}  // namespace hybrid_embedding

}  // namespace HugeCTR
