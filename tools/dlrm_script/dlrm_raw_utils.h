/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#ifndef DLRM_RAW_UTILS_H_
#define DLRM_RAW_UTILS_H_

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include <HugeCTR/include/common.hpp>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <fstream>
#include <hash/concurrent_unordered_map.cuh>
#include <iostream>
#include <numeric>
#include <regex>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <string>
#include <vector>

static size_t process_read_bytes = 0;
static size_t process_write_bytes = 0;

namespace DLRM_RAW {

namespace cudf_io = cudf::io;
template <typename key, typename value>
using map_type = concurrent_unordered_map<key, value>;
using key_type = int32_t;
using value_type = int32_t;
using bitmask_type = cudf::bitmask_type;
using internal_runtime_error = HugeCTR::internal_runtime_error;
using Error_t = HugeCTR::Error_t;

typedef struct {
  int32_t begin_idx;
  int32_t end_idx;
} Index;

template <typename key, typename value>
__global__ void build_historgram(char *in_char, int *offsets, int num_strings,
                                 map_type<key, value> hist_map, uint32_t *global_idx_range) {
  __shared__ int smem_str_offsets[1024 + 1];
  int str_idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (str_idx < num_strings) smem_str_offsets[threadIdx.x] = offsets[str_idx];

  if (threadIdx.x == 0) {
    if ((str_idx + blockDim.x) <= num_strings)
      smem_str_offsets[blockDim.x] = offsets[str_idx + blockDim.x];
    else {
      int smem_idx = num_strings - str_idx;
      smem_str_offsets[smem_idx] = offsets[num_strings];
    }
  }

  __syncthreads();

  if (str_idx < num_strings) {
    int start = smem_str_offsets[threadIdx.x];
    int str_length = smem_str_offsets[threadIdx.x + 1] - start;

    uint32_t number = 0;
    // assuming dont need strict digit check for Criteo
    for (int k = 0; k < str_length; k++) {
      char x = in_char[start + k];
      int digit = 0;

      // 2-way divergence max
      if (x < 'a') {
        digit = x - '0';
      } else {
        digit = 10 + (x - 'a');
      }
      number = 16 * number + digit;
    }

    uint32_t capped_value = number;
    auto ht_pair = hist_map.insert(thrust::make_pair(capped_value, 1));

    if (ht_pair.second) {
      // ht_pair.first->second = atomicAdd(global_idx_range, 1);

      // increment unique count
      atomicAdd(global_idx_range, 1);
    } else {
      // didnt insert, check if iterator is not end()
      if (ht_pair.first != hist_map.end()) {
        // increment count
        atomicAdd(&(ht_pair.first->second), 1);
      } else {
        printf("insert and increment error");
      }
    }
  }
}

template <typename key, typename value>
__global__ void build_historgram_from_ints(key *in_col, bitmask_type *mask_ptr, int num_rows,
                                           map_type<key, value> hist_map,
                                           uint32_t *global_idx_range) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int bits_in_mask = sizeof(bitmask_type) * 8;  // entries per bitmask element
  if (idx < num_rows) {
    int data_bitmask_idx = idx / bits_in_mask;
    int data_bitmask_bit = idx % bits_in_mask;
    bitmask_type mask = mask_ptr[data_bitmask_idx];
    int32_t valid = (mask & (1 << data_bitmask_bit));
    bool valid_b = (valid != 0);
    key insert_val = 0;

    if (valid_b) insert_val = in_col[idx];

    auto ht_pair = hist_map.insert(thrust::make_pair(insert_val, 1));

    if (ht_pair.second) {
      // increment unique count
      atomicAdd(global_idx_range, 1);
    } else {
      // didnt insert, check if iterator is not end()
      if (ht_pair.first != hist_map.end()) {
        // increment count
        atomicAdd(&(ht_pair.first->second), 1);
      } else {
        printf("insert and increment error");
      }
    }
  }
}

template <typename key, typename value>
__global__ void build_categorical_index_from_ints(key *in_col, bitmask_type *mask_ptr, int num_rows,
                                                  map_type<key, value> hist_map, int32_t mod_idx,
                                                  uint32_t *global_idx_range) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int bits_in_mask = sizeof(bitmask_type) * 8;  // entries per bitmask element

  if (idx < num_rows) {
    int data_bitmask_idx = idx / bits_in_mask;
    int data_bitmask_bit = idx % bits_in_mask;
    bitmask_type mask = mask_ptr[data_bitmask_idx];
    int32_t valid = (mask & (1 << data_bitmask_bit));
    bool valid_b = (valid != 0);
    key in_val = 0;

    if (valid_b) in_val = in_col[idx];
    key capped_value = abs((int32_t)in_val) % mod_idx;
    auto ht_pair = hist_map.insert(thrust::make_pair(capped_value, 1));

    if (ht_pair.second) {
      ht_pair.first->second = atomicAdd(global_idx_range, 1);
    }
  }
}

template <typename key, typename value>
__global__ void cull_and_assign_idx(map_type<key, value> hist, uint32_t *idx_tracker,
                                    uint32_t *indices_removed, int32_t cutoff_count,
                                    size_t ht_size) {
  if (threadIdx.x == 0 && blockIdx.x == 0 && cutoff_count > 0) {
    idx_tracker[0] = 1;
  }
  __threadfence();

  int32_t loc = threadIdx.x + blockIdx.x * blockDim.x;

  if (loc < ht_size) {
    auto it = hist.data();
    it += loc;

    if ((it->second != hist.get_unused_element()) && (it->first != hist.get_unused_key())) {
      if (it->second <= cutoff_count) {
        // map to embedding idx 0
        it->second = 0;
        atomicAdd(indices_removed, 1);
      } else {
        // assign index
        it->second = atomicAdd(idx_tracker, 1);
      }
    }
  }
}

template <typename key, typename value>
__global__ void zero_drop_reset_idx(map_type<key, value> hist, size_t ht_size) {
  int32_t loc = threadIdx.x + blockIdx.x * blockDim.x;
  if (loc < ht_size) {
    auto it = hist.data();
    it += loc;

    if ((it->second != hist.get_unused_element()) && (it->first != hist.get_unused_key())) {
      it->second = it->second - 1;
    }
  }
}

template <typename key, typename value>
__global__ void build_categorical_index(char *in_char, int *offsets, int num_strings,
                                        map_type<key, value> hist_map, int32_t mod_idx,
                                        uint32_t *global_idx_range) {
  // blockDim.x == 1024
  __shared__ int smem_str_offsets[1024 + 1];
  int str_idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (str_idx < num_strings) smem_str_offsets[threadIdx.x] = offsets[str_idx];

  if (threadIdx.x == 0) {
    if ((str_idx + blockDim.x) <= num_strings)
      smem_str_offsets[blockDim.x] = offsets[str_idx + blockDim.x];
    else {
      int smem_idx = num_strings - str_idx;
      smem_str_offsets[smem_idx] = offsets[num_strings];
    }
  }

  __syncthreads();

  if (str_idx < num_strings) {
    int start = smem_str_offsets[threadIdx.x];
    int str_length = smem_str_offsets[threadIdx.x + 1] - start;

    // assuming dont need strict digit check for Criteo, convert hex to int
    uint32_t number = 0;
    for (int k = 0; k < str_length; k++) {
      char x = in_char[start + k];
      int digit = 0;

      // 2-way divergence max
      if (x < 'a') {  // then x is in [0, 9]
        digit = x - '0';
      } else {  // then x is in [a, f]
        digit = 10 + (x - 'a');
      }
      number = 16 * number + digit;
    }

    uint32_t capped_value = number % mod_idx;  // do mapping to this feature

    {  // process missing value
      if (0 == str_length) {
        // capped_value = 4294967295; // the maximum of uint32_t
        capped_value = mod_idx;  // missing mapped to mod_idx
      }
    }

    auto ht_pair = hist_map.insert(thrust::make_pair(capped_value, 1));

    if (ht_pair.second) {  // counting slot-size k
      ht_pair.first->second = atomicAdd(global_idx_range, 1);
    }
  }
}

#define SMEM_PITCH 40
template <typename key, typename value>
__global__ void process_data_rows(int64_t *int_array_local, int64_t *dev_int_array_nullmask,
                                  map_type<key, value> *hash_maps, int64_t *str_array,
                                  int64_t *str_offsets, int num_data_rows, int mod_idx,
                                  bool do_freq_encoding, int32_t *output,
                                  uint32_t *dev_slot_size_array, int64_t *dev_cat_col_nullmask) {
  extern __shared__ char smem_[];  // 512 * (SMEM_PITCH)
  int32_t *smem_output = reinterpret_cast<int32_t *>(smem_);
  int32_t *smem_str_offsets = reinterpret_cast<int32_t *>(smem_ + (SMEM_PITCH * 512 * 4));

  int num_ints = 14;
  int num_categoricals = 26;
  int total_features = num_ints + num_categoricals;

  int bits_in_mask = sizeof(bitmask_type) * 8;  // entries per bitmask element
  int start_idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (start_idx < num_data_rows) {
    int data_bitmask_idx = start_idx / bits_in_mask;
    int data_bitmask_bit = start_idx % bits_in_mask;

    // label and dense-features
    for (int i = 0; i < num_ints; i++) {
      int64_t addr = int_array_local[i];
      int32_t *fea_dev_ptr = reinterpret_cast<int32_t *>(addr);
      int64_t addr_mask = dev_int_array_nullmask[i];
      bitmask_type *fea_mask_ptr = reinterpret_cast<bitmask_type *>(addr_mask);
      bitmask_type mask = fea_mask_ptr[data_bitmask_idx];
      int32_t valid = (mask & (1 << data_bitmask_bit));
      bool valid_b = (valid != 0);

      if (valid_b) {
        smem_output[(threadIdx.x * SMEM_PITCH) + i] = fea_dev_ptr[start_idx];
      } else {  // dense missing value
        smem_output[(threadIdx.x * SMEM_PITCH) + i] = 0;
      }
    }
  }
  __syncthreads();

  // categorical features
  for (int i = 0; i < num_categoricals; i++) {
    int64_t addr = str_offsets[i];
    int32_t *offsets = reinterpret_cast<int32_t *>(addr);
    if (start_idx < num_data_rows) smem_str_offsets[threadIdx.x] = offsets[start_idx];

    if (threadIdx.x == 0) {
      if ((start_idx + blockDim.x) <= num_data_rows)
        smem_str_offsets[blockDim.x] = offsets[start_idx + blockDim.x];
      else {
        int smem_idx = num_data_rows - start_idx;
        smem_str_offsets[smem_idx] = offsets[num_data_rows];
      }
    }
    __syncthreads();
    if (start_idx < num_data_rows) {
      // int64_t cat_addr_mask = dev_cat_col_nullmask[i];
      // bitmask_type *cat_mask_ptr = reinterpret_cast<bitmask_type*>(cat_addr_mask);

      // bool null_column_b = false; // whether this column has missing value, true means has
      // missing value bool cat_valid_b = true; // whether this feature is valid, true is valid

      // if (cat_mask_ptr) {
      //   null_column_b = true;

      //   int cat_bitmask_idx = start_idx / bits_in_mask;
      //   int cat_bitmask_bit = start_idx % bits_in_mask;
      //   bitmask_type cat_mask = cat_mask_ptr[cat_bitmask_idx];
      //   int32_t cat_valid = (cat_mask & (1 << cat_bitmask_bit));
      //   cat_valid_b = (cat_valid != 0);
      // }

      // if (true == null_column_b && false == cat_valid_b) { // this one is missing value
      //   //  smem_output[(threadIdx.x * SMEM_PITCH) + i + num_ints] = 0;
      //   smem_output[(threadIdx.x * SMEM_PITCH) + i + num_ints] = (int32_t)(mod_idx);

      // }
      // else
      {  // this feature is valid
        int start = smem_str_offsets[threadIdx.x];
        int str_length = smem_str_offsets[threadIdx.x + 1] - start;

        // convert hex to int
        uint32_t number = 0;
        addr = str_array[i];
        char *in_char = reinterpret_cast<char *>(addr);
        for (int k = 0; k < str_length; k++) {
          char x = in_char[start + k];
          int digit = 0;

          // 2-way divergence max
          if (x < 'a') {
            digit = x - '0';
          } else {
            digit = 10 + (x - 'a');
          }
          number = 16 * number + digit;
        }
        uint32_t capped_value = number;

        if (!do_freq_encoding) capped_value = number % mod_idx;

        {  // process missing value
          if (0 == str_length) {
            // capped_value = 4294967295; // the maximum of uint32_t
            capped_value = mod_idx;  // missing value mapped to mod_idx
          }
        }

        auto hist_map = hash_maps[i];
        auto it = hist_map.find(capped_value);
        if (it != hist_map.end()) {
          smem_output[(threadIdx.x * SMEM_PITCH) + i + num_ints] =
              (int32_t)it->second;  // assumes idx wont go beyond int32
        } else {
          printf("error: %d-%d-%d", i, number, capped_value);
        }
      }
    }
  }

  __syncthreads();
  // start writing out from smem to output
  int block_start_idx = blockIdx.x * blockDim.x;
  int block_end_idx = block_start_idx + blockDim.x;
  int warp_id = threadIdx.x / warpSize;
  int lane_id = threadIdx.x % warpSize;
  int warp_per_block = blockDim.x / warpSize;

  // warp iterate over each
  int smem_row = warp_id;
  for (int idx = (block_start_idx + warp_id); idx < block_end_idx; idx += warp_per_block) {
    if (idx < num_data_rows) {
      for (int i = lane_id; i < total_features; i += warpSize) {
        output[idx * total_features + i] = smem_output[smem_row * SMEM_PITCH + i];
      }
    }
    smem_row += warp_per_block;
  }
}

// This function used to decide which row indices in this iteration should be write out. [begin_idx,
// end_idx) Total range is [save_rows_begin, save_rows_end), row index start from 0.
///@param save_rows_begin, Rows begin to save from source, -1 means the very beginning
///@param save_rows_end, Rows end to save from source, -1 means till the file ending.
///@param read_row_nums, currently, how many rows have been read in total.
///@param current_in_rows, currently, how many rows have been read in this iteration.
Index write_indices(const int32_t save_rows_begin, const int32_t save_rows_end,
                    const int32_t read_row_nums, const int32_t current_in_rows) {
  const int32_t save_rows_end_ =
      (-1 == save_rows_end) ? std::numeric_limits<int32_t>::max() : save_rows_end;
  if (save_rows_begin >= save_rows_end_) {
    std::cout << "save_rows_begin should be less than save_rows_end" << std::endl;
    exit(-1);
  }

  int32_t begin_idx = read_row_nums - current_in_rows;
  int32_t end_idx = read_row_nums;

  if ((save_rows_begin < end_idx) && (end_idx <= save_rows_end_)) {
    begin_idx = (begin_idx >= save_rows_begin) ? begin_idx : save_rows_begin;
  } else if ((begin_idx < save_rows_end_) && (save_rows_end_ < end_idx)) {
    end_idx = save_rows_end_;
  } else {  // -1, -1 means no writing
    begin_idx = -1;
    end_idx = -1;
  }

  Index out_idx{begin_idx, end_idx};
  return out_idx;
}

/// this function is used to ensure dense-feature >= 0
void process_dense_features(int32_t *host_out_buffer, int32_t rows_num) {
  // 1-label + 13-dense + 26-cate = 40
  for (int32_t row = 0; row < rows_num; ++row) {
    for (int32_t col = 0; col < 40; ++col) {
      int32_t gid = row * 40 + col;
      if (1 <= col && col < 14) {  // dense
        if (host_out_buffer[gid] < 0) host_out_buffer[gid] = 0;
      }
    }
  }
}

// this function is used to ensure cate-feature is [0, slot_size[j])
void process_cate_features(int32_t *host_out_buffer, int32_t rows_num,
                           std::vector<uint32_t> &slot_size_array) {
  for (int32_t row = 0; row < rows_num; ++row) {
    for (int32_t col = 0; col < 40; ++col) {
      int32_t gid = row * 40 + col;
      if (14 <= col) {  // cate-feature
        host_out_buffer[gid] = (host_out_buffer[gid] < 0) ? 0 : host_out_buffer[gid];
        host_out_buffer[gid] = host_out_buffer[gid] % slot_size_array[col - 14];
      }
    }
  }
}

__global__ void data_preprocess(int32_t *dev_out_buffer, int32_t rows_num,
                                uint32_t *slot_size_array, int32_t dense_bias = 0) {
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid < (rows_num * 40)) {  // num_numericals + num_categoricals = 40
    int col = gid % 40;
    if (1 <= col && col < 14) {  // dense features
      // dev_out_buffer[gid] = (dev_out_buffer[gid] >= 0) ? dev_out_buffer[gid] : 0;
      dev_out_buffer[gid] += dense_bias;  // the minimum in dense-feature is 3.0
    } else if (14 <= col) {               // categorical features
      dev_out_buffer[gid] = (dev_out_buffer[gid] < 0) ? 0 : dev_out_buffer[gid];
      dev_out_buffer[gid] = dev_out_buffer[gid] % (slot_size_array[col - 14] +
                                                   1);  // missing value mapped to old slot-size[j]
    }
  }
}

///@param save_rows_begin, Rows begin to save from source, -1 means the very beginning
///@param save_rows_end, Rows end to save from source, -1 means till the file ending.
template <typename key, typename value>
size_t convert_input_binaries(rmm::mr::device_memory_resource *mr, std::string input_file_path,
                              const std::vector<std::string> &column_dtypes,
                              const std::vector<std::string> &column_names, int32_t hash_bucket,
                              int max_chunk_per_file, size_t file_skip_bytes, bool do_freq_encoding,
                              int64_t *dev_int_col_ptrs, int64_t *dev_int_col_nullmask_ptrs,
                              int64_t *dev_cat_col_nullmask_ptrs,
                              int64_t *dev_categorical_col_hash_obj, int64_t *dev_char_ptrs,
                              int64_t *dev_offset_ptrs, int32_t *dev_out_buffer,
                              int32_t *host_out_buffer, std::ofstream *binary_writer,
                              uint32_t *dev_slot_size_array, int32_t save_rows_begin = -1,
                              int32_t save_rows_end = -1, int32_t dense_bias = 0) {
  const int num_numericals = 14;
  const int num_categoricals = 26;

  size_t sz_total_output_binary = 0;
  size_t sz_dev_int_col = num_numericals * sizeof(int64_t);
  size_t sz_dev_str_ptrs = num_categoricals * sizeof(int64_t);
  size_t maxbytes = 96 * 1024;

  std::ifstream binary_reader(input_file_path, std::ios::binary);
  binary_reader.seekg(0, std::ios::end);
  size_t file_size = binary_reader.tellg();
  binary_reader.close();

  size_t read_chunks = 128 * 1024 * 1024;
  cudf_io::csv_reader_options in_args =
      cudf_io::csv_reader_options::builder(cudf_io::source_info{input_file_path}).header(-1);
  // reader crashes without adding dtypes of data
  in_args.set_dtypes(column_dtypes);
  in_args.set_names(column_names);
  in_args.set_delimiter('\t');
  in_args.set_byte_range_size(read_chunks);  // how many bytes to read at one time.
  in_args.set_skipfooter(0);
  in_args.set_skiprows(0);
  in_args.set_byte_range_offset(file_skip_bytes);

  int loop_count = 0;
  int32_t read_row_nums = 0;  // already read how many rows

  while (true) {
    process_read_bytes += in_args.get_byte_range_size();
    auto tbl_w_metadata = cudf_io::read_csv(in_args, mr);
    int32_t num_rows = tbl_w_metadata.tbl->num_rows();
    read_row_nums += num_rows;

    // label and dense features
    std::vector<cudf::column *> col_logs;
    std::vector<const int32_t *> int_col_dev_ptrs;
    std::vector<const bitmask_type *> int_col_nullmask_dev_ptrs;
    for (int k = 0; k < num_numericals; k++) {
      col_logs.push_back(&(tbl_w_metadata.tbl->get_column(k)));
      int_col_dev_ptrs.push_back(col_logs[k]->view().data<int32_t>());

      int_col_nullmask_dev_ptrs.push_back(col_logs[k]->view().null_mask());
    }

    // future: const size init
    // categorical features
    std::vector<const char *> char_ptrs;
    std::vector<int *> offset_ptrs;
    std::vector<const bitmask_type *> cat_col_nullmask_ptrs;
    std::vector<std::pair<rmm::device_vector<char>, rmm::device_vector<cudf::size_type>>>
        str_offsets;
    for (int k = num_numericals; k < (num_numericals + num_categoricals); k++) {
      col_logs.push_back(&(tbl_w_metadata.tbl->get_column(k)));

      cat_col_nullmask_ptrs.push_back(col_logs[k]->view().null_mask());
    }

    for (int k = 0; k < num_categoricals; k++) {
      auto str_col_view = cudf::strings_column_view((col_logs[k + num_numericals]->view()));
      char_ptrs.push_back(const_cast<char *>(str_col_view.chars().data<char>()));
      offset_ptrs.push_back(const_cast<int32_t *>(str_col_view.offsets().data<int32_t>()));
    }

    CK_CUDA_THROW_(cudaMemset(dev_int_col_nullmask_ptrs, 0, sz_dev_int_col));
    CK_CUDA_THROW_(cudaMemset(dev_cat_col_nullmask_ptrs, 0, sz_dev_str_ptrs));

    CK_CUDA_THROW_(cudaMemcpy((void *)dev_int_col_ptrs, (void *)int_col_dev_ptrs.data(),
                              sz_dev_int_col, cudaMemcpyHostToDevice));
    CK_CUDA_THROW_(cudaMemcpy((void *)dev_int_col_nullmask_ptrs,
                              (void *)int_col_nullmask_dev_ptrs.data(), sz_dev_int_col,
                              cudaMemcpyHostToDevice));
    CK_CUDA_THROW_(cudaMemcpy((void *)dev_char_ptrs, (void *)char_ptrs.data(), sz_dev_str_ptrs,
                              cudaMemcpyHostToDevice));
    CK_CUDA_THROW_(cudaMemcpy((void *)dev_offset_ptrs, (void *)offset_ptrs.data(), sz_dev_str_ptrs,
                              cudaMemcpyHostToDevice));
    CK_CUDA_THROW_(cudaMemcpy((void *)dev_cat_col_nullmask_ptrs,
                              (void *)cat_col_nullmask_ptrs.data(), sz_dev_str_ptrs,
                              cudaMemcpyHostToDevice));

    dim3 block(512, 1, 1);
    dim3 grid((num_rows - 1) / block.x + 1, 1, 1);
    process_data_rows<key, value><<<grid, block, maxbytes>>>(
        dev_int_col_ptrs, dev_int_col_nullmask_ptrs,
        (map_type<key, value> *)dev_categorical_col_hash_obj, dev_char_ptrs, dev_offset_ptrs,
        num_rows, hash_bucket, do_freq_encoding, dev_out_buffer, dev_slot_size_array,
        dev_cat_col_nullmask_ptrs);

    size_t size_of_output_binary = num_rows * (num_numericals + num_categoricals) * sizeof(int32_t);

    CK_CUDA_THROW_(cudaDeviceSynchronize());

    {
      int block = 32;
      int grid = (num_rows * (num_numericals + num_categoricals) + block - 1) / block;
      data_preprocess<<<grid, block>>>(dev_out_buffer, num_rows, dev_slot_size_array, dense_bias);
      CK_CUDA_THROW_(cudaDeviceSynchronize());
    }

    CK_CUDA_THROW_(
        cudaMemcpy(host_out_buffer, dev_out_buffer, size_of_output_binary, cudaMemcpyDeviceToHost));

    if (binary_writer) {
      Index indices = write_indices(save_rows_begin, save_rows_end, read_row_nums, num_rows);
      if (-1 != indices.begin_idx && -1 != indices.end_idx) {
        int32_t offset_rows = indices.begin_idx - (read_row_nums - num_rows);
        int32_t offset_elems = offset_rows * (num_numericals + num_categoricals);
        int32_t write_rows = indices.end_idx - indices.begin_idx;

        if (write_rows <= 0) {
          ERROR_MESSAGE_("begin_idx = " + std::to_string(indices.begin_idx) +
                         ", end_idx = " + std::to_string(indices.end_idx) +
                         ", total rows now = " + std::to_string(read_row_nums));
          exit(-1);
        }

        size_of_output_binary = write_rows * (num_numericals + num_categoricals) * sizeof(int32_t);
        binary_writer->write((const char *)(host_out_buffer + offset_elems), size_of_output_binary);

        process_write_bytes += size_of_output_binary;
        sz_total_output_binary += size_of_output_binary;
      }
    }

    size_t new_byte_range_offset = in_args.get_byte_range_offset() + read_chunks;
    in_args.set_byte_range_offset(new_byte_range_offset);
    if (in_args.get_byte_range_offset() >= file_size) break;

    if ((in_args.get_byte_range_offset() + read_chunks) > file_size) {
      size_t new_byte_range_size = file_size - in_args.get_byte_range_offset();
      in_args.set_byte_range_size(new_byte_range_size);
    }
    loop_count++;

    if (loop_count == max_chunk_per_file) break;
  }

  return sz_total_output_binary;
}

std::vector<std::string> split_string(const std::string &text, const char *delimiters = ",") {
  std::regex pattern(delimiters);
  return std::vector<std::string>(std::sregex_token_iterator(text.begin(), text.end(), pattern, -1),
                                  std::sregex_token_iterator());
}

}  // namespace DLRM_RAW

} // namespace DLRM_RAW

#endif // DLRM_RAW_UTILS_H_
