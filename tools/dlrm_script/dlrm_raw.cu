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

#include <cstring>

#include "dlrm_raw_utils.h"
using namespace DLRM_RAW;
#include <iostream>

void process_kaggle_dataset(const std::string &input_dir_path, const std::string &output_dir_path,
                            const int num_numericals, const int num_categoricals) {
  int max_chunk_per_file = 10000;  // loop count, in a signle binary data, store how many chunks

  bool process_output = true;
  bool write_out = true;

  // int32_t hash_bucket = 40000000;  // mod-idx
  // int max_cat_fea_cardi = 40000000;  // 40M
  // int avg_cat_fea_cardi = 1000000;    // 1M
  // int min_cat_fea_cardi = 1000000;    // 1M
  // std::vector<int32_t> hist_sizes = {max_cat_fea_cardi, avg_cat_fea_cardi, avg_cat_fea_cardi,
  // avg_cat_fea_cardi, avg_cat_fea_cardi,
  //                                    min_cat_fea_cardi, min_cat_fea_cardi, min_cat_fea_cardi,
  //                                    min_cat_fea_cardi, max_cat_fea_cardi, max_cat_fea_cardi,
  //                                    avg_cat_fea_cardi, min_cat_fea_cardi, min_cat_fea_cardi,
  //                                    avg_cat_fea_cardi, min_cat_fea_cardi, min_cat_fea_cardi,
  //                                    min_cat_fea_cardi, min_cat_fea_cardi, max_cat_fea_cardi,
  //                                    max_cat_fea_cardi, max_cat_fea_cardi, avg_cat_fea_cardi,
  //                                    min_cat_fea_cardi, min_cat_fea_cardi, min_cat_fea_cardi}; //
  //                                    mod-idx

  int min_cat_fea_cardi = 10000000;  // 10M
  int32_t hash_bucket = min_cat_fea_cardi;
  std::vector<int32_t> hist_sizes = {
      hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket,
      hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket,
      hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket,
      hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket};  // mod-idx

  size_t pool_alloc_size = (size_t)4 * 1024 * 1024 * 1024;  // 4 GB
  // std::vector<int> dev = {0};
  rmm::mr::device_memory_resource *base_mr = new rmm::mr::cuda_memory_resource();
  auto *p_mr =
      new rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>(base_mr, pool_alloc_size);
  rmm::mr::set_current_device_resource(p_mr);

  std::vector<std::string> column_dtypes;                 // dtypes of label, dense, categorical
  std::vector<std::string> column_names;                  // names of label, dense, categorical
  std::vector<std::string> cat_column_names;              // names of categorical
  std::map<std::string, int32_t> column_name_to_col_idx;  // <col-name, idx>
  std::unordered_map<std::string, map_type<key_type, value_type> *>
      categorical_col_hash_tables;  // <name, <key, value>>

  // label
  column_dtypes.push_back("int32");
  column_names.push_back("label");
  column_name_to_col_idx.insert(std::make_pair("label", 0));

  // dense-features
  for (int k = 1; k <= 13; k++) {
    column_dtypes.push_back("int32");
    std::string name = "I" + std::to_string(k);
    column_names.push_back(name);
    column_name_to_col_idx.insert(std::make_pair(name, k));
  }

  // categorical-features
  for (int k = 1; k <= num_categoricals; k++) {
    column_dtypes.push_back("str");
    std::string name = "C" + std::to_string(k);
    column_names.push_back(name);
    cat_column_names.push_back(name);
    column_name_to_col_idx.insert(std::make_pair(name, k + num_numericals - 1));

    auto cuda_map_obj =
        map_type<key_type, value_type>::create(compute_hash_table_size(hist_sizes[k - 1]))
            .release();
    ;
    categorical_col_hash_tables.insert(std::make_pair(name, cuda_map_obj));
  }

  int current_device = 0;
  cudaDeviceProp prop;
  CK_CUDA_THROW_(cudaGetDeviceProperties(&prop, current_device));

  size_t read_chunks = 128 * 1024 * 1024;  // read 128MB at one time

  uint32_t *accum_location = nullptr;                // slot-size
  CK_CUDA_THROW_(cudaMalloc(&accum_location, 128));  // 128 Bytes = 32 * uint32_t
  CK_CUDA_THROW_(cudaMemset(accum_location, 0, 128));

  // uint32_t *culled_index_count = nullptr;
  // CK_CUDA_THROW_(cudaMalloc(&culled_index_count, 128)); // 128 Bytes = 32 * uint32_t

  size_t total_file_bytes_read = 0;
  const auto time_map_start = std::chrono::high_resolution_clock::now();

  // get file size, hard-coded filename
  std::string input_file_name = std::string(input_dir_path + "/train.txt");
  std::ifstream binary_reader(input_file_name, std::ios::binary);
  binary_reader.seekg(0, std::ios::end);
  size_t file_size = binary_reader.tellg();
  binary_reader.close();

  // csv arguments,
  // https://docs.rapids.ai/api/libcudf/stable/structcudf_1_1io_1_1read__csv__args.html
  cudf_io::csv_reader_options in_args =
      cudf_io::csv_reader_options::builder(cudf_io::source_info{input_file_name}).header(-1);
  in_args.set_dtypes(column_dtypes);
  in_args.set_names(column_names);
  in_args.set_delimiter('\t');
  in_args.set_byte_range_size(read_chunks);  // how many bytes to read at one time.
  in_args.set_skipfooter(0);
  in_args.set_skiprows(0);
  in_args.set_use_cols_names(cat_column_names);

  int32_t total_row_nums = 0;

  int loop_count = 0;
  while (true) {
    total_file_bytes_read += in_args.get_byte_range_size();
    cudf_io::table_with_metadata tbl_w_metadata = cudf_io::read_csv(in_args, p_mr);
    total_row_nums += tbl_w_metadata.tbl->num_rows();

    dim3 block(prop.maxThreadsPerBlock, 1, 1);
    dim3 grid((tbl_w_metadata.tbl->num_rows() - 1) / block.x + 1, 1, 1);

    // categorical-features
    for (unsigned int k = 0; k < cat_column_names.size(); ++k) {
      auto col = std::move(tbl_w_metadata.tbl->get_column(k));
      if (col.type().id() == cudf::type_id::STRING) {
        auto str_col = cudf::strings_column_view(col.view());
        int64_t num_strings = str_col.size();
        char *char_array = const_cast<char *>(str_col.chars().data<char>());
        int32_t *offsets = const_cast<int32_t *>(str_col.offsets().data<int32_t>());

        build_categorical_index<key_type, value_type><<<grid, block>>>(
            char_array, offsets, num_strings,
            // *categorical_col_hash_tables[cat_column_names[k]], hash_bucket, &accum_location[k]);
            *categorical_col_hash_tables[cat_column_names[k]], hist_sizes[k], &accum_location[k]);

      } else if (col.type().id() == cudf::type_id::INT32) {
        key_type *data = const_cast<key_type *>(col.view().data<key_type>());
        bitmask_type *in_mask = const_cast<bitmask_type *>(col.view().null_mask());

        build_categorical_index_from_ints<key_type, value_type><<<grid, block>>>(
            data, in_mask, tbl_w_metadata.tbl->num_rows(),
            // *categorical_col_hash_tables[cat_column_names[k]], hash_bucket, &accum_location[k]);
            *categorical_col_hash_tables[cat_column_names[k]], hist_sizes[k], &accum_location[k]);

      } else {
        ERROR_MESSAGE_("col.type().id() != [STRING, INT32]");
      }
    }

    size_t new_byte_range_offset = in_args.get_byte_range_offset() + read_chunks;
    in_args.set_byte_range_offset(new_byte_range_offset);
    if (in_args.get_byte_range_offset() >= file_size) break;

    if ((in_args.get_byte_range_offset() + read_chunks) > file_size) {
      size_t new_byte_range_size = file_size - in_args.get_byte_range_offset();
      in_args.set_byte_range_size(new_byte_range_size);
    }

    ++loop_count;

    if (loop_count == max_chunk_per_file) break;
  }
  MESSAGE_(input_file_name + "'s total rows number = " + std::to_string(total_row_nums));

  // show: slot size array
  std::vector<uint32_t> host_sz_per_fea(num_categoricals);
  CK_CUDA_THROW_(cudaMemcpy(host_sz_per_fea.data(), accum_location,
                            num_categoricals * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  MESSAGE_("Slot size array in " + input_file_name + ", missing value mapped to unused key: ");
  for (auto c : host_sz_per_fea) std::cout << (c) << ", ";
  std::cout << "\b\b" << std::endl;

  const auto time_map_stop = std::chrono::high_resolution_clock::now();
  const auto time_map_build =
      std::chrono::duration_cast<std::chrono::milliseconds>(time_map_stop - time_map_start);
  MESSAGE_("Time used to build map: " + std::to_string(time_map_build.count()) + " milliseconds.");

  double read_bw = double(total_file_bytes_read) / (1024.0 * 1024.0 * 1024.0);
  read_bw = (read_bw / time_map_build.count()) * 1000.f;
  MESSAGE_("Total bytes read: " + std::to_string(total_file_bytes_read) +
           " Effective Read B/W: " + std::to_string(read_bw) + " GB/s.");

  // CK_CUDA_THROW_(cudaFree(culled_index_count));
  CK_CUDA_THROW_(cudaFree(accum_location));

  // starting to do the convertion
  if (process_output) {
    uint32_t *dev_slot_size_array = nullptr;
    size_t slot_size_array_size = num_categoricals * sizeof(uint32_t);
    CK_CUDA_THROW_(cudaMalloc(&dev_slot_size_array, slot_size_array_size));
    CK_CUDA_THROW_(cudaMemcpy(dev_slot_size_array, host_sz_per_fea.data(), slot_size_array_size,
                              cudaMemcpyHostToDevice));

    int32_t *dev_out_buffer = nullptr;
    int32_t *host_out_buffer = nullptr;

    size_t sz_output_buffer = 128 * 1024 * 1024;  // 128 MB, = read_chunks
    CK_CUDA_THROW_(cudaMalloc(&dev_out_buffer, sz_output_buffer));
    CK_CUDA_THROW_(cudaMallocHost(&host_out_buffer, sz_output_buffer));

    int64_t *dev_int_col_ptrs = nullptr;
    int64_t *dev_int_col_nullmask_ptrs = nullptr;
    int64_t *dev_cat_col_nullmask_ptrs = nullptr;
    int64_t *dev_categorical_col_hash_obj = nullptr;
    int64_t *dev_char_ptrs = nullptr;
    int64_t *dev_offset_ptrs = nullptr;

    size_t sz_dev_int_col = num_numericals * sizeof(int64_t);
    size_t sz_dev_cat_hash_obj = num_categoricals * sizeof(map_type<key_type, value_type>);
    size_t sz_dev_str_ptrs = num_categoricals * sizeof(int64_t);

    CK_CUDA_THROW_(cudaMalloc(&dev_int_col_ptrs, sz_dev_int_col));
    CK_CUDA_THROW_(cudaMalloc(&dev_int_col_nullmask_ptrs, sz_dev_int_col));
    CK_CUDA_THROW_(cudaMalloc(&dev_cat_col_nullmask_ptrs, sz_dev_str_ptrs));
    CK_CUDA_THROW_(cudaMalloc(&dev_categorical_col_hash_obj, sz_dev_cat_hash_obj));
    CK_CUDA_THROW_(cudaMalloc(&dev_char_ptrs, sz_dev_str_ptrs));
    CK_CUDA_THROW_(cudaMalloc(&dev_offset_ptrs, sz_dev_str_ptrs));

    // encode and write out binary
    int maxbytes = 96 * 1024;  // dynamic shared memory size 96 KB
    cudaFuncSetAttribute(process_data_rows<key_type, value_type>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

    std::vector<map_type<key_type, value_type>> categorical_col_hash_obj;
    for (auto c : cat_column_names) {
      categorical_col_hash_obj.push_back(*categorical_col_hash_tables[c]);
    }

    CK_CUDA_THROW_(cudaMemcpy((void *)dev_categorical_col_hash_obj,
                              (void *)categorical_col_hash_obj.data(), sz_dev_cat_hash_obj,
                              cudaMemcpyHostToDevice));

    if (process_output) {
      std::ofstream *binary_writer = nullptr;

      if (write_out)
        binary_writer =
            new std::ofstream(std::string(output_dir_path + "/train_data.bin"), std::ios::binary);
      size_t sz_total_output_binary = 0;

      const auto time_convert_start = std::chrono::high_resolution_clock::now();

      // train_data.bin
      {
        int32_t rows_begin_train = 0, rows_end_train = 36672493;  // train.txt [:36672493)
        std::string input_file_path = std::string(input_dir_path + "/train.txt");
        sz_total_output_binary = convert_input_binaries<key_type, value_type>(
            p_mr, input_file_path, column_dtypes, column_names, hash_bucket, max_chunk_per_file, 0,
            false, dev_int_col_ptrs, dev_int_col_nullmask_ptrs, dev_cat_col_nullmask_ptrs,
            dev_categorical_col_hash_obj, dev_char_ptrs, dev_offset_ptrs, dev_out_buffer,
            host_out_buffer, binary_writer, dev_slot_size_array, rows_begin_train, rows_end_train,
            3);

        MESSAGE_("Porcessed file: " + input_file_path + " for /train_data.bin");
        MESSAGE_("Size of train_data.bin: " + std::to_string(sz_total_output_binary) + " Bytes.");

        if (binary_writer) binary_writer->close();
      }

      // validation-data and testing-data
      {
        int32_t rows_begin_val = 36672493,
                rows_end_val = 41256555;  // train.txt [36672493, 41256555)
        int32_t rows_begin_test = 41256555,
                rows_end_test = 45840617;  // train.txt [41256555, 45840617]
        std::string input_file_path = std::string(input_dir_path + "/train.txt");

        // val
        std::ofstream *binary_writer_val = nullptr;
        if (write_out)
          binary_writer_val =
              new std::ofstream(std::string(output_dir_path + "/val_data.bin"), std::ios::binary);

        sz_total_output_binary = convert_input_binaries<key_type, value_type>(
            p_mr, input_file_path, column_dtypes, column_names, hash_bucket, max_chunk_per_file, 0,
            false, dev_int_col_ptrs, dev_int_col_nullmask_ptrs, dev_cat_col_nullmask_ptrs,
            dev_categorical_col_hash_obj, dev_char_ptrs, dev_offset_ptrs, dev_out_buffer,
            host_out_buffer, binary_writer_val, dev_slot_size_array, rows_begin_val, rows_end_val,
            3);

        MESSAGE_("Size of val_data.bin: " + std::to_string(sz_total_output_binary) + " Bytes.");

        if (binary_writer_val) binary_writer_val->close();

        // test
        std::ofstream *binary_writer_test = nullptr;
        if (write_out)
          binary_writer_test =
              new std::ofstream(std::string(output_dir_path + "/test_data.bin"), std::ios::binary);

        sz_total_output_binary = convert_input_binaries<key_type, value_type>(
            p_mr, input_file_path, column_dtypes, column_names, hash_bucket, max_chunk_per_file, 0,
            false, dev_int_col_ptrs, dev_int_col_nullmask_ptrs, dev_cat_col_nullmask_ptrs,
            dev_categorical_col_hash_obj, dev_char_ptrs, dev_offset_ptrs, dev_out_buffer,
            host_out_buffer, binary_writer_test, dev_slot_size_array, rows_begin_test,
            rows_end_test, 3);

        MESSAGE_("Size of test_data.bin: " + std::to_string(sz_total_output_binary) + " Bytes.");

        if (binary_writer_test) binary_writer_test->close();
        MESSAGE_("Processed file: " + input_file_path + " for val_data.bin and test_data.bin");
      }

      const auto time_convert_stop = std::chrono::high_resolution_clock::now();
      const auto time_convert_total = std::chrono::duration_cast<std::chrono::milliseconds>(
          time_convert_stop - time_convert_start);
      MESSAGE_("Time to process binaries: " + std::to_string(time_convert_total.count()) +
               " milliseconds.");
      double p_read_bw = (double)process_read_bytes / (1024.0 * 1024.0 * 1024.0);
      p_read_bw = (p_read_bw / time_convert_total.count()) * 1000.f;

      double p_write_bw = (double)process_write_bytes / (1024.0 * 1024.0 * 1024.0);
      p_write_bw = (p_write_bw / time_convert_total.count()) * 1000.f;

      size_t total_second_pass_bytes = process_read_bytes + process_write_bytes;
      double p_2nd_bw = (double)total_second_pass_bytes / (1024.0 * 1024.0 * 1024.0);
      p_2nd_bw = (p_2nd_bw / time_convert_total.count()) * 1000.f;

      MESSAGE_("Convert Bytes reading: " + std::to_string(process_read_bytes) +
               ", Effective reading B/W: " + std::to_string(p_read_bw) + " GB/s.");
      MESSAGE_("Convert Bytes writing: " + std::to_string(process_write_bytes) +
               ", Effective reading B/W: " + std::to_string(p_write_bw) + " GB/s.");
      MESSAGE_("Convert Bytes total: " + std::to_string(total_second_pass_bytes) +
               ", Effective reading B/W: " + std::to_string(p_2nd_bw) + " GB/s.");
    }

    const auto program_end_time = std::chrono::high_resolution_clock::now();
    const auto application_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(program_end_time - time_map_start);
    double app_bw = (double)total_file_bytes_read / (1024.0 * 1024.0 * 1024.0);
    app_bw = (app_bw / application_time.count()) * 1000.f;

    MESSAGE_("Application process B/W: " + std::to_string(app_bw) + " GB/s.");

    CK_CUDA_THROW_(cudaFree(dev_out_buffer));
    CK_CUDA_THROW_(cudaFreeHost(host_out_buffer));

    CK_CUDA_THROW_(cudaFree(dev_int_col_ptrs));
    CK_CUDA_THROW_(cudaFree(dev_int_col_nullmask_ptrs));
    CK_CUDA_THROW_(cudaFree(dev_categorical_col_hash_obj));
    CK_CUDA_THROW_(cudaFree(dev_char_ptrs));
    CK_CUDA_THROW_(cudaFree(dev_offset_ptrs));
    CK_CUDA_THROW_(cudaFree(dev_slot_size_array));
    CK_CUDA_THROW_(cudaFree(dev_cat_col_nullmask_ptrs));
  }
  // destory map objects
  for (auto c : categorical_col_hash_tables) c.second->destroy();

  delete p_mr;
  p_mr = nullptr;
}

void process_terabyte_dataset(const std::string &input_dir_path, const std::string &output_dir_path,
                              const int num_numericals, const int num_categoricals,
                              const std::vector<std::string> &train_days,
                              const std::vector<std::string> &test_days) {
  int max_chunk_per_file = 10000;  // loop count, in a signle binary data, store how many chunks

  bool process_output = true;
  bool write_out = true;

  int min_cat_fea_cardi = 40000000;  // 40M
  int32_t hash_bucket = min_cat_fea_cardi;
  std::vector<int32_t> hist_sizes = {
      hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket,
      hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket,
      hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket,
      hash_bucket, hash_bucket, hash_bucket, hash_bucket, hash_bucket};  // mod-idx

  size_t pool_alloc_size = (size_t)10 * 1024 * 1024 * 1024;  // 10 GB
  rmm::mr::device_memory_resource *base_mr = new rmm::mr::cuda_memory_resource();
  auto *p_mr =
      new rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>(base_mr, pool_alloc_size);
  rmm::mr::set_current_device_resource(p_mr);

  std::vector<std::string> column_dtypes;                 // dtypes of label, dense, categorical
  std::vector<std::string> column_names;                  // names of label, dense, categorical
  std::vector<std::string> cat_column_names;              // names of categorical
  std::map<std::string, int32_t> column_name_to_col_idx;  // <col-name, idx>
  std::unordered_map<std::string, map_type<key_type, value_type> *>
      categorical_col_hash_tables;  // <name, <key, value>>

  // label
  column_dtypes.push_back("int32");
  column_names.push_back("label");
  column_name_to_col_idx.insert(std::make_pair("label", 0));

  // dense-features
  for (int k = 1; k <= 13; k++) {
    column_dtypes.push_back("int32");
    std::string name = "I" + std::to_string(k);
    column_names.push_back(name);
    column_name_to_col_idx.insert(std::make_pair(name, k));
  }

  // categorical-features
  for (int k = 1; k <= num_categoricals; k++) {
    column_dtypes.push_back("str");
    std::string name = "C" + std::to_string(k);
    column_names.push_back(name);
    cat_column_names.push_back(name);
    column_name_to_col_idx.insert(std::make_pair(name, k + num_numericals - 1));

    auto cuda_map_obj =
        map_type<key_type, value_type>::create(compute_hash_table_size(hist_sizes[k - 1]))
            .release();
    ;
    categorical_col_hash_tables.insert(std::make_pair(name, cuda_map_obj));
  }

  int current_device = 0;
  cudaDeviceProp prop;
  CK_CUDA_THROW_(cudaGetDeviceProperties(&prop, current_device));

  size_t read_chunks = 128 * 1024 * 1024;  // read 128MB at one time

  uint32_t *accum_location = nullptr;                // slot-size
  CK_CUDA_THROW_(cudaMalloc(&accum_location, 128));  // 128 Bytes = 32 * uint32_t
  CK_CUDA_THROW_(cudaMemset(accum_location, 0, 128));

  // uint32_t *culled_index_count = nullptr;
  // CK_CUDA_THROW_(cudaMalloc(&culled_index_count, 128)); // 128 Bytes = 32 * uint32_t

  size_t total_file_bytes_read = 0;
  const auto time_map_start = std::chrono::high_resolution_clock::now();

  // iteration on each day's data, including training and testing.
  std::vector<std::string> all_days;
  all_days.insert(all_days.end(), train_days.begin(), train_days.end());
  all_days.insert(all_days.end(), test_days.begin(), test_days.end());

  std::vector<size_t> sample_nums;
  for (const auto &day : all_days) {
    // get file size
    std::string input_file_name = input_dir_path + "/day_" + day;
    std::ifstream binary_reader(input_file_name, std::ios::binary);
    binary_reader.seekg(0, std::ios::end);
    size_t file_size = binary_reader.tellg();
    binary_reader.close();

    // csv arguments,
    // https://docs.rapids.ai/api/libcudf/stable/structcudf_1_1io_1_1read__csv__args.html
    cudf_io::csv_reader_options in_args =
        cudf_io::csv_reader_options::builder(cudf_io::source_info{input_file_name}).header(-1);

    in_args.set_dtypes(column_dtypes);
    in_args.set_names(column_names);
    in_args.set_delimiter('\t');
    in_args.set_byte_range_size(read_chunks);  // how many bytes to read at one time.
    in_args.set_skipfooter(0);
    in_args.set_skiprows(0);
    in_args.set_use_cols_names(cat_column_names);

    int32_t total_row_nums = 0;

    int loop_count = 0;
    while (true) {
      total_file_bytes_read += in_args.get_byte_range_size();
      cudf_io::table_with_metadata tbl_w_metadata = cudf_io::read_csv(in_args, p_mr);
      total_row_nums += tbl_w_metadata.tbl->num_rows();

      dim3 block(prop.maxThreadsPerBlock, 1, 1);
      dim3 grid((tbl_w_metadata.tbl->num_rows() - 1) / block.x + 1, 1, 1);

      // categorical-features
      for (unsigned int k = 0; k < cat_column_names.size(); ++k) {
        auto col = std::move(tbl_w_metadata.tbl->get_column(k));
        if (col.type().id() == cudf::type_id::STRING) {
          auto str_col = cudf::strings_column_view(col.view());
          int64_t num_strings = str_col.size();
          char *char_array = const_cast<char *>(str_col.chars().data<char>());
          int32_t *offsets = const_cast<int32_t *>(str_col.offsets().data<int32_t>());

          build_categorical_index<key_type, value_type><<<grid, block>>>(
              char_array, offsets, num_strings,
              // *categorical_col_hash_tables[cat_column_names[k]], hash_bucket,
              // &accum_location[k]);
              *categorical_col_hash_tables[cat_column_names[k]], hist_sizes[k], &accum_location[k]);

        } else if (col.type().id() == cudf::type_id::INT32) {
          key_type *data = const_cast<key_type *>(col.view().data<key_type>());
          bitmask_type *in_mask = const_cast<bitmask_type *>(col.view().null_mask());

          build_categorical_index_from_ints<key_type, value_type><<<grid, block>>>(
              data, in_mask, tbl_w_metadata.tbl->num_rows(),
              // *categorical_col_hash_tables[cat_column_names[k]], hash_bucket,
              // &accum_location[k]);
              *categorical_col_hash_tables[cat_column_names[k]], hist_sizes[k], &accum_location[k]);

        } else {
          ERROR_MESSAGE_("col.type().id() != [STRING, INT32]");
        }
      }

      size_t new_byte_range_offset = in_args.get_byte_range_offset() + read_chunks;
      in_args.set_byte_range_offset(new_byte_range_offset);
      if (in_args.get_byte_range_offset() >= file_size) break;

      if ((in_args.get_byte_range_offset() + read_chunks) > file_size) {
        size_t new_byte_range_size = file_size - in_args.get_byte_range_offset();
        in_args.set_byte_range_size(new_byte_range_size);
      }

      ++loop_count;

      if (loop_count == max_chunk_per_file) break;
    }
    MESSAGE_(input_file_name + "'s total rows number = " + std::to_string(total_row_nums));
    sample_nums.push_back(total_row_nums);

  }  // end for all_days

  // show: slot size array
  std::vector<uint32_t> host_sz_per_fea(num_categoricals);
  CK_CUDA_THROW_(cudaMemcpy(host_sz_per_fea.data(), accum_location,
                            num_categoricals * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  MESSAGE_("Slot size array, missing value mapped to unused key: ");
  for (auto c : host_sz_per_fea) std::cout << (c) << ", ";
  std::cout << "\b\b" << std::endl;

  const auto time_map_stop = std::chrono::high_resolution_clock::now();
  const auto time_map_build =
      std::chrono::duration_cast<std::chrono::milliseconds>(time_map_stop - time_map_start);
  MESSAGE_("Time used to build map: " + std::to_string(time_map_build.count()) + " milliseconds.");

  double read_bw = double(total_file_bytes_read) / (1024.0 * 1024.0 * 1024.0);
  read_bw = (read_bw / time_map_build.count()) * 1000.f;
  MESSAGE_("Total bytes read: " + std::to_string(total_file_bytes_read) +
           " Effective Read B/W: " + std::to_string(read_bw) + " GB/s.");

  // CK_CUDA_THROW_(cudaFree(culled_index_count));
  CK_CUDA_THROW_(cudaFree(accum_location));

  // starting to do the convertion
  if (process_output) {
    uint32_t *dev_slot_size_array = nullptr;
    size_t slot_size_array_size = num_categoricals * sizeof(uint32_t);
    CK_CUDA_THROW_(cudaMalloc(&dev_slot_size_array, slot_size_array_size));
    CK_CUDA_THROW_(cudaMemcpy(dev_slot_size_array, host_sz_per_fea.data(), slot_size_array_size,
                              cudaMemcpyHostToDevice));

    int32_t *dev_out_buffer = nullptr;
    int32_t *host_out_buffer = nullptr;

    size_t sz_output_buffer = 128 * 1024 * 1024;  // 128 MB, = read_chunks
    CK_CUDA_THROW_(cudaMalloc(&dev_out_buffer, sz_output_buffer));
    CK_CUDA_THROW_(cudaMallocHost(&host_out_buffer, sz_output_buffer));

    int64_t *dev_int_col_ptrs = nullptr;
    int64_t *dev_int_col_nullmask_ptrs = nullptr;
    int64_t *dev_cat_col_nullmask_ptrs = nullptr;
    int64_t *dev_categorical_col_hash_obj = nullptr;
    int64_t *dev_char_ptrs = nullptr;
    int64_t *dev_offset_ptrs = nullptr;

    size_t sz_dev_int_col = num_numericals * sizeof(int64_t);
    size_t sz_dev_cat_hash_obj = num_categoricals * sizeof(map_type<key_type, value_type>);
    size_t sz_dev_str_ptrs = num_categoricals * sizeof(int64_t);

    CK_CUDA_THROW_(cudaMalloc(&dev_int_col_ptrs, sz_dev_int_col));
    CK_CUDA_THROW_(cudaMalloc(&dev_int_col_nullmask_ptrs, sz_dev_int_col));
    CK_CUDA_THROW_(cudaMalloc(&dev_cat_col_nullmask_ptrs, sz_dev_str_ptrs));
    CK_CUDA_THROW_(cudaMalloc(&dev_categorical_col_hash_obj, sz_dev_cat_hash_obj));
    CK_CUDA_THROW_(cudaMalloc(&dev_char_ptrs, sz_dev_str_ptrs));
    CK_CUDA_THROW_(cudaMalloc(&dev_offset_ptrs, sz_dev_str_ptrs));

    // encode and write out binary
    int maxbytes = 96 * 1024;  // dynamic shared memory size 96 KB
    cudaFuncSetAttribute(process_data_rows<key_type, value_type>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

    std::vector<map_type<key_type, value_type>> categorical_col_hash_obj;
    for (auto c : cat_column_names) {
      categorical_col_hash_obj.push_back(*categorical_col_hash_tables[c]);
    }

    CK_CUDA_THROW_(cudaMemcpy((void *)dev_categorical_col_hash_obj,
                              (void *)categorical_col_hash_obj.data(), sz_dev_cat_hash_obj,
                              cudaMemcpyHostToDevice));

    if (process_output) {
      const auto time_convert_start = std::chrono::high_resolution_clock::now();

      std::ofstream *binary_writer = nullptr;
      if (write_out)
        binary_writer =
            new std::ofstream(std::string(output_dir_path + "/train_data.bin"), std::ios::binary);
      size_t sz_total_output_binary = 0;

      // train_data.bin
      size_t saved_samples_num = 0;
      for (size_t i = 0; i < train_days.size(); i++) {
        const auto &day = train_days[i];
        size_t needed_samples_num = 4195197692 - saved_samples_num;  // total should be 4195197692
        int32_t rows_begin_train = -1, rows_end_train = -1;          // train.txt [:36672000)
        if (needed_samples_num < sample_nums[i]) rows_end_train = needed_samples_num;

        std::string input_file_path = input_dir_path + "/day_" + day;
        sz_total_output_binary += convert_input_binaries<key_type, value_type>(
            p_mr, input_file_path, column_dtypes, column_names, hash_bucket, max_chunk_per_file, 0,
            false, dev_int_col_ptrs, dev_int_col_nullmask_ptrs, dev_cat_col_nullmask_ptrs,
            dev_categorical_col_hash_obj, dev_char_ptrs, dev_offset_ptrs, dev_out_buffer,
            host_out_buffer, binary_writer, dev_slot_size_array, rows_begin_train, rows_end_train,
            1);

        MESSAGE_("Porcessed file: " + input_file_path + " for /train_data.bin");

        if (needed_samples_num < sample_nums[i]) {
          saved_samples_num += needed_samples_num;
          break;
        } else {
          saved_samples_num += sample_nums[i];
        }

      }  // end for train_days
      MESSAGE_("Size of train_data.bin: " + std::to_string(sz_total_output_binary) + " Bytes.");
      if (binary_writer) binary_writer->close();

      // testing-data
      {
        // test_data.bin
        std::ofstream *binary_writer_test = nullptr;
        if (write_out)
          binary_writer_test =
              new std::ofstream(std::string(output_dir_path + "/test_data.bin"), std::ios::binary);

        sz_total_output_binary = 0;
        size_t saved_samples_num = 0;
        for (size_t i = 0; i < test_days.size(); ++i) {
          const auto &day = test_days[i];
          size_t needed_samples_num = 89137319 - saved_samples_num;  // total should be 89137319
          int32_t rows_begin_test = -1, rows_end_test = -1;
          if (needed_samples_num < sample_nums[train_days.size() + i])
            rows_end_test = needed_samples_num;

          // rows_begin_test = 89137318; rows_end_test = -1; // [89137318: ), second half

          std::string input_file_path = input_dir_path + "/day_" + day;
          sz_total_output_binary += convert_input_binaries<key_type, value_type>(
              p_mr, input_file_path, column_dtypes, column_names, hash_bucket, max_chunk_per_file,
              0, false, dev_int_col_ptrs, dev_int_col_nullmask_ptrs, dev_cat_col_nullmask_ptrs,
              dev_categorical_col_hash_obj, dev_char_ptrs, dev_offset_ptrs, dev_out_buffer,
              host_out_buffer, binary_writer_test, dev_slot_size_array, rows_begin_test,
              rows_end_test, 1);

          MESSAGE_("Porcessed file: " + input_file_path + " for /test_data.bin");

          if (needed_samples_num < sample_nums[train_days.size() + i]) {
            saved_samples_num += needed_samples_num;
            break;
          } else {
            saved_samples_num += sample_nums[train_days.size() + i];
          }

        }  // end for test_days

        MESSAGE_("Size of test_data.bin: " + std::to_string(sz_total_output_binary) + " Bytes.");

        if (binary_writer_test) binary_writer_test->close();
      }

      const auto time_convert_stop = std::chrono::high_resolution_clock::now();
      const auto time_convert_total = std::chrono::duration_cast<std::chrono::milliseconds>(
          time_convert_stop - time_convert_start);
      MESSAGE_("Time to process binaries: " + std::to_string(time_convert_total.count()) +
               " milliseconds.");
      double p_read_bw = (double)process_read_bytes / (1024.0 * 1024.0 * 1024.0);
      p_read_bw = (p_read_bw / time_convert_total.count()) * 1000.f;

      double p_write_bw = (double)process_write_bytes / (1024.0 * 1024.0 * 1024.0);
      p_write_bw = (p_write_bw / time_convert_total.count()) * 1000.f;

      size_t total_second_pass_bytes = process_read_bytes + process_write_bytes;
      double p_2nd_bw = (double)total_second_pass_bytes / (1024.0 * 1024.0 * 1024.0);
      p_2nd_bw = (p_2nd_bw / time_convert_total.count()) * 1000.f;

      MESSAGE_("Convert Bytes reading: " + std::to_string(process_read_bytes) +
               ", Effective reading B/W: " + std::to_string(p_read_bw) + " GB/s.");
      MESSAGE_("Convert Bytes writing: " + std::to_string(process_write_bytes) +
               ", Effective reading B/W: " + std::to_string(p_write_bw) + " GB/s.");
      MESSAGE_("Convert Bytes total: " + std::to_string(total_second_pass_bytes) +
               ", Effective reading B/W: " + std::to_string(p_2nd_bw) + " GB/s.");
    }

    const auto program_end_time = std::chrono::high_resolution_clock::now();
    const auto application_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(program_end_time - time_map_start);
    double app_bw = (double)total_file_bytes_read / (1024.0 * 1024.0 * 1024.0);
    app_bw = (app_bw / application_time.count()) * 1000.f;

    MESSAGE_("Application process B/W: " + std::to_string(app_bw) + " GB/s.");

    CK_CUDA_THROW_(cudaFree(dev_out_buffer));
    CK_CUDA_THROW_(cudaFreeHost(host_out_buffer));

    CK_CUDA_THROW_(cudaFree(dev_int_col_ptrs));
    CK_CUDA_THROW_(cudaFree(dev_int_col_nullmask_ptrs));
    CK_CUDA_THROW_(cudaFree(dev_categorical_col_hash_obj));
    CK_CUDA_THROW_(cudaFree(dev_char_ptrs));
    CK_CUDA_THROW_(cudaFree(dev_offset_ptrs));
    CK_CUDA_THROW_(cudaFree(dev_slot_size_array));
    CK_CUDA_THROW_(cudaFree(dev_cat_col_nullmask_ptrs));
  }
  // destory map objects
  for (auto c : categorical_col_hash_tables) c.second->destroy();

  delete p_mr;
  p_mr = nullptr;
}

int main(const int argc, const char *argv[]) {
  if (argc < 3) {
    MESSAGE_("Need min 2 args: input_dir output_dir");
    MESSAGE_("Usage for Kaggle Datasets: ./dlrm_raw input_dir output_dir");
    MESSAGE_(
        "Usage for TeraBytes Datasets: ./dlrm_raw input_dir output_dir --train [days for training] "
        "--test [days for testing]"
        ", those days are seperated with comma, no whitespace.");
    return -1;
  }

  const int num_numericals = 14;    // label + 13 int-dense-feature
  const int num_categoricals = 26;  // 26 int-categorical-feature

  std::string input_dir_path(argv[1]);
  std::string output_dir_path(argv[2]);

  switch (argc) {
    case 3: {
      MESSAGE_("Processing Kaggle datasets");
      MESSAGE_("input_dir: " + input_dir_path);
      MESSAGE_("output_dir: " + output_dir_path);

      process_kaggle_dataset(input_dir_path, output_dir_path, num_numericals, num_categoricals);
      break;
    }

    case 7: {
      if (argc == 7 &&
          (std::strcmp(argv[3], "--train") != 0 || std::strcmp(argv[5], "--test") != 0)) {
        MESSAGE_(
            "Usage for TeraBytes Datasets: ./dlrm_raw input_dir output_dir --train [days for "
            "training] --test [days for testing]"
            ", those days are seperated with comma, no whitespace.");
        MESSAGE_("For example: ./dlrm_raw ./ ./ --train 0,1,2,3,4 --test 5,6,7");
        return -1;
      }

      const std::vector<std::string> train_days = split_string(std::string(argv[4]), ",");
      const std::vector<std::string> test_days = split_string(std::string(argv[6]), ",");

      MESSAGE_("Processing TeraBytes datasets.");
      MESSAGE_("input_dir: " + input_dir_path);
      MESSAGE_("output_dir: " + output_dir_path);
      MESSAGE_("days for training: " + std::string(argv[4]));
      MESSAGE_("days for testing: " + std::string(argv[6]));

      process_terabyte_dataset(input_dir_path, output_dir_path, num_numericals, num_categoricals,
                               train_days, test_days);
      break;
    }

    default: {
      MESSAGE_("Usage for Kaggle Datasets: ./dlrm_raw input_dir output_dir");
      MESSAGE_(
          "Usage for TeraBytes Datasets: ./dlrm_raw input_dir output_dir --train [days for "
          "training] --test [days for testing]"
          ", those days are seperated with comma, no whitespace.");
      return -1;
      break;
    }
  }

  MESSAGE_("Done.");
  return 0;
}