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

#include <sys/stat.h>

#include <base/debug/logger.hpp>
#include <common.hpp>
#include <fstream>
#include <memory>
#include <random>

#ifndef DISABLE_CUDF
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#endif

#include <rmm/device_buffer.hpp>

namespace HugeCTR {

/**
 * Check if file exist.
 */
inline bool file_exist(const std::string& name) {
  if (FILE* file = fopen(name.c_str(), "r")) {
    fclose(file);
    return true;
  } else {
    return false;
  }
}

/**
 * Check if file path exist if not create it.
 */
inline void check_make_dir(const std::string& finalpath) {
  if (mkdir(finalpath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
    if (errno == EEXIST) {
      HCTR_LOG_S(INFO, WORLD) << finalpath << " exist" << std::endl;
    } else {
      HCTR_OWN_THROW(Error_t::UnspecificError, "cannot create" + finalpath + ": unexpected error");
      exit(-1);
    }
  }
}

/**
 * Extract directory from path
 */
inline std::string extract_dir(const std::string& source_data) {
  std::string source_dir;
  const size_t last_slash_idx = source_data.rfind('/');
  if (std::string::npos != last_slash_idx) {
    source_dir = source_data.substr(0, last_slash_idx);
  }
  return source_dir;
}

template <typename T>
class IDataSimulator {
 public:
  virtual ~IDataSimulator() {}
  virtual T get_num() = 0;
};

template <typename T>
class FloatUniformDataSimulator {
 public:
  FloatUniformDataSimulator(T min, T max) : gen_(std::random_device()()), dis_(min, max) {}

  T get_num() { return dis_(gen_); }

 private:
  std::mt19937 gen_;
  std::uniform_real_distribution<T> dis_;
};

template <typename T>
class IntUniformDataSimulator : public IDataSimulator<T> {
 public:
  IntUniformDataSimulator(T min, T max) : gen_(std::random_device()()), dis_(min, max) {}

  T get_num() override { return dis_(gen_); }

 private:
  std::mt19937 gen_;
  std::uniform_int_distribution<T> dis_;
};

template <typename T>
class IntPowerLawDataSimulator : public IDataSimulator<T> {
 public:
  IntPowerLawDataSimulator(T min, T max, float alpha)
      : gen_(std::random_device()()), dis_(0, 1), alpha_(alpha) {
    min_ = 1.0;
    max_ = max - min + 1.0;
    offset_ = min - 1.0;  // to handle the case min_ <= 0
  }

  T get_num() override {
    double x = dis_(gen_);
    double y = (pow((pow(max_, 1 - alpha_) - pow(min_, 1 - alpha_)) * x + pow(min_, 1 - alpha_),
                    1.0 / (1.0 - alpha_)));
    return static_cast<T>(round(y) + offset_);
  }

 private:
  std::mt19937 gen_;
  std::uniform_real_distribution<float> dis_;
  float alpha_;  // requiring alpha_ > 0 and alpha_ != 1.0
  double min_, max_, offset_;
};

/**
 * Generate random dataset for HugeCTR test.
 */

template <Check_t T>
class Checker_Traits;

template <>
class Checker_Traits<Check_t::Sum> {
 public:
  static char zero() { return 0; }

  static char accum(char pre, char x) { return pre + x; }

  static void write(int N, char* array, char chk_bits, std::ofstream& stream) {
    stream.write(reinterpret_cast<char*>(&N), sizeof(int));
    stream.write(reinterpret_cast<char*>(array), N);
    stream.write(reinterpret_cast<char*>(&chk_bits), sizeof(char));
  }

  static long long ID() { return 1; }
};

template <>
class Checker_Traits<Check_t::None> {
 public:
  static char zero() { return 0; }

  static char accum(char pre, char x) { return 0; }

  static void write(int N, char* array, char chk_bits, std::ofstream& stream) {
    stream.write(reinterpret_cast<char*>(array), N);
  }

  static long long ID() { return 0; }
};

template <Check_t T>
class DataWriter {
  std::vector<char> array_;
  std::ofstream& stream_;
  char check_char_{0};

 public:
  DataWriter(std::ofstream& stream) : stream_(stream) { check_char_ = Checker_Traits<T>::zero(); }
  void append(char* array, int N) {
    for (int i = 0; i < N; i++) {
      array_.push_back(array[i]);
      check_char_ = Checker_Traits<T>::accum(check_char_, array[i]);
    }
  }
  void write() {
    Checker_Traits<T>::write(static_cast<int>(array_.size()), array_.data(), check_char_, stream_);
    check_char_ = Checker_Traits<T>::zero();
    array_.clear();
  }
};

template <typename T, Check_t CK_T>
void data_generation_for_test(std::string file_list_name, std::string data_prefix, int num_files,
                              int num_records_per_file, int slot_num, int vocabulary_size,
                              int label_dim, int dense_dim, int max_nnz, bool long_tail = false,
                              float alpha = 0.0, std::vector<T>* generated_value = nullptr,
                              std::vector<T>* generated_rowoffset = nullptr,
                              std::vector<float>* generated_label = nullptr,
                              std::vector<float>* generated_dense = nullptr) {
  if (file_exist(file_list_name)) {
    HCTR_LOG_S(INFO, WORLD) << "File (" << file_list_name
                            << ") exist. To generate new dataset plesae remove this file."
                            << std::endl;
    return;
  }
  std::string directory;
  const size_t last_slash_idx = data_prefix.rfind('/');
  if (std::string::npos != last_slash_idx) {
    directory = data_prefix.substr(0, last_slash_idx);
  }
  check_make_dir(directory);

  std::ofstream file_list_stream(file_list_name, std::ofstream::out);
  file_list_stream << (std::to_string(num_files) + "\n");
  for (int k = 0; k < num_files; k++) {
    std::string tmp_file_name(data_prefix + std::to_string(k) + ".data");
    file_list_stream << (tmp_file_name + "\n");
    HCTR_LOG_S(INFO, WORLD) << tmp_file_name << std::endl;
    // data generation;
    std::ofstream out_stream(tmp_file_name, std::ofstream::binary);

    DataWriter<CK_T> data_writer(out_stream);

    DataSetHeader header = {
        Checker_Traits<CK_T>::ID(), num_records_per_file, label_dim, dense_dim, slot_num, 0, 0, 0};

    data_writer.append(reinterpret_cast<char*>(&header), sizeof(DataSetHeader));
    data_writer.write();

    for (int i = 0; i < num_records_per_file; i++) {
      IntUniformDataSimulator<int> idata_sim(1, max_nnz);  // for nnz
      FloatUniformDataSimulator<float> fdata_sim(0, 1);    // for lable and dense
      std::shared_ptr<IDataSimulator<T>> ldata_sim;
      if (long_tail)
        ldata_sim.reset(new IntPowerLawDataSimulator<T>(0, vocabulary_size - 1, alpha));
      else
        ldata_sim.reset(new IntUniformDataSimulator<T>(0, vocabulary_size - 1));  // for key
      for (int j = 0; j < label_dim + dense_dim; j++) {
        float label_dense = fdata_sim.get_num();
        if (j < label_dim && generated_label != nullptr) {
          generated_label->push_back(label_dense);
        }
        if (j >= label_dim && generated_dense != nullptr) {
          generated_dense->push_back(label_dense);
        }
        data_writer.append(reinterpret_cast<char*>(&label_dense), sizeof(float));
      }
      for (int k = 0; k < slot_num; k++) {
        int nnz = idata_sim.get_num();
        data_writer.append(reinterpret_cast<char*>(&nnz), sizeof(int));
        for (int j = 0; j < nnz; j++) {
          T key = ldata_sim->get_num();
          while ((key % static_cast<T>(slot_num)) !=
                 static_cast<T>(k)) {  // guarantee the key belongs to the current slot_id(=k)
            key = ldata_sim->get_num();
          }
          data_writer.append(reinterpret_cast<char*>(&key), sizeof(T));
          if (generated_value != nullptr) {
            generated_value->push_back(key);
          }
        }
        if (generated_rowoffset != nullptr) {
          generated_rowoffset->push_back(nnz);
        }
      }
      data_writer.write();
    }
    out_stream.close();
  }
  file_list_stream.close();
  HCTR_LOG_S(INFO, WORLD) << file_list_name << " done!" << std::endl;
  return;
}

/**
 * Adding finer control e.g.: --voc-size-array=312,32,231234,124332,4554
 * --nnz-array=23,3,45,66,23,1,1,1,1
 */
template <typename T, Check_t CK_T>
void data_generation_for_test2(std::string file_list_name, std::string data_prefix, int num_files,
                               int num_records_per_file, int slot_num,
                               std::vector<size_t> voc_size_array, int label_dim, int dense_dim,
                               std::vector<int> nnz_array, bool long_tail = false,
                               float alpha = 0.0) {
  // check if slot_num == voc_size_array.size == nnz_array.size
  if (slot_num != (int)voc_size_array.size() || slot_num != (int)nnz_array.size()) {
    HCTR_LOG(ERROR, WORLD, "slot_num != voc_size_array.size() || slot_num != nnz_array.size()\n");
    exit(-1);
  }

  if (file_exist(file_list_name)) {
    HCTR_LOG_S(INFO, WORLD) << "File (" << file_list_name
                            << ") exist. To generate new dataset plesae remove this file."
                            << std::endl;
    return;
  }
  std::string directory;
  const size_t last_slash_idx = data_prefix.rfind('/');
  if (std::string::npos != last_slash_idx) {
    directory = data_prefix.substr(0, last_slash_idx);
  }
  check_make_dir(directory);

  std::ofstream file_list_stream(file_list_name, std::ofstream::out);
  file_list_stream << (std::to_string(num_files) + "\n");
  for (int k = 0; k < num_files; k++) {
    std::string tmp_file_name(data_prefix + std::to_string(k) + ".data");
    file_list_stream << (tmp_file_name + "\n");
    HCTR_LOG_S(INFO, WORLD) << tmp_file_name << std::endl;
    // data generation;
    std::ofstream out_stream(tmp_file_name, std::ofstream::binary);

    DataWriter<CK_T> data_writer(out_stream);

    DataSetHeader header = {
        Checker_Traits<CK_T>::ID(), num_records_per_file, label_dim, dense_dim, slot_num, 0, 0, 0};

    data_writer.append(reinterpret_cast<char*>(&header), sizeof(DataSetHeader));
    data_writer.write();
    // Initialize Simulators
    FloatUniformDataSimulator<float> fdata_sim(0, 1);  // for lable and dense
    std::vector<std::shared_ptr<IDataSimulator<T>>> ldata_sim_vec;
    size_t accum = 0;
    // todo risk of type Int
    for (auto& voc : voc_size_array) {
      size_t accum_next = accum + voc;
      if (long_tail) {
        ldata_sim_vec.emplace_back(new IntPowerLawDataSimulator<T>(accum, accum_next - 1, alpha));
      } else {
        ldata_sim_vec.emplace_back(new IntUniformDataSimulator<T>(accum, accum_next - 1));
      }
      accum = accum_next;
    }

    for (int i = 0; i < num_records_per_file; i++) {
      for (int j = 0; j < label_dim + dense_dim; j++) {
        float label_dense = fdata_sim.get_num();
        data_writer.append(reinterpret_cast<char*>(&label_dense), sizeof(float));
      }

      for (int k = 0; k < slot_num; k++) {
        int nnz = nnz_array[k];
        data_writer.append(reinterpret_cast<char*>(&nnz), sizeof(int));
        for (int j = 0; j < nnz; j++) {
          T key = ldata_sim_vec[k]->get_num();
          data_writer.append(reinterpret_cast<char*>(&key), sizeof(T));
        }
      }

      data_writer.write();
    }

    // for (int i = 0; i < num_records_per_file; i++) {
    //   IntUniformDataSimulator<int> idata_sim(1, max_nnz);            // for nnz
    //   FloatUniformDataSimulator<float> fdata_sim(0, 1);              // for lable and dense
    //   std::shared_ptr<IDataSimulator<T>> ldata_sim;
    //   if (long_tail)
    //     ldata_sim.reset(new IntPowerLawDataSimulator<T>(0, vocabulary_size - 1, alpha));
    //   else
    //     ldata_sim.reset(new IntUniformDataSimulator<T>(0, vocabulary_size - 1));  // for key
    //   for (int j = 0; j < label_dim + dense_dim; j++) {
    //     float label_dense = fdata_sim.get_num();
    //     data_writer.append(reinterpret_cast<char*>(&label_dense), sizeof(float));
    //   }
    //   for (int k = 0; k < slot_num; k++) {
    //     int nnz = idata_sim.get_num();
    //     data_writer.append(reinterpret_cast<char*>(&nnz), sizeof(int));
    //     for (int j = 0; j < nnz; j++) {
    //       T key = ldata_sim->get_num();
    //       while ((key % static_cast<T>(slot_num)) !=
    //              static_cast<T>(k)) {  // guarantee the key belongs to the current slot_id(=k)
    //         key = ldata_sim->get_num();
    //       }
    //       data_writer.append(reinterpret_cast<char*>(&key), sizeof(T));
    //     }
    //   }
    //   data_writer.write();
    // }
    out_stream.close();
  }
  file_list_stream.close();
  HCTR_LOG_S(INFO, WORLD) << file_list_name << " done!" << std::endl;
  return;
}

#ifndef DISABLE_CUDF
template <typename T>
void data_generation_for_parquet(std::string file_list_name, std::string data_prefix, int num_files,
                                 int num_records_per_file, int slot_num, int label_dim,
                                 int dense_dim, const std::vector<size_t> slot_size_array,
                                 std::vector<int> nnz_array, bool long_tail = false,
                                 float alpha = 0.0) {
  if (slot_num != (int)slot_size_array.size() || slot_num != (int)nnz_array.size()) {
    HCTR_LOG(ERROR, WORLD, "slot_num != slot_size_array.size() || slot_num != nnz_array.size()\n");
    exit(-1);
  }
  std::string directory;
  const size_t last_slash_idx = data_prefix.rfind('/');
  if (std::string::npos != last_slash_idx) {
    directory = data_prefix.substr(0, last_slash_idx);
  }
  check_make_dir(directory);

  std::ofstream file_list_stream(file_list_name, std::ofstream::out);
  file_list_stream << (std::to_string(num_files) + "\n");

  using CVector = std::vector<std::unique_ptr<cudf::column>>;
  for (int k = 0; k < num_files; k++) {
    // cudf columns
    CVector cols;
    std::string tmp_file_name(data_prefix + std::to_string(k) + ".parquet");
    file_list_stream << (tmp_file_name + "\n");
    HCTR_LOG_S(INFO, WORLD) << tmp_file_name << std::endl;
    // Initialize Simulators
    FloatUniformDataSimulator<float> fdata_sim(0, 1);  // for lable and dense
    std::vector<std::shared_ptr<IDataSimulator<T>>> ldata_sim_vec;
    // todo risk of type Int
    for (auto& voc : slot_size_array) {
      size_t accum_next = voc;
      if (long_tail) {
        ldata_sim_vec.emplace_back(new IntPowerLawDataSimulator<T>(0, accum_next - 1, alpha));
      } else {
        ldata_sim_vec.emplace_back(new IntUniformDataSimulator<T>(0, accum_next - 1));
      }
    }

    // for label columns
    for (int j = 0; j < label_dim; j++) {
      std::vector<float> label_vector(num_records_per_file);
      for (int i = 0; i < num_records_per_file; i++) {
        float label_ = fdata_sim.get_num();
        label_vector[i] = label_;
      }
      rmm::device_buffer dev_buffer(label_vector.data(), sizeof(float) * num_records_per_file,
                                    rmm::cuda_stream_default);
      cols.emplace_back(std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<float>()},
                                                       cudf::size_type(num_records_per_file),
                                                       std::move(dev_buffer)));
    }
    // for dense columns
    for (int j = 0; j < dense_dim; j++) {
      std::vector<float> dense_vector(num_records_per_file);
      for (int i = 0; i < num_records_per_file; i++) {
        float _dense = fdata_sim.get_num();
        dense_vector[i] = _dense;
      }
      rmm::device_buffer dev_buffer(dense_vector.data(), sizeof(float) * num_records_per_file,
                                    rmm::cuda_stream_default);
      cols.emplace_back(std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<float>()},
                                                       cudf::size_type(num_records_per_file),
                                                       std::move(dev_buffer)));
    }
    // for sparse columns
    // size_t offset = 0;
    for (int k = 0; k < slot_num; k++) {
      std::vector<T> slot_vector;
      std::vector<int32_t> row_offset_vector(num_records_per_file + 1);
      constexpr size_t bitmask_bits = cudf::detail::size_in_bits<cudf::bitmask_type>();
      size_t bits = (num_records_per_file + bitmask_bits - 1) / bitmask_bits;
      std::vector<cudf::bitmask_type> null_mask(bits, 0);
      int32_t offset = 0;
      for (int i = 0; i < num_records_per_file; i++) {
        int nnz = 1 + rand() % nnz_array[k];
        row_offset_vector[i] = offset;
        offset += nnz;
        for (int j = 0; j < nnz; j++) {
          T key = ldata_sim_vec[k]->get_num();
          slot_vector.push_back(key);
        }
      }
      row_offset_vector[num_records_per_file] = offset;
      if (nnz_array[k] == 1) {
        rmm::device_buffer dev_buffer(slot_vector.data(), sizeof(T) * slot_vector.size(),
                                      rmm::cuda_stream_default);
        cols.emplace_back(std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<T>()},
                                                         cudf::size_type(slot_vector.size()),
                                                         std::move(dev_buffer)));

      } else {
        rmm::device_buffer dev_buffer_0(slot_vector.data(), sizeof(T) * slot_vector.size(),
                                        rmm::cuda_stream_default);
        auto child = std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<T>()},
                                                    cudf::size_type(slot_vector.size()),
                                                    std::move(dev_buffer_0));
        rmm::device_buffer dev_buffer_1(row_offset_vector.data(),
                                        sizeof(int32_t) * row_offset_vector.size(),
                                        rmm::cuda_stream_default);
        auto row_off = std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<int32_t>()},
                                                      cudf::size_type(row_offset_vector.size()),
                                                      std::move(dev_buffer_1));
        cols.emplace_back(cudf::make_lists_column(
            num_records_per_file, std::move(row_off), std::move(child), cudf::UNKNOWN_NULL_COUNT,
            rmm::device_buffer(null_mask.data(), null_mask.size() * sizeof(cudf::bitmask_type),
                               rmm::cuda_stream_default)));
      }
    }
    cudf::table input_table(std::move(cols));
    cudf::io::parquet_writer_options writer_args = cudf::io::parquet_writer_options::builder(
        cudf::io::sink_info{tmp_file_name}, input_table.view());
    cudf::io::write_parquet(writer_args);
  }
  file_list_stream.close();
  HCTR_LOG_S(INFO, WORLD) << file_list_name << " done!" << std::endl;
  // also write metadata
  std::ostringstream metadata;
  metadata << "{ \"file_stats\": [";
  for (int i = 0; i < num_files - 1; i++) {
    std::string filepath = std::to_string(i) + std::string(".parquet");
    metadata << "{\"file_name\": \"" << filepath << "\", "
             << "\"num_rows\":" << num_records_per_file << "}, ";
  }
  std::string filepath = std::to_string(num_files - 1) + std::string(".parquet");
  metadata << "{\"file_name\": \"" << filepath << "\", "
           << "\"num_rows\":" << num_records_per_file << "} ";
  metadata << "], ";
  metadata << "\"labels\": [";
  for (int i = 0; i < label_dim - 1; i++) {
    metadata << "{\"col_name\": \"label" << i << "\", "
             << "\"index\":" << i << "}, ";
  }
  metadata << "{\"col_name\": \"label" << label_dim - 1 << "\", "
           << "\"index\":" << label_dim - 1 << "} ";
  metadata << "], ";

  metadata << "\"conts\": [";
  for (int i = label_dim; i < (label_dim + dense_dim - 1); i++) {
    metadata << "{\"col_name\": \"C" << i << "\", "
             << "\"index\":" << i << "}, ";
  }
  metadata << "{\"col_name\": \"C" << (label_dim + dense_dim - 1) << "\", "
           << "\"index\":" << (label_dim + dense_dim - 1) << "} ";
  metadata << "], ";

  metadata << "\"cats\": [";
  for (int i = label_dim + dense_dim; i < (label_dim + dense_dim + slot_num - 1); i++) {
    metadata << "{\"col_name\": \"C" << i << "\", "
             << "\"index\":" << i << "}, ";
  }
  metadata << "{\"col_name\": \"C" << (label_dim + dense_dim + slot_num - 1) << "\", "
           << "\"index\":" << (label_dim + dense_dim + slot_num - 1) << "} ";
  metadata << "] ";
  metadata << "}";

  std::ofstream metadata_file_stream{directory + "/_metadata.json"};
  metadata_file_stream << metadata.str();
  metadata_file_stream.close();
}
#endif

// Add a new data_generation function for LocalizedSparseEmbedding testing
// In this function, the relationship between key and slot_id is: key's slot_id=(key%slot_num)
// Add a new data_generation function for LocalizedSparseEmbedding testing
// In this function, the relationship between key and slot_id is: key's slot_id=(key%slot_num)
template <typename T, Check_t CK_T>
void data_generation_for_localized_test(std::string file_list_name, std::string data_prefix,
                                        int num_files, int num_records_per_file, int slot_num,
                                        int vocabulary_size, int label_dim, int dense_dim,
                                        int max_nnz, bool long_tail = false, float alpha = 0.0) {
  if (file_exist(file_list_name)) {
    return;
  }
  std::string directory;
  const size_t last_slash_idx = data_prefix.rfind('/');
  if (std::string::npos != last_slash_idx) {
    directory = data_prefix.substr(0, last_slash_idx);
  }
  check_make_dir(directory);

  std::ofstream file_list_stream(file_list_name, std::ofstream::out);
  file_list_stream << (std::to_string(num_files) + "\n");
  for (int k = 0; k < num_files; k++) {
    std::string tmp_file_name(data_prefix + std::to_string(k) + ".data");
    file_list_stream << (tmp_file_name + "\n");

    // data generation;
    std::ofstream out_stream(tmp_file_name, std::ofstream::binary);

    DataWriter<CK_T> data_writer(out_stream);

    DataSetHeader header = {1, num_records_per_file, label_dim, dense_dim, slot_num, 0, 0, 0};

    data_writer.append(reinterpret_cast<char*>(&header), sizeof(DataSetHeader));
    data_writer.write();

    for (int i = 0; i < num_records_per_file; i++) {
      IntUniformDataSimulator<int> idata_sim(1, max_nnz);  // for nnz
      FloatUniformDataSimulator<float> fdata_sim(0, 1);    // for lable and dense
      std::shared_ptr<IDataSimulator<T>> ldata_sim;
      if (long_tail)
        ldata_sim.reset(new IntPowerLawDataSimulator<T>(0, vocabulary_size - 1, alpha));
      else
        ldata_sim.reset(new IntUniformDataSimulator<T>(0, vocabulary_size - 1));  // for key
      for (int j = 0; j < label_dim + dense_dim; j++) {
        float label_dense = fdata_sim.get_num();
        data_writer.append(reinterpret_cast<char*>(&label_dense), sizeof(float));
      }
      for (int k = 0; k < slot_num; k++) {
        int nnz = idata_sim.get_num();
        data_writer.append(reinterpret_cast<char*>(&nnz), sizeof(int));
        for (int j = 0; j < nnz; j++) {
          T key = ldata_sim->get_num();
          while ((key % slot_num) != k) {  // guarantee the key belongs to the current slot_id(=k)
            key = ldata_sim->get_num();
          }
          data_writer.append(reinterpret_cast<char*>(&key), sizeof(T));
        }
      }
      data_writer.write();
    }
    out_stream.close();
  }
  file_list_stream.close();
  return;
}

template <typename T, Check_t CK_T>
void data_generation_for_localized_test(std::string file_list_name, std::string data_prefix,
                                        int num_files, int num_records_per_file, int slot_num,
                                        int vocabulary_size, int label_dim, int dense_dim,
                                        int max_nnz, const std::vector<size_t> slot_sizes,
                                        bool long_tail = false, float alpha = 0.0) {
  if (file_exist(file_list_name)) {
    return;
  }
  std::string directory;
  const size_t last_slash_idx = data_prefix.rfind('/');
  if (std::string::npos != last_slash_idx) {
    directory = data_prefix.substr(0, last_slash_idx);
  }
  check_make_dir(directory);

  std::ofstream file_list_stream(file_list_name, std::ofstream::out);
  file_list_stream << (std::to_string(num_files) + "\n");
  for (int k = 0; k < num_files; k++) {
    std::string tmp_file_name(data_prefix + std::to_string(k) + ".data");
    file_list_stream << (tmp_file_name + "\n");

    // data generation;
    std::ofstream out_stream(tmp_file_name, std::ofstream::binary);

    DataWriter<CK_T> data_writer(out_stream);

    DataSetHeader header = {1, num_records_per_file, label_dim, dense_dim, slot_num, 0, 0, 0};

    data_writer.append(reinterpret_cast<char*>(&header), sizeof(DataSetHeader));
    data_writer.write();

    for (int i = 0; i < num_records_per_file; i++) {
      IntUniformDataSimulator<int> idata_sim(1, max_nnz);  // for nnz per slot
      FloatUniformDataSimulator<float> fdata_sim(0, 1);    // for lable and dense
      for (int j = 0; j < label_dim + dense_dim; j++) {
        float label_dense = fdata_sim.get_num();
        data_writer.append(reinterpret_cast<char*>(&label_dense), sizeof(float));
      }
      size_t offset = 0;
      for (int k = 0; k < slot_num; k++) {
        // int nnz = idata_sim.get_num();
        int nnz = max_nnz;  // for test
        data_writer.append(reinterpret_cast<char*>(&nnz), sizeof(int));
        size_t slot_size = slot_sizes[k];
        std::shared_ptr<IDataSimulator<T>> ldata_sim;
        if (long_tail)
          ldata_sim.reset(new IntPowerLawDataSimulator<T>(0, slot_size - 1, alpha));
        else
          ldata_sim.reset(new IntUniformDataSimulator<T>(0, slot_size - 1));  // for key
        for (int j = 0; j < nnz; j++) {
          T key = ldata_sim->get_num() + offset;
          data_writer.append(reinterpret_cast<char*>(&key), sizeof(T));
        }
        offset += slot_size;
      }
      data_writer.write();
    }
    out_stream.close();
  }
  file_list_stream.close();
  return;
}

template <typename T = unsigned int>
inline void data_generation_for_raw(std::string file_name, long long num_samples, int label_dim,
                                    int dense_dim, float float_label_dense,
                                    const std::vector<size_t> slot_size,
                                    std::vector<int> nnz_array = std::vector<int>(),
                                    bool long_tail = false, float alpha = 0.0,
                                    std::vector<T>* generated_sparse_data = nullptr,
                                    std::vector<float>* generated_dense_data = nullptr,
                                    std::vector<float>* generated_label_data = nullptr) {
  if (file_exist(file_name)) {
    HCTR_LOG_S(INFO, WORLD) << "File (" + file_name + ") exists and it will be overwritten."
                            << std::endl;
  }
  static_assert(std::is_same<T, long long>::value || std::is_same<T, unsigned int>::value,
                "type not support");

  std::ofstream out_stream(file_name, std::ofstream::binary);
  size_t size_label_dense = float_label_dense ? sizeof(float) : sizeof(T);
  // check input

  std::vector<std::shared_ptr<IDataSimulator<long long>>> ldata_sim_vec;

  if (slot_size.size() != nnz_array.size() && !nnz_array.empty()) {
    HCTR_LOG(ERROR, WORLD, "Error: slot_size.size() != nnz_array.size() && !nnz_array.empty()\n");
    exit(-1);
  }

  for (auto& voc : slot_size) {
    if (long_tail) {
      ldata_sim_vec.emplace_back(new IntPowerLawDataSimulator<long long>(0, voc - 1, alpha));
    } else {
      ldata_sim_vec.emplace_back(new IntUniformDataSimulator<long long>(0, voc - 1));
    }
  }

  for (long long i = 0; i < num_samples; i++) {
    for (int j = 0; j < label_dim; j++) {
      T label_int = i % 2;
      float label_float = static_cast<float>(label_int);
      char* label_ptr = float_label_dense ? reinterpret_cast<char*>(&label_float)
                                          : reinterpret_cast<char*>(&label_int);
      if (generated_label_data != nullptr) {
        generated_label_data->push_back(label_float);
      }
      out_stream.write(label_ptr, size_label_dense);
    }
    for (int j = 0; j < dense_dim; j++) {
      T dense_int = j;
      float dense_float = static_cast<float>(dense_int);
      char* dense_ptr = float_label_dense ? reinterpret_cast<char*>(&dense_float)
                                          : reinterpret_cast<char*>(&dense_int);
      if (generated_dense_data != nullptr) {
        generated_dense_data->push_back(dense_float);
      }
      out_stream.write(dense_ptr, size_label_dense);
    }

    for (size_t j = 0; j < ldata_sim_vec.size(); j++) {
      int nnz = 1;
      if (!nnz_array.empty()) {
        nnz = nnz_array[j];
      }
      for (int k = 0; k < nnz; k++) {
        long long num_tmp = ldata_sim_vec[j]->get_num();
        T sparse =
            num_tmp > std::numeric_limits<T>::max() ? std::numeric_limits<T>::max() : num_tmp;
        if (generated_sparse_data != nullptr) {
          generated_sparse_data->push_back(sparse);
        }
        out_stream.write(reinterpret_cast<char*>(&sparse), sizeof(T));
      }
    }

    // for (int j = 0; j < sparse_dim; j++) {
    //   int sparse = 0;
    //   if (slot_size.size() != 0) {
    //     std::shared_ptr<IDataSimulator<long long>> temp_sim;
    //     if (long_tail)
    //       temp_sim.reset(new IntPowerLawDataSimulator<long long>(0, (slot_size[j] - 1) < 0 ? 0 :
    //       (slot_size[j] - 1), alpha));
    //     else
    //       temp_sim.reset(new IntUniformDataSimulator<long long>(
    //         0, (slot_size[j] - 1) < 0 ? 0 : (slot_size[j] - 1)));  // range = [0, slot_size[j])
    //     long long num_ = temp_sim->get_num();
    //     sparse = num_ > std::numeric_limits<int>::max() ? std::numeric_limits<int>::max() : num_;
    //   } else {
    //     sparse = j;
    //   }
    //   out_stream.write(reinterpret_cast<char*>(&sparse), sizeof(int));
    // }
  }
  out_stream.close();
  return;
}

struct DataGeneratorParams {
  DataReaderType_t format;
  int label_dim;
  int dense_dim;
  int num_slot;
  bool i64_input_key;
  std::string source;
  std::string eval_source;
  std::vector<size_t> slot_size_array;
  std::vector<int> nnz_array;
  Check_t check_type;
  Distribution_t dist_type;
  PowerLaw_t power_law_type;
  float alpha;
  int num_files;
  int eval_num_files;
  int num_samples_per_file;
  int num_samples;
  int eval_num_samples;
  bool float_label_dense;
  DataGeneratorParams(DataReaderType_t format, int label_dim, int dense_dim, int num_slot,
                      bool i64_input_key, const std::string& source, const std::string& eval_source,
                      const std::vector<size_t>& slot_size_array, const std::vector<int>& nnz_array,
                      Check_t check_type, Distribution_t dist_type, PowerLaw_t power_law_type,
                      float alpha, int num_files, int eval_num_files, int num_samples_per_file,
                      int num_samples, int eval_num_samples, bool float_label_dense);
};

class DataGenerator {
 public:
  ~DataGenerator();
  DataGenerator(const DataGeneratorParams& data_generator_params);
  DataGenerator(const DataGenerator&) = delete;
  DataGenerator& operator=(const DataGenerator&) = delete;
  void generate();

 private:
  DataGeneratorParams data_generator_params_;
};

}  // namespace HugeCTR
