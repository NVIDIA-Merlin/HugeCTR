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

#include <sys/stat.h>
#include <common.hpp>
#include <fstream>
#include <random>

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
      std::cout << (finalpath + " exist") << std::endl;
    } else {
      // something else
      std::cerr << ("cannot create" + finalpath + ": unexpected error") << std::endl;
    }
  }
}

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
class IntUniformDataSimulator {
 public:
  IntUniformDataSimulator(T min, T max) : gen_(std::random_device()()), dis_(min, max) {}

  T get_num() { return dis_(gen_); }

 private:
  std::mt19937 gen_;
  std::uniform_int_distribution<T> dis_;
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
                              int label_dim, int dense_dim, int max_nnz) {
  if (file_exist(file_list_name)) {
    std::cout << "File (" + file_list_name +
                     ") exist. To generate new dataset plesae remove this file."
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
    std::cout << tmp_file_name << std::endl;
    // data generation;
    std::ofstream out_stream(tmp_file_name, std::ofstream::binary);

    DataWriter<CK_T> data_writer(out_stream);

    DataSetHeader header = {
        Checker_Traits<CK_T>::ID(), num_records_per_file, label_dim, dense_dim, slot_num, 0, 0, 0};

    data_writer.append(reinterpret_cast<char*>(&header), sizeof(DataSetHeader));
    data_writer.write();

    for (int i = 0; i < num_records_per_file; i++) {
      IntUniformDataSimulator<int> idata_sim(1, max_nnz);            // for nnz
      FloatUniformDataSimulator<float> fdata_sim(0, 1);              // for lable and dense
      IntUniformDataSimulator<T> ldata_sim(0, vocabulary_size - 1);  // for key
      for (int j = 0; j < label_dim + dense_dim; j++) {
        float label_dense = fdata_sim.get_num();
        data_writer.append(reinterpret_cast<char*>(&label_dense), sizeof(float));
      }
      for (int k = 0; k < slot_num; k++) {
        int nnz = idata_sim.get_num();
        data_writer.append(reinterpret_cast<char*>(&nnz), sizeof(int));
        for (int j = 0; j < nnz; j++) {
          T key = ldata_sim.get_num();
          while ((key % static_cast<T>(slot_num)) !=
                 static_cast<T>(k)) {  // guarantee the key belongs to the current slot_id(=k)
            key = ldata_sim.get_num();
          }
          data_writer.append(reinterpret_cast<char*>(&key), sizeof(T));
        }
      }
      data_writer.write();
    }
    out_stream.close();
  }
  file_list_stream.close();
  std::cout << file_list_name << " done!" << std::endl;
  return;
}

// Add a new data_generation function for LocalizedSparseEmbedding testing
// In this function, the relationship between key and slot_id is: key's slot_id=(key%slot_num)
// Add a new data_generation function for LocalizedSparseEmbedding testing
// In this function, the relationship between key and slot_id is: key's slot_id=(key%slot_num)
template <typename T, Check_t CK_T>
void data_generation_for_localized_test(std::string file_list_name, std::string data_prefix,
                                        int num_files, int num_records_per_file, int slot_num,
                                        int vocabulary_size, int label_dim, int dense_dim,
                                        int max_nnz) {
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
      IntUniformDataSimulator<int> idata_sim(1, max_nnz);            // for nnz
      FloatUniformDataSimulator<float> fdata_sim(0, 1);              // for lable and dense
      IntUniformDataSimulator<T> ldata_sim(0, vocabulary_size - 1);  // for key
      for (int j = 0; j < label_dim + dense_dim; j++) {
        float label_dense = fdata_sim.get_num();
        data_writer.append(reinterpret_cast<char*>(&label_dense), sizeof(float));
      }
      for (int k = 0; k < slot_num; k++) {
        int nnz = idata_sim.get_num();
        data_writer.append(reinterpret_cast<char*>(&nnz), sizeof(int));
        for (int j = 0; j < nnz; j++) {
          T key = ldata_sim.get_num();
          while ((key % slot_num) != k) {  // guarantee the key belongs to the current slot_id(=k)
            key = ldata_sim.get_num();
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
                                        int max_nnz, const std::vector<size_t> slot_sizes) {
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
        IntUniformDataSimulator<T> ldata_sim(0, slot_size - 1);  // for key
        for (int j = 0; j < nnz; j++) {
          T key = ldata_sim.get_num() + offset;
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

inline void data_generation_for_raw(
    std::string file_name, long long num_samples, int label_dim = 1, int dense_dim = 13,
    int sparse_dim = 26, float float_label_dense = false,
    const std::vector<long long> slot_size = std::vector<long long>()) {
  std::ofstream out_stream(file_name, std::ofstream::binary);
  size_t size_label_dense = float_label_dense ? sizeof(float) : sizeof(int);
  for (long long i = 0; i < num_samples; i++) {
    for (int j = 0; j < label_dim; j++) {
      int label_int = i % 2;
      float label_float = static_cast<float>(label_int);
      char* label_ptr = float_label_dense ? reinterpret_cast<char*>(&label_float)
                                          : reinterpret_cast<char*>(&label_int);
      out_stream.write(label_ptr, size_label_dense);
    }
    for (int j = 0; j < dense_dim; j++) {
      int dense_int = j;
      float dense_float = static_cast<float>(dense_int);
      char* dense_ptr = float_label_dense ? reinterpret_cast<char*>(&dense_float)
                                          : reinterpret_cast<char*>(&dense_int);
      out_stream.write(dense_ptr, size_label_dense);
    }
    for (int j = 0; j < sparse_dim; j++) {
      int sparse = 0;
      if (slot_size.size() != 0) {
        IntUniformDataSimulator<long long> temp_sim(
            0, (slot_size[j] - 1) < 0 ? 0 : (slot_size[j] - 1));  // range = [0, slot_size[j])
        long long num_ = temp_sim.get_num();
        sparse = num_ > std::numeric_limits<int>::max() ? std::numeric_limits<int>::max() : num_;
      } else {
        sparse = j;
      }
      out_stream.write(reinterpret_cast<char*>(&sparse), sizeof(int));
    }
  }
  out_stream.close();
  return;
}
}  // namespace HugeCTR