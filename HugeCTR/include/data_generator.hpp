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

#include <sys/stat.h>
#include <common.hpp>
#include <fstream>
#include <random>
#include <memory>

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


template<typename T>
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
class IntUniformDataSimulator: public IDataSimulator<T> {
 public:
  IntUniformDataSimulator(T min, T max) : gen_(std::random_device()()), dis_(min, max) {}

  T get_num() override { return dis_(gen_); }

 private:
  std::mt19937 gen_;
  std::uniform_int_distribution<T> dis_;
};

template <typename T>
class IntPowerLawDataSimulator: public IDataSimulator<T> {
  public:
    IntPowerLawDataSimulator(T min, T max, float alpha): gen_(std::random_device()()), dis_(0, 1), alpha_(alpha){
      min_ = 1;
      max_ = max - min + 1;
      offset_ = min - 1; //to handle the case min_ <= 0 and alpha_ < -1
    }
    
    T get_num() override {
      double x = dis_(gen_);
      double y = (pow( (pow(max_, alpha_+1) - pow(min_, alpha_+1)) * x + pow(min_, alpha_+1) , 1.0/(alpha_+1.0)) );
      return static_cast<T>(round(y) + offset_);
    }
  
  private:
  
    std::mt19937 gen_;
    std::uniform_real_distribution<float> dis_;
    float alpha_;
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
                              int label_dim, int dense_dim, int max_nnz, bool long_tail = false, float alpha = 0.0) {
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
          while ((key % static_cast<T>(slot_num)) !=
                 static_cast<T>(k)) {  // guarantee the key belongs to the current slot_id(=k)
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
  std::cout << file_list_name << " done!" << std::endl;
  return;
}


  /**
   * Adding finer control e.g.: --voc-size-array=312,32,231234,124332,4554 --nnz-array=23,3,45,66,23,1,1,1,1
   */
template <typename T, Check_t CK_T>
void data_generation_for_test2(std::string file_list_name, std::string data_prefix, int num_files,
                              int num_records_per_file, int slot_num, std::vector<size_t> voc_size_array,
                              int label_dim, int dense_dim, std::vector<int> nnz_array, bool long_tail = false, float alpha = 0.0) {

  //check if slot_num == voc_size_array.size == nnz_array.size
  if(slot_num != (int)voc_size_array.size() || slot_num != (int)nnz_array.size()){
    std::cout << "Error: slot_num != voc_size_array.size() || slot_num != nnz_array.size()" << std::endl;
    exit(-1);
  }

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
    // Initialize Simulators
    FloatUniformDataSimulator<float> fdata_sim(0, 1);              // for lable and dense
    std::vector<std::shared_ptr<IDataSimulator<T>>> ldata_sim_vec;
    size_t accum = 0;
    //todo risk of type Int
    for(auto& voc: voc_size_array){
      size_t accum_next = accum+voc; 
      if(long_tail){
	ldata_sim_vec.emplace_back(new IntPowerLawDataSimulator<T>(accum, accum_next - 1, alpha));
      }
      else{
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
      IntUniformDataSimulator<int> idata_sim(1, max_nnz);            // for nnz
      FloatUniformDataSimulator<float> fdata_sim(0, 1);              // for lable and dense
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
inline void data_generation_for_raw(
    std::string file_name, long long num_samples, int label_dim, int dense_dim,
    float float_label_dense,
    const std::vector<size_t> slot_size, std::vector<int> nnz_array = std::vector<int>(),
    bool long_tail = false, float alpha = 0.0) {

  static_assert(std::is_same<T, long long>::value || std::is_same<T, unsigned int>::value,
		"type not support");


  std::ofstream out_stream(file_name, std::ofstream::binary);
  size_t size_label_dense = float_label_dense ? sizeof(float) : sizeof(T);
  //check input

  std::vector<std::shared_ptr<IDataSimulator<long long>>> ldata_sim_vec;
  
  if(slot_size.size() != nnz_array.size() && !nnz_array.empty()){
    std::cout << "Error: slot_size.size() != nnz_array.size() && !nnz_array.empty()" << std::endl;
    exit(-1);
  }

  for(auto& voc: slot_size){
    if(long_tail){
      ldata_sim_vec.emplace_back(new IntPowerLawDataSimulator<long long>(0, voc-1, alpha));
    }
    else{
      ldata_sim_vec.emplace_back(new IntUniformDataSimulator<long long>(0, voc-1));
    }
  }

  for (long long i = 0; i < num_samples; i++) {
    for (int j = 0; j < label_dim; j++) {
      T label_int = i % 2;
      float label_float = static_cast<float>(label_int);
      char* label_ptr = float_label_dense ? reinterpret_cast<char*>(&label_float)
                                          : reinterpret_cast<char*>(&label_int);
      out_stream.write(label_ptr, size_label_dense);
    }
    for (int j = 0; j < dense_dim; j++) {
      T dense_int = j;
      float dense_float = static_cast<float>(dense_int);
      char* dense_ptr = float_label_dense ? reinterpret_cast<char*>(&dense_float)
                                          : reinterpret_cast<char*>(&dense_int);
      out_stream.write(dense_ptr, size_label_dense);
    }
    

    for (size_t j = 0; j < ldata_sim_vec.size(); j++) {
      int nnz = 1;
      if(!nnz_array.empty()){
	nnz = nnz_array[j];
      }
      for(int k = 0; k < nnz; k ++){
	long long num_tmp = ldata_sim_vec[j]->get_num();
	T sparse = num_tmp > std::numeric_limits<T>::max() ? std::numeric_limits<T>::max() : num_tmp;
	out_stream.write(reinterpret_cast<char*>(&sparse), sizeof(T));
      }
    }

    // for (int j = 0; j < sparse_dim; j++) {
    //   int sparse = 0;
    //   if (slot_size.size() != 0) {
    //     std::shared_ptr<IDataSimulator<long long>> temp_sim;
    //     if (long_tail)
    //       temp_sim.reset(new IntPowerLawDataSimulator<long long>(0, (slot_size[j] - 1) < 0 ? 0 : (slot_size[j] - 1), alpha));
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
}  // namespace HugeCTR
