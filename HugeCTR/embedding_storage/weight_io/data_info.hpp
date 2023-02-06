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

#include <common.hpp>
#include <core/datatype.hpp>
#include <optimizer.hpp>
#include <unordered_map>

#define FileHeadLength 32
#define FileHeadNbytes 128
#define MetaDataHeadLength 20
#define MetaDataValidLength 36

using embeddingFilter = std::function<bool(size_t)>;

namespace embedding {

template <typename LambdaFilter>
struct LoadFilter {
  bool filter(size_t key) { return filter_(key); }
  LambdaFilter filter_;
};

template <typename LambdaFilter>
LoadFilter<LambdaFilter> make_LoadFilter(LambdaFilter filter) {
  return {filter};
};

enum class SparseFSType { AUTO, MPI, FS };

enum class EmbeddingFileType { Key, Weight, Optimizer };

struct GlobalEmbeddingDistribution {
 public:
  /*
     distribute_array_ is a two dimension array , store every gpu take key num in every tables, just
     like below: gpu_0 gpu_1 gpu_2

     table_0  10    0     10

     table_1  100   100   100

     table_2  0     0     50

     parellel_array_ is a one dimension array , store every table's parellel mode ,1=data parallel
     ,2=model_parallel,3=hybrid_parallel
  */
  GlobalEmbeddingDistribution(size_t gpu_num, size_t table_num)
      : gpu_num_(gpu_num), table_num_(table_num) {
    // distribute_array_ = new size_t[gpu_num * table_num];
    // parellel_array_ = new size_t[table_num];
    distribute_array_ = std::vector<size_t>(gpu_num * table_num, 0);
    parellel_array_ = std::vector<size_t>(table_num, 0);
    // memset(distribute_array_, 0, sizeof(size_t) * gpu_num * table_num);
    // memset(parellel_array_, 0, sizeof(size_t) * table_num);
  }

  size_t get(size_t gpu_id, size_t table_id) const {
    if (gpu_id < 0 || gpu_id >= gpu_num_ || table_id < 0 || table_id >= table_num_) {
      HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError, "gpu_id or table_id is out of range");
    }
    return distribute_array_[table_id * gpu_num_ + gpu_id];
  }

  size_t get_table_keynum(size_t table_id) const {
    size_t key_num_sum = 0;
    if (parellel_array_[table_id] == 1) {
      key_num_sum = distribute_array_[table_id * gpu_num_];
    } else if (parellel_array_[table_id] == 2) {
      for (size_t gpu_id = 0; gpu_id < gpu_num_; ++gpu_id) {
        key_num_sum += distribute_array_[table_id * gpu_num_ + gpu_id];
      }
    }
    return key_num_sum;
  }

  void set(size_t value, size_t gpu_id, size_t table_id) {
    if (gpu_id < 0 || gpu_id >= gpu_num_ || table_id < 0 || table_id >= table_num_) {
      HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError, "gpu_id or table_id is out of range");
    }
    distribute_array_[table_id * gpu_num_ + gpu_id] = value;
    return;
  }

  void set_parallel(size_t table_id, size_t mode) {
    if (table_id < 0 || table_id >= table_num_) {
      HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError, "table_id is out of range");
    }
    parellel_array_[table_id] = mode;
    return;
  }

  size_t get_parallel(size_t table_id) const {
    if (table_id < 0 || table_id >= table_num_) {
      HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError, "table_id is out of range");
    }

    return parellel_array_[table_id];
  }

  size_t* get_buffer() const { return (size_t*)distribute_array_.data(); }

  void print_info() const {
    std::cout << "Table distribute:" << std::endl;
    for (size_t gpu_id = 0; gpu_id < gpu_num_; ++gpu_id) {
      std::cout << "global gpu id:" << gpu_id;
      for (size_t table_id = 0; table_id < table_num_; ++table_id) {
        std::cout << "table" << table_id << ":" << distribute_array_[table_id * gpu_num_ + gpu_id]
                  << ",";
      }
      std::cout << "." << std::endl;
      ;
    }
    std::cout << "Table distribute end" << std::endl;

    std::cout << "Table info:" << std::endl;

    for (size_t table_id = 0; table_id < table_num_; ++table_id) {
      if (parellel_array_[table_id] == 1) {
        std::cout << "table" << table_id << ": data parallel" << std::endl;
      } else if (parellel_array_[table_id] == 2) {
        std::cout << "table" << table_id << ": model parallel" << std::endl;
      } else if (parellel_array_[table_id] == 3) {
        std::cout << "table" << table_id << ": hybrid parallel" << std::endl;
      }
    }
    std::cout << "Table info end" << std::endl;
  }

  ~GlobalEmbeddingDistribution() {
    // delete[] distribute_array_;
    // delete[] parellel_array_;
  }

 private:
  size_t gpu_num_;
  size_t table_num_;
  std::vector<size_t> distribute_array_;
  std::vector<size_t> parellel_array_;
};

struct EmbeddingParameterInfo {
  // for train and inference dump and load
  std::string parameter_folder_path;    // parameter saved folder path
  int table_nums = 0;                   // number of all table
  core::DataType key_type;              // data type of key
  core::DataType embedding_value_type;  // data type of embedding_value_type
  HugeCTR::OptParams optimizer_type;    // type of optimizer_type
  int max_embedding_vector_length;      // max_length of all embedding_vector_length
  std::vector<int> table_ids;
  std::unordered_map<int, size_t> table_key_nums;  // store all table's key numbers
  std::unordered_map<int, size_t>
      table_embedding_vector_lengths;  // store all table's embedding vector lengths

  // only for train dump,other parts don't need this variable
  int embedding_collection_id = 0;
  std::shared_ptr<struct GlobalEmbeddingDistribution> gemb_distribution;
};

}  // namespace embedding
