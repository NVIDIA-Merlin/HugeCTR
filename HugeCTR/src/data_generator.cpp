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

#include <data_generator.hpp>
#include <fstream>
#include <ios>
#include <iostream>
#include <parser.hpp>
#include <sstream>
#include <unordered_set>
#include <utils.hpp>
#include <vector>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

using namespace HugeCTR;
using HugeCTR::Logger;

DataGeneratorParams::DataGeneratorParams(
    DataReaderType_t format, int label_dim, int dense_dim, int num_slot, bool i64_input_key,
    const std::string& source, const std::string& eval_source,
    const std::vector<size_t>& slot_size_array, const std::vector<int>& nnz_array,
    Check_t check_type, Distribution_t dist_type, PowerLaw_t power_law_type, float alpha,
    int num_files, int eval_num_files, int num_samples_per_file, int num_samples,
    int eval_num_samples, bool float_label_dense, int num_threads)
    : format(format),
      label_dim(label_dim),
      dense_dim(dense_dim),
      num_slot(num_slot),
      i64_input_key(i64_input_key),
      source(source),
      eval_source(eval_source),
      slot_size_array(slot_size_array),
      nnz_array(nnz_array),
      check_type(check_type),
      dist_type(dist_type),
      power_law_type(power_law_type),
      alpha(alpha),
      num_files(num_files),
      eval_num_files(eval_num_files),
      num_samples_per_file(num_samples_per_file),
      num_samples(num_samples),
      eval_num_samples(eval_num_samples),
      float_label_dense(float_label_dense),
      num_threads(num_threads) {
  if (this->nnz_array.size() == 0) {
    this->nnz_array.assign(num_slot, 1);
  }
  if (slot_size_array.size() != static_cast<size_t>(num_slot)) {
    HCTR_OWN_THROW(Error_t::WrongInput, "slot_size_array.size() should be equal to num_slot");
  }
  if (this->nnz_array.size() != static_cast<size_t>(num_slot)) {
    HCTR_OWN_THROW(Error_t::WrongInput, "nnz_array.size() should be equal to num_slot");
  }
  if (dist_type == Distribution_t::PowerLaw && power_law_type == PowerLaw_t::Specific &&
      (alpha <= 0 || abs(alpha - 1.0) < 1e-6)) {
    HCTR_OWN_THROW(
        Error_t::WrongInput,
        "alpha should be greater than zero and should not equal to 1.0 for power law distribution");
  }
  if (this->num_threads < 1) {
    HCTR_OWN_THROW(Error_t::WrongInput, "must have num_threads at least 1");
  }
}

DataGenerator::~DataGenerator() {}

DataGenerator::DataGenerator(const DataGeneratorParams& data_generator_params)
    : data_generator_params_(data_generator_params) {
  if (data_generator_params_.format == DataReaderType_t::Raw) {
    data_generator_params_.check_type = Check_t::None;
  }
}

void DataGenerator::generate() {
  bool use_long_tail = (data_generator_params_.dist_type == Distribution_t::PowerLaw);
  float alpha = 0.0;
  std::string train_data_folder = extract_dir(data_generator_params_.source);
  std::string eval_data_folder = extract_dir(data_generator_params_.eval_source);
  if (use_long_tail) {
    switch (data_generator_params_.power_law_type) {
      case PowerLaw_t::Long: {
        alpha = 0.9;
        break;
      }
      case PowerLaw_t::Medium: {
        alpha = 1.1;
        break;
      }
      case PowerLaw_t::Short: {
        alpha = 1.3;
        break;
      }
      case PowerLaw_t::Specific: {
        alpha = data_generator_params_.alpha;
        break;
      }
      default: {
        assert(!"Error: no such option && should never get here!");
        break;
      }
    }
  }
  switch (data_generator_params_.format) {
    case DataReaderType_t::Norm: {
      HCTR_LOG_S(INFO, WORLD) << "Generate Norm dataset" << std::endl;
      HCTR_LOG_S(INFO, WORLD) << "train data folder: " << train_data_folder
                              << ", eval data folder: " << eval_data_folder << ", slot_size_array: "
                              << vec_to_string(data_generator_params_.slot_size_array)
                              << ", nnz array: " << vec_to_string(data_generator_params_.nnz_array)
                              << ", #files for train: " << data_generator_params_.num_files
                              << ", #files for eval: " << data_generator_params_.eval_num_files
                              << ", #threads: " << data_generator_params_.num_threads
                              << ", #samples per file: "
                              << data_generator_params_.num_samples_per_file
                              << ", Use power law distribution: " << use_long_tail
                              << ", alpha of power law: " << alpha << std::endl;
      check_make_dir(train_data_folder);
      check_make_dir(eval_data_folder);
      if (data_generator_params_.check_type == Check_t::Sum) {
        if (data_generator_params_.i64_input_key) {
          data_generation_for_test2<long long, Check_t::Sum>(
              data_generator_params_.source, train_data_folder + "/train/gen_",
              data_generator_params_.num_files, data_generator_params_.num_samples_per_file,
              data_generator_params_.num_slot, data_generator_params_.slot_size_array,
              data_generator_params_.label_dim, data_generator_params_.dense_dim,
              data_generator_params_.nnz_array, data_generator_params_.num_threads, use_long_tail,
              alpha);
          data_generation_for_test2<long long, Check_t::Sum>(
              data_generator_params_.eval_source, eval_data_folder + "/val/gen_",
              data_generator_params_.eval_num_files, data_generator_params_.num_samples_per_file,
              data_generator_params_.num_slot, data_generator_params_.slot_size_array,
              data_generator_params_.label_dim, data_generator_params_.dense_dim,
              data_generator_params_.nnz_array, data_generator_params_.num_threads, use_long_tail,
              alpha);
        } else {
          data_generation_for_test2<unsigned int, Check_t::Sum>(
              data_generator_params_.source, train_data_folder + "/train/gen_",
              data_generator_params_.num_files, data_generator_params_.num_samples_per_file,
              data_generator_params_.num_slot, data_generator_params_.slot_size_array,
              data_generator_params_.label_dim, data_generator_params_.dense_dim,
              data_generator_params_.nnz_array, data_generator_params_.num_threads, use_long_tail,
              alpha);
          data_generation_for_test2<unsigned int, Check_t::Sum>(
              data_generator_params_.eval_source, eval_data_folder + "/val/gen_",
              data_generator_params_.eval_num_files, data_generator_params_.num_samples_per_file,
              data_generator_params_.num_slot, data_generator_params_.slot_size_array,
              data_generator_params_.label_dim, data_generator_params_.dense_dim,
              data_generator_params_.nnz_array, data_generator_params_.num_threads, use_long_tail,
              alpha);
        }
      } else {
        if (data_generator_params_.i64_input_key) {
          data_generation_for_test2<long long, Check_t::None>(
              data_generator_params_.source, train_data_folder + "/train/gen_",
              data_generator_params_.num_files, data_generator_params_.num_samples_per_file,
              data_generator_params_.num_slot, data_generator_params_.slot_size_array,
              data_generator_params_.label_dim, data_generator_params_.dense_dim,
              data_generator_params_.nnz_array, data_generator_params_.num_threads, use_long_tail,
              alpha);
          data_generation_for_test2<long long, Check_t::None>(
              data_generator_params_.eval_source, eval_data_folder + "/val/gen_",
              data_generator_params_.eval_num_files, data_generator_params_.num_samples_per_file,
              data_generator_params_.num_slot, data_generator_params_.slot_size_array,
              data_generator_params_.label_dim, data_generator_params_.dense_dim,
              data_generator_params_.nnz_array, data_generator_params_.num_threads, use_long_tail,
              alpha);
        } else {
          data_generation_for_test2<unsigned int, Check_t::None>(
              data_generator_params_.source, train_data_folder + "/train/gen_",
              data_generator_params_.num_files, data_generator_params_.num_samples_per_file,
              data_generator_params_.num_slot, data_generator_params_.slot_size_array,
              data_generator_params_.label_dim, data_generator_params_.dense_dim,
              data_generator_params_.nnz_array, data_generator_params_.num_threads, use_long_tail,
              alpha);
          data_generation_for_test2<unsigned int, Check_t::None>(
              data_generator_params_.eval_source, eval_data_folder + "/val/gen_",
              data_generator_params_.eval_num_files, data_generator_params_.num_samples_per_file,
              data_generator_params_.num_slot, data_generator_params_.slot_size_array,
              data_generator_params_.label_dim, data_generator_params_.dense_dim,
              data_generator_params_.nnz_array, data_generator_params_.num_threads, use_long_tail,
              alpha);
        }
      }
      break;
    }
    case DataReaderType_t::Raw: {
      HCTR_LOG_S(INFO, WORLD) << "Generate Raw dataset" << std::endl;
      HCTR_LOG_S(INFO, WORLD) << "train data folder: " << train_data_folder
                              << ", eval data folder: " << eval_data_folder << ", slot_size_array: "
                              << vec_to_string(data_generator_params_.slot_size_array)
                              << ", nnz array: " << vec_to_string(data_generator_params_.nnz_array)
                              << ", Number of train samples: " << data_generator_params_.num_samples
                              << ", Number of eval samples: "
                              << data_generator_params_.eval_num_samples
                              << ", Use power law distribution: " << use_long_tail
                              << ", alpha of power law: " << alpha << std::endl;
      check_make_dir(train_data_folder);
      check_make_dir(eval_data_folder);
      if (data_generator_params_.i64_input_key) {
        data_generation_for_raw<long long>(
            data_generator_params_.source, data_generator_params_.num_samples,
            data_generator_params_.label_dim, data_generator_params_.dense_dim,
            data_generator_params_.float_label_dense, data_generator_params_.slot_size_array,
            data_generator_params_.nnz_array, use_long_tail, alpha);
        data_generation_for_raw<long long>(
            data_generator_params_.eval_source, data_generator_params_.eval_num_samples,
            data_generator_params_.label_dim, data_generator_params_.dense_dim,
            data_generator_params_.float_label_dense, data_generator_params_.slot_size_array,
            data_generator_params_.nnz_array, use_long_tail, alpha);
      } else {
        data_generation_for_raw<unsigned int>(
            data_generator_params_.source, data_generator_params_.num_samples,
            data_generator_params_.label_dim, data_generator_params_.dense_dim,
            data_generator_params_.float_label_dense, data_generator_params_.slot_size_array,
            data_generator_params_.nnz_array, use_long_tail, alpha);
        data_generation_for_raw<unsigned int>(
            data_generator_params_.eval_source, data_generator_params_.eval_num_samples,
            data_generator_params_.label_dim, data_generator_params_.dense_dim,
            data_generator_params_.float_label_dense, data_generator_params_.slot_size_array,
            data_generator_params_.nnz_array, use_long_tail, alpha);
      }
      break;
    }
    case DataReaderType_t::Parquet: {
#ifdef DISABLE_CUDF
      HCTR_OWN_THROW(Error_t::WrongInput, "Parquet is not supported under DISABLE_CUDF");
#else
      HCTR_LOG_S(INFO, WORLD) << "Generate Parquet dataset" << std::endl;
      HCTR_LOG_S(INFO, WORLD) << "train data folder: " << train_data_folder
                              << ", eval data folder: " << eval_data_folder << ", slot_size_array: "
                              << vec_to_string(data_generator_params_.slot_size_array)
                              << ", nnz array: " << vec_to_string(data_generator_params_.nnz_array)
                              << ", #files for train: " << data_generator_params_.num_files
                              << ", #files for eval: " << data_generator_params_.eval_num_files
                              << ", #samples per file: "
                              << data_generator_params_.num_samples_per_file
                              << ", Use power law distribution: " << use_long_tail
                              << ", alpha of power law: " << alpha << std::endl;

      check_make_dir(train_data_folder);
      check_make_dir(eval_data_folder);
      if (data_generator_params_.i64_input_key) {  // I64 = long long
        data_generation_for_parquet<int64_t>(
            data_generator_params_.source, train_data_folder + "/train/gen_",
            data_generator_params_.num_files, data_generator_params_.num_samples_per_file,
            data_generator_params_.num_slot, data_generator_params_.label_dim,
            data_generator_params_.dense_dim, data_generator_params_.slot_size_array,
            data_generator_params_.nnz_array, use_long_tail, alpha);

        data_generation_for_parquet<int64_t>(
            data_generator_params_.eval_source, eval_data_folder + "/val/gen_",
            data_generator_params_.eval_num_files, data_generator_params_.num_samples_per_file,
            data_generator_params_.num_slot, data_generator_params_.label_dim,
            data_generator_params_.dense_dim, data_generator_params_.slot_size_array,
            data_generator_params_.nnz_array, use_long_tail, alpha);
      } else {  // I32 = unsigned int
        data_generation_for_parquet<unsigned int>(
            data_generator_params_.source, train_data_folder + "/train/gen_",
            data_generator_params_.num_files, data_generator_params_.num_samples_per_file,
            data_generator_params_.num_slot, data_generator_params_.label_dim,
            data_generator_params_.dense_dim, data_generator_params_.slot_size_array,
            data_generator_params_.nnz_array, use_long_tail, alpha);
        data_generation_for_parquet<unsigned int>(
            data_generator_params_.eval_source, eval_data_folder + "/val/gen_",
            data_generator_params_.eval_num_files, data_generator_params_.num_samples_per_file,
            data_generator_params_.num_slot, data_generator_params_.label_dim,
            data_generator_params_.dense_dim, data_generator_params_.slot_size_array,
            data_generator_params_.nnz_array, use_long_tail, alpha);
      }
#endif
      break;
    }

    default: {
      assert(!"Error: no such option && should never get here!");
      break;
    }
  }
}
