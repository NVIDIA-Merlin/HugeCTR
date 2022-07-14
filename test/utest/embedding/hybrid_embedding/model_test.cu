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

#include <algorithm>
#include <set>
#include <vector>

#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/utils.cuh"
#include "test/utest/embedding/hybrid_embedding/data_test.hpp"
#include "test/utest/embedding/hybrid_embedding/input_generator.hpp"
#include "test/utest/embedding/hybrid_embedding/statistics_test.hpp"

using namespace HugeCTR;
using namespace hybrid_embedding;

namespace {

template <typename dtype>
void print_vector(const std::vector<dtype> &vec) {
  for (auto v : vec) {
    std::cout << v << " ,";
  }
  std::cout << std::endl;
}

template <typename dtype, typename emtype = float>
void model_test() {
  size_t batch_size = 4;
  size_t num_iterations = 2;
  CommunicationType comm_type = CommunicationType::IB_NVLink;
  uint32_t global_instance_id = 1;
  std::vector<uint32_t> num_instances_per_node{2, 2};
  std::vector<size_t> table_sizes{100, 10, 10, 20};
  std::vector<dtype> data_in{99, 3, 7, 19, 0,  0, 0, 0,  1, 1, 1, 1, 2, 2, 2, 2,
                             3,  3, 3, 3,  50, 2, 4, 10, 2, 2, 2, 2, 1, 1, 1, 1};
  std::vector<dtype> data_to_unique_categories_ref{
      99, 103, 117, 139, 0,  100, 110, 120, 1, 101, 111, 121, 2, 102, 112, 122,
      3,  103, 113, 123, 50, 102, 114, 130, 2, 102, 112, 122, 1, 101, 111, 121};

  Tensor2<dtype> d_data_in;
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();
  buff->reserve({batch_size * num_iterations * table_sizes.size()}, &d_data_in);
  buff->allocate();
  upload_tensor(data_in, d_data_in, 0);

  /*1. Data() and data.data_to_unique_categories()*/
  Data<dtype> data(table_sizes, batch_size, num_iterations);
  data.data_to_unique_categories(d_data_in, 0);
  std::vector<dtype> data_to_unique_categories_ret;
  download_tensor(data_to_unique_categories_ret, data.samples, 0);
  EXPECT_THAT(data_to_unique_categories_ret,
              ::testing::ElementsAreArray(data_to_unique_categories_ref));

  /*2. Model()*/
  size_t num_categories = EmbeddingTableFunctors<dtype>::get_num_categories(table_sizes);
  // std::cout << "debug0:" << num_categories << std::endl;
  Model<dtype> model(comm_type, global_instance_id, num_instances_per_node, num_categories);

  /*3. CalibrationData()*/
  size_t num_nodes = num_instances_per_node.size();
  CalibrationData calibration_data(num_nodes, 1.0 / 10.0, 4.0, 1.0, 1.0);

  /*4. Statistics()*/
  Statistics<dtype> statistics(data.batch_size * data.num_iterations * data.table_sizes.size(),
                               data.table_sizes.size(), model.num_instances, num_categories);
  statistics.sort_categories_by_count(data.samples, 0);
  std::vector<dtype> categories_sorted_ret;
  std::vector<uint32_t> counts_sorted_ret;
  download_tensor(categories_sorted_ret, statistics.categories_sorted, 0);
  download_tensor(counts_sorted_ret, statistics.counts_sorted, 0);
  std::vector<dtype> categories_sorted_ref{102, 1,  2,   101, 103, 111, 112, 121, 122, 0,   3,
                                           50,  99, 100, 110, 113, 114, 117, 120, 123, 130, 139};
  std::vector<uint32_t> counts_sorted_ref{3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1,
                                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  EXPECT_THAT(categories_sorted_ret, ::testing::ElementsAreArray(categories_sorted_ref));
  EXPECT_THAT(counts_sorted_ret, ::testing::ElementsAreArray(counts_sorted_ref));
  // print_vector(counts_sorted_ret);
  // print_vector(categories_sorted_ret);

  /*5. Model<dtype>::init_hybrid_model*/
  model.init_hybrid_model(calibration_data, statistics, data, 0);
  EXPECT_EQ(model.num_frequent, 12);
  std::vector<dtype> category_location_ret;
  download_tensor(category_location_ret, model.category_location, 0);

  std::vector<dtype> category_location_ref{
      4, 4,  4, 3,  4, 6,  4, 7,  0, 0,  1, 0,  2, 0,  3, 0,  0, 1,  1, 1,  2,   1,  3, 1,  0, 2,
      1, 2,  2, 2,  3, 2,  0, 3,  1, 3,  2, 3,  3, 3,  0, 4,  1, 4,  2, 4,  3,   4,  0, 5,  1, 5,
      2, 5,  3, 5,  0, 6,  1, 6,  2, 6,  3, 6,  0, 7,  1, 7,  2, 7,  3, 7,  0,   8,  1, 8,  2, 8,
      3, 8,  0, 9,  1, 9,  2, 9,  3, 9,  0, 10, 1, 10, 2, 10, 3, 10, 0, 11, 1,   11, 4, 9,  2, 11,
      3, 11, 0, 12, 1, 12, 2, 12, 3, 12, 0, 13, 1, 13, 2, 13, 3, 13, 0, 14, 1,   14, 2, 14, 3, 14,
      0, 15, 1, 15, 2, 15, 3, 15, 0, 16, 1, 16, 2, 16, 3, 16, 0, 17, 1, 17, 2,   17, 3, 17, 0, 18,
      1, 18, 2, 18, 3, 18, 0, 19, 1, 19, 2, 19, 3, 19, 0, 20, 1, 20, 2, 20, 3,   20, 0, 21, 1, 21,
      2, 21, 3, 21, 0, 22, 1, 22, 2, 22, 3, 22, 0, 23, 1, 23, 2, 23, 3, 23, 4,   10, 4, 0,  4, 1,
      0, 24, 1, 24, 2, 24, 3, 24, 0, 25, 1, 25, 2, 25, 4, 5,  4, 8,  3, 25, 0,   26, 1, 26, 2, 26,
      3, 26, 0, 27, 1, 27, 2, 27, 4, 11, 4, 2,  3, 27, 0, 28, 1, 28, 2, 28, 3,   28, 0, 29, 1, 29,
      2, 29, 3, 29, 0, 30, 1, 30, 2, 30, 3, 30, 0, 31, 1, 31, 2, 31, 3, 31, 140, 140};
  EXPECT_THAT(category_location_ret, ::testing::ElementsAreArray(category_location_ref));

  std::vector<dtype> h_frequent_model_table_offsets_ref{0, 0, 2, 2, 3, 3, 5,  5,  6,  6,
                                                        6, 8, 8, 9, 9, 9, 10, 11, 11, 12};
  std::vector<dtype> h_infrequent_model_table_offsets_ref{0, 24, 26, 28, 32};
  EXPECT_THAT(model.h_frequent_model_table_offsets,
              ::testing::ElementsAreArray(h_frequent_model_table_offsets_ref));
  EXPECT_THAT(model.h_infrequent_model_table_offsets,
              ::testing::ElementsAreArray(h_infrequent_model_table_offsets_ref));
};

template <typename dtype>
void model_init_test(const size_t num_instances, const size_t num_tables, const size_t batch_size,
                     CommunicationType ctype) {
  // 1. generate the reference model from reference stats and corresponding data
  // std::vector<size_t> categories;
  // std::vector<size_t> counts;

  const size_t num_iterations = 1;
  std::cout << "Model init test ...  " << std::endl << std::endl;
  std::cout << "number of instances  : " << num_instances << std::endl;
  std::cout << "Number of tables     : " << num_tables << std::endl;
  std::cout << "Batch size           : " << batch_size << std::endl;
  std::cout << "Number of iterations : " << num_iterations << std::endl;

  HybridEmbeddingInputGenerator<dtype> input_generator(848484);
  std::vector<dtype> raw_data = input_generator.generate_categorical_input(batch_size, num_tables);
  std::vector<size_t> table_sizes = input_generator.get_table_sizes();
  const size_t num_categories = std::accumulate(table_sizes.begin(), table_sizes.end(), 0);
  std::cout << "Table sizes          : ";
  for (size_t embedding = 0; embedding < table_sizes.size(); ++embedding)
    std::cout << '\t' << table_sizes[embedding];
  std::cout << std::endl;

  // create the gpu tensor for the raw data
  cudaStream_t stream = 0;
  Tensor2<dtype> d_raw_data;
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
  std::cout << "number of samples  : " << raw_data.size() << std::endl;
  buf->reserve({raw_data.size(), 1}, &d_raw_data);
  buf->allocate();
  upload_tensor(raw_data, d_raw_data, stream);

  std::cout << "Testing raw data..." << std::endl;
  test_raw_data(d_raw_data.get_ptr(), batch_size, num_tables, num_iterations, table_sizes);
  std::cout << "Done testing raw data..." << std::endl;

  // 2. perform model initialization, data
  std::cout << "performing statistics and calibration intialization..." << std::endl;
  Data<dtype> data(table_sizes, batch_size, num_iterations);
  data.data_to_unique_categories(d_raw_data, stream);
  HCTR_LIB_THROW(cudaStreamSynchronize(stream));

  std::cout << "Testing samples..." << std::endl;
  test_samples<dtype>(d_raw_data.get_ptr(), data);
  std::cout << "Done testing samples!" << std::endl;

  Statistics<dtype> statistics(data, num_instances);
  CalibrationData calibration(1, 1. / 10., 130.e9, 190.e9, 1.0);

  //    model creation
  std::cout << "performing model intialization..." << std::endl;
  std::vector<uint32_t> num_instances_per_node(1);
  num_instances_per_node[0] = (uint32_t)num_instances;
  // Model<dtype> model(ctype, 0, num_instances_per_node, num_categories);
  //  = {(uint32_t)num_instances};
  std::vector<Model<dtype>> models;
  for (size_t instance = 0; instance < num_instances; ++instance) {
    // std::cout << "instance : " << instance << std::endl;
    // std::cout << "constructing instance, allocating memory..." << std::endl;
    models.emplace_back(ctype, (uint32_t)instance, num_instances_per_node, num_categories);
    // std::cout << "initializing model..." << std::endl;
    models[instance].init_hybrid_model(calibration, statistics, data, stream);
    // std::cout << "done initializing model" << std::endl;
  }

  std::vector<dtype> categories_sorted_stats;
  std::vector<uint32_t> counts_sorted_stats;
  download_tensor(categories_sorted_stats, statistics.categories_sorted, stream);
  download_tensor(counts_sorted_stats, statistics.counts_sorted, stream);

  // TODO: check consistency of
  //   global_instance_id,
  //   num_instances_per_node,
  //   node_id,

  // Check defining properties

  std::cout << "Checking consistency and completeness of infrequent embedding..." << std::endl;
  // check order of categories for infrequent
  // - assuming default distributed embedding
  std::vector<size_t> num_infrequent_model_vec(num_instances);
  size_t num_infrequent_tables = 0;
  for (size_t instance = 0; instance < num_instances; ++instance) {
    Model<dtype> &model = models[instance];

    std::vector<dtype> category_location;
    download_tensor(category_location, model.category_location, stream);

    size_t indx_infrequent = 0;
    for (size_t category = 0; category < num_categories; ++category) {
      if (category_location[2 * category] < num_instances) {
        size_t instance_location = category_location[2 * category];
        size_t buffer_index = category_location[2 * category + 1];

        EXPECT_EQ(instance_location, indx_infrequent % num_instances);
        EXPECT_EQ(buffer_index, indx_infrequent / num_instances);

        indx_infrequent++;
      }
    }
    const size_t num_infrequent_model = indx_infrequent;
    num_infrequent_model_vec[instance] = num_infrequent_model;

    // check consistency table offsets
    size_t num_infrequent_tables_instance = 0;
    for (size_t embedding = 0; embedding < num_tables; ++embedding) {
      size_t cur_offset = model.h_infrequent_model_table_offsets[embedding];
      size_t next_offset = model.h_infrequent_model_table_offsets[embedding + 1];
      size_t indx_infrequent_instance = 0;
      for (size_t category = 0; category < num_categories; ++category) {
        if (category_location[2 * category] == instance) {
          if (indx_infrequent_instance >= cur_offset && indx_infrequent_instance < next_offset) {
            size_t embedding_category =
                EmbeddingTableFunctors<dtype>::get_embedding_table_index(table_sizes, category);
            EXPECT_EQ(embedding_category, embedding);
          }
          indx_infrequent_instance++;
        }
      }
      num_infrequent_tables_instance = indx_infrequent_instance;
    }
    num_infrequent_tables += num_infrequent_tables_instance;
  }
  // Check that the total number of embedding vectors in all instances for all tables equals
  // the total number of infrequent embedding vectors
  if (num_infrequent_model_vec.size() > 0) {
    EXPECT_EQ(num_infrequent_tables, num_infrequent_model_vec[0]);
    if (num_infrequent_tables != num_infrequent_model_vec[0]) {
      std::cout << "num_infrequent_tables       = " << num_infrequent_tables << std::endl;
      std::cout << "num_infrequent_model_vec[0] = " << num_infrequent_model_vec[0] << std::endl;
    }
  }
  // Check that the number of infrequent categories is the same for all instances.
  for (size_t instance = 1; instance < num_instances; ++instance) {
    EXPECT_EQ(num_infrequent_model_vec[instance], num_infrequent_model_vec[0]);
  }

  std::cout << "Checking consistency and completeness of frequent embedding..." << std::endl;
  // Check that the frequent embedding model is complete and self-consistent
  //
  // - num_frequent is consistent with data and num_categories - i.e. table_sizes
  // - category_frequent_index and frequent_categories are consistent
  // - both are consistent with num_frequent
  // - table offsets frequent embedding are consistent with frequent_categories array
  //
  for (size_t instance = 0; instance < num_instances; ++instance) {
    Model<dtype> &model = models[instance];
    const size_t num_categories = model.num_categories;

    std::vector<dtype> &frequent_table_offsets = model.h_frequent_model_table_offsets;
    std::vector<dtype> category_location;
    download_tensor(category_location, model.category_location, stream);
    std::vector<dtype> frequent_categories;
    download_tensor(frequent_categories, model.frequent_categories, stream);

    // check that number of frequent categories in category_location == model.num_frequent
    size_t num_frequent_model = 0;
    for (size_t i = 0; i < num_categories; ++i) {
      num_frequent_model += (size_t)(category_location[2 * i] == num_instances ? 1 : 0);
    }
    EXPECT_EQ(num_frequent_model, model.num_frequent);

    // check that category in frequent_categories has corresponding index in category_frequent_index
    for (size_t i = 0; i < frequent_categories.size(); ++i) {
      size_t category = frequent_categories[i];
      EXPECT_EQ(category_location[2 * category + 1], i);
    }

    std::map<dtype, size_t> category_to_stats_map;
    for (size_t i = 0; i < categories_sorted_stats.size(); ++i) {
      category_to_stats_map[categories_sorted_stats[i]] = i;
    }

    // check that table offsets are consistent with the frequent_categories array
    //   - check that categories corresponding to embedding actually part of embedding
    std::set<dtype> set_categories_from_table_offsets;
    std::set<dtype> set_categories_frequent_categories_array(frequent_categories.begin(),
                                                             frequent_categories.end());
    for (size_t em_instance = 0; em_instance < num_instances; ++em_instance) {
      for (size_t embedding = 0; embedding < num_tables; ++embedding) {
        size_t cur_offset = frequent_table_offsets[em_instance * (num_tables + 1) + embedding];
        size_t next_offset = frequent_table_offsets[em_instance * (num_tables + 1) + embedding + 1];
        size_t counts_cur = 0;
        size_t counts_prev = 0;
        for (size_t frequent_category_index = cur_offset; frequent_category_index < next_offset;
             ++frequent_category_index) {
          size_t category = frequent_categories[frequent_category_index];
          size_t embedding_category =
              EmbeddingTableFunctors<dtype>::get_embedding_table_index(table_sizes, category);

          EXPECT_EQ(embedding, embedding_category);

          // find category in category_sorted_stats array
          size_t indx_stats = category_to_stats_map[category];
          counts_cur = (size_t)counts_sorted_stats[indx_stats];
          if (frequent_category_index > cur_offset) {
            // find category in category_sorted_stats array
            EXPECT_TRUE(counts_prev >= counts_cur);
          }
          counts_prev = counts_cur;

          set_categories_from_table_offsets.insert(category);
        }
      }
    }
    //   - check that the table offsets cover all frequent categories
    EXPECT_TRUE(set_categories_from_table_offsets == set_categories_frequent_categories_array);

    // check that infrequent categories as per category_location are not present in
    // frequent_categories array
    for (size_t category = 0; category < num_categories; ++category) {
      if (category_location[2 * category] < num_instances) {
        EXPECT_TRUE(set_categories_frequent_categories_array.find(category) ==
                    set_categories_frequent_categories_array.end());
      }
    }
  }

  // TODO:
  // // Check that the models of all the instances are identical
  // std::vector<dtype> category_frequent_index;
  // std::vector<dtype> category_location;
  // download_tensor(category_frequent_index, models[0].category_frequent_index, stream);
  // download_tensor(category_location, models[0].category_location, stream);
  // for (size_t instance = 0; instance < num_instances; ++instance) {
  //   for (size_t category = 0; category < num_categories; ++category) {

  //   }
  // }

  std::cout << "Finished the unit test for model init()!" << std::endl;
}

}  // namespace

TEST(hybrid_embedding_model_test, uint32) { model_test<uint32_t>(); }
TEST(hybrid_embedding_model_test, long_long) { model_test<long long>(); }
TEST(hybrid_embedding_model_test, init_model) {
  const size_t N = 5;
  const size_t batch_size = 15 * 64 * 1024;

  for (size_t num_instances = 1; num_instances <= 16; num_instances = 4 * num_instances) {
    for (size_t num_tables = 1; num_tables <= 32; num_tables = 4 * num_tables) {
      for (size_t i = 0; i < N; ++i) {
        model_init_test<uint32_t>(num_instances, num_tables, batch_size,
                                  CommunicationType::NVLink_SingleNode);
      }
    }
  }
}
