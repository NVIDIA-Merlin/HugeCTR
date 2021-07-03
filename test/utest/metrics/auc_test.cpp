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

#include "HugeCTR/include/metrics.hpp"

#include "HugeCTR/include/general_buffer2.hpp"
#include "HugeCTR/include/resource_manager.hpp"
#include "gtest/gtest.h"
#include "utest/test_utils.h"

#include <vector>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <functional>

using namespace HugeCTR;

namespace {

const float eps = 2.0e-6;

template <typename T>
float sklearn_auc(size_t num_total_samples, const std::vector<float>& labels, const std::vector<T>& scores) {
  int num_procs = 1, rank = 0;
  #ifdef ENABLE_MPI
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
  #endif
  std::string temp_name = "tmpdata.bin";

  char *labels_ptr, *scores_ptr;
  std::vector<float> glob_labels;
  std::vector<T>     glob_scores;

  if (num_procs == 1)
  {
    labels_ptr = (char*)labels.data();
    scores_ptr = (char*)scores.data();
  }
  else
  {
    #ifdef ENABLE_MPI
      int my_size = labels.size();
      int offset = 0;
      std::vector<int> recv_offsets(num_procs+1);

      CK_MPI_THROW_(MPI_Exscan(&my_size, &offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
      CK_MPI_THROW_(MPI_Gather(&offset, 1, MPI_INT,
                               recv_offsets.data(), 1, MPI_INT,
                               0, MPI_COMM_WORLD));
      recv_offsets[num_procs] = num_total_samples;
      
      MPI_Datatype t_type;
      if (sizeof(T) == 4) { t_type = MPI_FLOAT; }
      if (sizeof(T) == 2) { t_type = MPI_SHORT; }
      
      std::vector<int> recv_sizes(num_procs);
      for (int i=0; i<num_procs; i++) {
        recv_sizes[i] = recv_offsets[i+1] - recv_offsets[i];
      }
    
      glob_labels.resize(num_total_samples);
      glob_scores.resize(num_total_samples);
      CK_MPI_THROW_(MPI_Gatherv(labels.data(), my_size, MPI_FLOAT,
                                glob_labels.data(), recv_sizes.data(), recv_offsets.data(), MPI_FLOAT,
                                0, MPI_COMM_WORLD));
      CK_MPI_THROW_(MPI_Gatherv(scores.data(), my_size, t_type,
                                glob_scores.data(), recv_sizes.data(), recv_offsets.data(), t_type,
                                0, MPI_COMM_WORLD));
    
      labels_ptr = (char*)glob_labels.data();
      scores_ptr = (char*)glob_scores.data();
    #endif
  }

  float result;
  if (rank == 0) {
    std::ofstream py_input(temp_name.c_str(), std::ios::binary | std::ios::out);
    py_input
      .write(labels_ptr, num_total_samples*sizeof(float))
      .write(scores_ptr, num_total_samples*sizeof(T));
    py_input.close();

    std::stringstream command;

    command << "python3 python_auc.py " << num_total_samples << " " << sizeof(T) << " " << temp_name;
    auto py_output = popen(command.str().c_str(), "r");
    int dummy = fscanf(py_output, "%f", &result);
    if (dummy != 1) {
      // Should never happen
      result = -1;
    }

    pclose(py_output);
  
    std::remove(temp_name.c_str());
  }

  #ifdef ENABLE_MPI
  CK_MPI_THROW_(MPI_Bcast(&result, 1, MPI_FLOAT, 0, MPI_COMM_WORLD));
  #endif

  return result;
}

template<typename T>
void gen_random(std::vector<float>& h_labels, std::vector<T>& h_scores, int offset) {
  std::mt19937 gen(424242 + offset);
  std::uniform_int_distribution<int> dis_label(0, 1);
  std::normal_distribution<float> dis_neg(0, 0.5);
  std::normal_distribution<float> dis_pos(1, 0.5);

  for (size_t i = 0; i < h_labels.size(); ++i) {
    int label = dis_label(gen);
    h_labels[i] = (float)label;

    h_scores[i] = (T)-1.0;
    while ( !( (T)0.0 <= h_scores[i] && h_scores[i] <= (T)1.0) ) {
      h_scores[i] = (float)( label ? dis_pos(gen) : dis_neg(gen) );
    }
  }
}

template<typename T>
void gen_same(std::vector<T>& h_labels, std::vector<T>& h_scores, int offset) {
  std::mt19937 gen(424242 + offset);
  std::uniform_int_distribution<int> dis_label(0, 1);

  for (size_t i = 0; i < h_labels.size(); ++i) {
    h_labels[i] = (float)dis_label(gen);
    h_scores[i] = 0.2345;
  }
}

template<typename T>
void gen_multilobe(std::vector<T>& h_labels, std::vector<T>& h_scores, int offset) {
  const int npeaks=2;
  std::mt19937 gen(424242 + offset);
  std::uniform_int_distribution<int> dis_label(0, 1);
  std::uniform_int_distribution<int> dis_score(1, npeaks);

  for (size_t i = 0; i < h_labels.size(); ++i) {
    h_labels[i] = (float)dis_label(gen);
    h_scores[i] = (float)dis_score(gen)/((float)npeaks+1);
  }
}

static int execution_number = 0;

template <typename T, typename Generator>
void auc_test(std::vector<int> device_list, size_t batch_size,
              size_t num_total_samples,
              Generator gen,
              size_t num_evals=1) {

  int num_procs = 1, rank = 0;
  #ifdef ENABLE_MPI
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
  #endif

  std::vector<std::vector<int>> vvgpu;
  int num_local_gpus = device_list.size();
  int num_total_gpus = num_procs*num_local_gpus;

  size_t batch_size_per_node = batch_size*num_local_gpus;
  size_t batch_size_per_iter = batch_size*num_total_gpus;
  size_t num_batches = (num_total_samples+batch_size_per_iter-1) / batch_size_per_iter;

  size_t last_batch_iter = num_total_samples - (num_batches-1)*batch_size_per_iter;
  size_t last_batch_gpu  = last_batch_iter > rank*batch_size_per_node    ?
                              last_batch_iter - rank*batch_size_per_node :
                              0;
                            
  size_t num_node_samples = (num_batches-1)*batch_size_per_node + std::min(last_batch_gpu, batch_size_per_node);

  // if there are multi-node, we assume each node has the same gpu device_list
  for (int i = 0; i < num_procs; i++) {
    vvgpu.push_back(device_list);
  }
  const auto resource_manager = ResourceManager::create(vvgpu, 424242);

  // Create AUC metric
  auto metric = std::make_unique<metrics::AUC<T>>(
    batch_size, num_batches, resource_manager);

  // Setup the containers
  std::vector<size_t> dims = {1, batch_size};

  std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>> bufs(num_local_gpus);
  std::vector<Tensor2<float>> labels_tensors(num_local_gpus);
  std::vector<Tensor2<T>>     scores_tensors(num_local_gpus);
  std::vector<metrics::RawMetricMap> metric_maps(num_local_gpus);

  for (int i=0; i<num_local_gpus; i++) {
    CudaDeviceContext context(resource_manager->get_local_gpu(i)->get_device_id());

    bufs[i] = GeneralBuffer2<CudaAllocator>::create();
    bufs[i]->reserve(dims, &labels_tensors[i]);
    bufs[i]->reserve(dims, &scores_tensors[i]);
    bufs[i]->allocate();

    metric_maps[i] = {
      {metrics::RawType::Pred,   scores_tensors[i].shrink()},
      {metrics::RawType::Label,  labels_tensors[i].shrink()}
    };
  }

  std::vector<float> h_labels(num_node_samples);
  std::vector<T>     h_scores(num_node_samples);
  gen(h_labels, h_scores, rank+num_procs*execution_number);
  execution_number++;

  float gpu_result;
  for (size_t eval = 0; eval < num_evals; eval++) {
    size_t num_processed = 0;
    for (size_t batch = 0; batch < num_batches; batch++) {
      // Populate device tensors
      metric->set_current_batch_size( std::min(batch_size_per_iter, num_total_samples-num_processed) );

      for (int i=0; i<num_local_gpus; i++) {
        CudaDeviceContext context(resource_manager->get_local_gpu(i)->get_device_id());
        size_t start = std::min(batch * num_local_gpus*batch_size + i*batch_size, num_node_samples);
        size_t count = std::min(batch * num_local_gpus*batch_size + (i+1)*batch_size, num_node_samples) - start;
        auto stream = resource_manager->get_local_gpu(i)->get_stream();

        cudaMemcpyAsync(labels_tensors[i].get_ptr(), h_labels.data() + start,
          count * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(scores_tensors[i].get_ptr(), h_scores.data() + start,
          count * sizeof(T),     cudaMemcpyHostToDevice, stream);

        metric->local_reduce(i, metric_maps[i]);
      }
      num_processed += batch_size_per_iter;
      metric->global_reduce(1);
    }
    gpu_result = metric->finalize_metric();
  }

  float ref_result = sklearn_auc(num_total_samples, h_labels, h_scores);
  //printf("GPU %f, ref %f \n", gpu_result, ref_result);

  ASSERT_NEAR(gpu_result, ref_result, eps);
}

class MPIEnvironment : public ::testing::Environment {
protected:
  virtual void SetUp()    { test::mpi_init(); }
  virtual void TearDown() { test::mpi_finalize(); }
  virtual ~MPIEnvironment() {};
};

}  // namespace

::testing::Environment* const mpi_env = ::testing::AddGlobalTestEnvironment(new MPIEnvironment);

TEST(auc_test, fp32_1gpu)            { auc_test<float>({0}, 10, 200, gen_random<float>); }
TEST(auc_test, fp32_1gpu_odd)        { auc_test<float>({0}, 10, 182, gen_random<float>); }
TEST(auc_test, fp32_2gpu)            { auc_test<float>({0, 1}, 10, 440, gen_random<float>); }
TEST(auc_test, fp32_2gpu_odd)        { auc_test<float>({0, 1}, 10, 443, gen_random<float>); }
TEST(auc_test, fp32_2_random_gpu)    { auc_test<float>({3, 5}, 12, 2341, gen_random<float>); }
TEST(auc_test, fp32_4gpu)            { auc_test<float>({0,1,2,3}, 5000, 22*5000+42, gen_random<float>); }
TEST(auc_test, fp32_4gpu_same)       { auc_test<float>({0,1,2,3}, 12, 154, gen_same<float>); }
TEST(auc_test, fp32_4gpu_same_large) { auc_test<float>({0,1,2,3}, 1312, 45155, gen_same<float>); }
TEST(auc_test, fp32_4gpu_multi)      { auc_test<float>({0,1,2,3}, 4143, 94622, gen_multilobe<float>); }
TEST(auc_test, fp32_8gpu)            { auc_test<float>({0,1,2,3,4,5,6,7}, 4231, 891373, gen_random<float>, 2); }
// TEST(auc_test, fp32_8gpu_large)      { auc_test<float>({0,1,2,3,4,5,6,7}, 131072, 89137319, gen_random<float>, 2); }

TEST(auc_test, fp16_1gpu)            { auc_test<__half>({0}, 15, 200, gen_random<__half>); }
TEST(auc_test, fp16_1gpu_odd)        { auc_test<__half>({0}, 11, 182, gen_random<__half>); }
TEST(auc_test, fp16_2gpu)            { auc_test<__half>({0, 1}, 10, 540, gen_random<__half>); }
TEST(auc_test, fp16_2gpu_odd)        { auc_test<__half>({0, 1}, 11, 443, gen_random<__half>); }
TEST(auc_test, fp16_2_random_gpu)    { auc_test<__half>({4, 6}, 13, 2351, gen_random<__half>); }
TEST(auc_test, fp16_4gpu)            { auc_test<__half>({0,1,2,3}, 5500, 22*5500+424, gen_random<__half>); }
TEST(auc_test, fp16_4gpu_multi)      { auc_test<__half>({0,1,2,3}, 7320, 81*7320+322, gen_random<__half>); }
TEST(auc_test, fp16_8gpu)            { auc_test<__half>({0,1,2,3,4,5,6,7}, 4321, 891573, gen_random<__half>, 2); }
// TEST(auc_test, fp16_8gpu_large)      { auc_test<__half>({0,1,2,3,4,5,6,7}, 131072, 89137319, gen_random<__half>, 2); }
