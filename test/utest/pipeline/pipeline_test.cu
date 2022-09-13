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

#include <cuda_profiler_api.h>
#include <omp.h>

#include "HugeCTR/include/gpu_resource.hpp"
#include "HugeCTR/include/pipeline.hpp"
#include "HugeCTR/include/resource_manager.hpp"
#include "utest/test_utils.h"
using namespace HugeCTR;

__global__ void setA(float *var, int count) {
  for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
    var[tid] = 1;
  }
}

__global__ void setB(float *var, int count) {
  for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
    var[tid] = 1;
  }
}

__global__ void setC(float *var, int count) {
  for (int tid = threadIdx.x; tid < count; tid += blockDim.x) {
    var[tid] = 1;
  }
}

void pipeline_test(const std::vector<int> &device_list, bool use_graph) {
  const auto &resource_manager = ResourceManager::create({device_list}, 0);
  cudaProfilerStart();
  std::vector<Pipeline> pipeline_list;
  std::vector<Pipeline> dup_pipeline_list;
  pipeline_list.resize(resource_manager->get_local_gpu_count());
  dup_pipeline_list.resize(resource_manager->get_local_gpu_count());

  for (size_t i = 0; i < resource_manager->get_local_gpu_count(); ++i) {
    auto gpu_resource = resource_manager->get_local_gpu(i);
    CudaDeviceContext context{gpu_resource->get_device_id()};
    int count = 1024 * 1024;
    float *data;
    HCTR_LIB_THROW(cudaMalloc(&data, count * sizeof(float)));
    float *data2;
    HCTR_LIB_THROW(cudaMalloc(&data2, count * sizeof(float)));
    float *data3;
    HCTR_LIB_THROW(cudaMalloc(&data3, count * sizeof(float)));

    std::shared_ptr<Scheduleable> b = std::make_shared<StreamContextScheduleable>(
        [=] { setB<<<1, 1024, 0, gpu_resource->get_stream()>>>(data2, count); });
    cudaEvent_t b_completion = b->record_done(use_graph);

    std::shared_ptr<Scheduleable> a = std::make_shared<StreamContextScheduleable>(
        [=] { setA<<<1, 1024, 0, gpu_resource->get_stream()>>>(data, count); });
    a->wait_event({b_completion}, use_graph);

    std::shared_ptr<Scheduleable> c = std::make_shared<StreamContextScheduleable>(
        [=] { setC<<<1, 1024, 0, gpu_resource->get_stream()>>>(data3, count); });

    pipeline_list[i] = Pipeline{"default", gpu_resource, {a, b, c}};
    dup_pipeline_list[i] = Pipeline{"overlap", gpu_resource, {a, b, c}};
  }
#pragma omp parallel num_threads(resource_manager->get_local_gpu_count())
  {
    size_t id = omp_get_thread_num();
    auto device_id = resource_manager->get_local_gpu(id)->get_device_id();
    CudaCPUDeviceContext context(device_id);
    if (use_graph) {
      pipeline_list[id].run_graph();
      dup_pipeline_list[id].run_graph();

      pipeline_list[id].run_graph();
      dup_pipeline_list[id].run_graph();
    }

    pipeline_list[id].run();
    dup_pipeline_list[id].run();

    pipeline_list[id].run();
    dup_pipeline_list[id].run();
  }
#pragma omp parallel num_threads(resource_manager->get_local_gpu_count())
  {
    size_t id = omp_get_thread_num();
    auto device_id = resource_manager->get_local_gpu(id)->get_device_id();
    CudaCPUDeviceContext context(device_id);
    HCTR_LIB_THROW(cudaStreamSynchronize(resource_manager->get_local_gpu(id)->get_stream()));
  }
  cudaProfilerStop();
}

TEST(pipeline_test, graph_test) { pipeline_test({0, 1}, true); }

TEST(pipeline_test, no_graph_test) { pipeline_test({0, 1}, false); }