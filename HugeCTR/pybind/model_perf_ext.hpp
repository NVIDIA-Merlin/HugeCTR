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
#include <HugeCTR/pybind/model.hpp>

namespace HugeCTR {

class ModelPerfExt final : public Model {
 public:
  ModelPerfExt(const Solver& solver, const DataReaderParams& reader_params,
               std::shared_ptr<OptParamsPy>& opt_params,
               std::shared_ptr<EmbeddingTrainingCacheParams>& etc_params);
  ~ModelPerfExt() override {}
  bool train() override;
  bool eval(int eval_batch = -1) override;
  void fit(int num_epochs, int max_iter, int display, int eval_interval, int snapshot,
           std::string snapshot_prefix) override;

 private:
  void train_overlapped() override;
  void exchange_wgrad(size_t device_id) override;
  

  struct GraphScheduler{
   private:
    size_t* executed_iter;
    size_t launched_iter; 
   public:
    GraphScheduler(std::shared_ptr<ResourceManager> resource_manager)
    : launched_iter(0) {
      // set up trickling launch
      CudaCPUDeviceContext ctx(resource_manager->get_local_gpu(0)->get_device_id());
      CK_CUDA_THROW_(cudaMallocHost((void**)&executed_iter, sizeof(size_t)));
      *executed_iter = 0;
    }
    ~GraphScheduler() {
      cudaFree(executed_iter);
    }
    void trickling(){
      // this function is called by the only thread, hence no need to specify the rank
      while (launched_iter > *(executed_iter) + 1){
        usleep(10);
      }
      launched_iter++;
    }
    void record_execution(size_t local_rank, cudaStream_t stream){
      // Only rank 0 needs to do the work
      if (local_rank == 0)
        inc_var(executed_iter, stream);
    }


  };

  std::unique_ptr<GraphScheduler> graph_scheduler_;
};

}  // namespace HugeCTR
