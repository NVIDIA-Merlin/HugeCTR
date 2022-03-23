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
#include <pybind/model.hpp>

namespace HugeCTR {

class ModelPerfExt final : public Model {
 public:
  ModelPerfExt(const Solver& solver, const DataReaderParams& reader_params,
               std::shared_ptr<OptParamsPy>& opt_params,
               std::shared_ptr<EmbeddingTrainingCacheParams>& etc_params);
  ~ModelPerfExt() override {}
  bool train(bool is_first_batch) override;
  bool eval(bool is_first_batch) override;
  void fit(int num_epochs, int max_iter, int display, int eval_interval, int snapshot,
           std::string snapshot_prefix, DataSourceParams data_source_params) override;
  void add(DenseLayer& dense_layer) override;
  void add(Input& input) override { Model::add(input); };
  void add(SparseEmbedding& sparse_embedding) override { Model::add(sparse_embedding); };
  void add_internal(DenseLayer& dense_layer);

 private:
  void train_overlapped() override;
  void exchange_wgrad(size_t device_id) override;
  /**
   * add layer to network, python interface use only
   */
  void add_dense_layer(DenseLayer& dense_layer) override;
  void add_dense_layer_internal(DenseLayer& dense_layer, std::vector<TensorEntry>& tensor_entries,
                                const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
                                const std::shared_ptr<BufferBlock2<float>>& weight_buff,
                                const std::shared_ptr<BufferBlock2<__half>>& weight_buff_half,
                                const std::shared_ptr<BufferBlock2<float>>& wgrad_buff,
                                const std::shared_ptr<BufferBlock2<__half>>& wgrad_buff_half,
                                std::map<std::string, Tensor2<float>>& loss_tensor,
                                std::vector<std::unique_ptr<Layer>>& layers,
                                std::map<std::string, std::unique_ptr<ILoss>>& loss,
                                bool enable_cuda_graph, bool async_mlp_wgrad,
                                metrics::RawMetricMap* raw_metrics, int num_networks_in_global,
                                const std::shared_ptr<GPUResource>& gpu_resource,
                                bool use_mixed_precision, bool enable_tf32_compute, float scaler,
                                bool use_algorithm_search, std::vector<Layer*>* top_layers,
                                std::vector<Layer*>* bottom_layers, bool dlrm_bottom_mlp) override;
};

}  // namespace HugeCTR
