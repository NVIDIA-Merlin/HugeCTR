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

#include <cpu/create_pipeline_cpu.hpp>
#include <cpu/create_embedding_cpu.hpp>

namespace HugeCTR {

template <typename TypeEmbeddingComp>
void create_pipeline_inference_cpu(const nlohmann::json& config,
                                  std::map<std::string, bool> tensor_active,
                                  const InferenceParser& inference_parser,
                                  Tensor2<float>& dense_input,
                                  std::vector<std::shared_ptr<Tensor2<int>>>& rows,
                                  std::vector<std::shared_ptr<Tensor2<float>>>& embeddingvecs,
                                  std::vector<size_t>& embedding_table_slot_size,
                                  std::vector<std::shared_ptr<LayerCPU>>* embeddings,
                                  NetworkCPU** network,
                                  const std::shared_ptr<CPUResource>& cpu_resource) {
  std::vector<TensorEntry> tensor_entries;

  auto j_layers_array = get_json(config, "layers");
  check_graph(tensor_active, j_layers_array);

  auto input_buffer = GeneralBuffer2<HostAllocator>::create();

  {
    const nlohmann::json& j_data = j_layers_array[0];
    auto j_dense = get_json(j_data, "dense");
    auto top_strs_dense = get_value_from_json<std::string>(j_dense, "top");
    auto dense_dim = get_value_from_json<size_t>(j_dense, "dense_dim");

    input_buffer->reserve({inference_parser.max_batchsize, dense_dim}, &dense_input);
    tensor_entries.push_back({top_strs_dense, dense_input.shrink()});
  }

  create_embedding_cpu<TypeEmbeddingComp>()(inference_parser, j_layers_array, rows, embeddingvecs, 
                                            embedding_table_slot_size, &tensor_entries,
                                            embeddings, input_buffer);
  input_buffer->allocate();

  *network = NetworkCPU::create_network(j_layers_array, tensor_entries, cpu_resource,
                                      inference_parser.use_mixed_precision);                  
}


void create_pipeline_cpu(const nlohmann::json& config,
                      std::map<std::string, bool> tensor_active,
                      const InferenceParser& inference_parser,
                      Tensor2<float>& dense_input,
                      std::vector<std::shared_ptr<Tensor2<int>>>& rows,
                      std::vector<std::shared_ptr<Tensor2<float>>>& embeddingvecs,
                      std::vector<size_t>& embedding_table_slot_size,
                      std::vector<std::shared_ptr<LayerCPU>>* embeddings, NetworkCPU** network,
                      const std::shared_ptr<CPUResource>& cpu_resource) {
  if (inference_parser.use_mixed_precision) {
    create_pipeline_inference_cpu<__half>(config, tensor_active, inference_parser, dense_input, rows, embeddingvecs,
                                        embedding_table_slot_size, embeddings, network, cpu_resource);
  } else {
    create_pipeline_inference_cpu<float>(config, tensor_active, inference_parser, dense_input, rows, embeddingvecs,
                                        embedding_table_slot_size, embeddings, network, cpu_resource);
  }
}

template void create_pipeline_inference_cpu<float>(const nlohmann::json& config,
                                  std::map<std::string, bool> tensor_active,
                                  const InferenceParser& inference_parser,
                                  Tensor2<float>& dense_input,
                                  std::vector<std::shared_ptr<Tensor2<int>>>& rows,
                                  std::vector<std::shared_ptr<Tensor2<float>>>& embeddingvecs,
                                  std::vector<size_t>& embedding_table_slot_size,
                                  std::vector<std::shared_ptr<LayerCPU>>* embeddings,
                                  NetworkCPU** network,
                                  const std::shared_ptr<CPUResource>& cpu_resource);
template void create_pipeline_inference_cpu<__half>(const nlohmann::json& config,
                                  std::map<std::string, bool> tensor_active,
                                  const InferenceParser& inference_parser,
                                  Tensor2<float>& dense_input,
                                  std::vector<std::shared_ptr<Tensor2<int>>>& rows,
                                  std::vector<std::shared_ptr<Tensor2<float>>>& embeddingvecs,
                                  std::vector<size_t>& embedding_table_slot_size,
                                  std::vector<std::shared_ptr<LayerCPU>>* embeddings,
                                  NetworkCPU** network,
                                  const std::shared_ptr<CPUResource>& cpu_resource);

} // namespace HugeCTR
