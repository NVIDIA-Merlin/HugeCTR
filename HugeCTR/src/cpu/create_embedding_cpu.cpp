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

#include <cpu/create_embedding_cpu.hpp>

namespace HugeCTR {

template <typename TypeFP>
void create_embedding_cpu<TypeFP>::operator()(
    const InferenceParams& inference_params, const nlohmann::json& j_layers_array,
    std::vector<std::shared_ptr<Tensor2<int>>>& rows,
    std::vector<std::shared_ptr<Tensor2<float>>>& embeddingvecs,
    std::vector<size_t>& embedding_table_slot_size, std::vector<TensorEntry>* tensor_entries,
    std::vector<std::shared_ptr<LayerCPU>>* embeddings,
    std::shared_ptr<GeneralBuffer2<HostAllocator>>& blobs_buff) {
  HCTR_LOG(INFO, ROOT, "start create embedding for inference\n");
  auto j_data = j_layers_array[0];
  if (!has_key_(j_data, "sparse")) {
    HCTR_LOG(INFO, ROOT, "no sparse data input\n");
    return;
  }
  auto j_sparse_input = get_json(j_data, "sparse");
  std::unordered_map<std::string, std::pair<int, int>> slot_nums_map;
  for (unsigned int i = 0; i < j_sparse_input.size(); ++i) {
    auto top = get_value_from_json<std::string>(j_sparse_input[i], "top");
    auto slot_num = get_value_from_json<int>(j_sparse_input[i], "slot_num");
    auto max_feature_num_per_sample =
        get_max_feature_num_per_sample_from_nnz_per_slot(j_sparse_input[i]);
    HCTR_LOG_S(INFO, ROOT) << "sparse_input name " << top << std::endl;
    slot_nums_map[top] = std::make_pair(slot_num, max_feature_num_per_sample);
  }
  if (j_layers_array.size() < 1) {
    HCTR_OWN_THROW(Error_t::WrongInput, "layer not defined in config");
  }
  for (unsigned int i = 1; i < j_layers_array.size(); i++) {
    const nlohmann::json& j = j_layers_array[i];
    auto bottom_array = get_json(j, "bottom");
    if (bottom_array.is_array()) {
      continue;
    }
    std::string bottom = bottom_array.get<std::string>();
    ;
    auto slot_nums_map_iter = slot_nums_map.find(bottom);
    if (slot_nums_map_iter == slot_nums_map.end()) {
      continue;
    }
    const std::string layer_top = get_value_from_json<std::string>(j, "top");
    int slot_num = slot_nums_map_iter->second.first;
    int max_feature_num_per_sample = slot_nums_map_iter->second.second;
    auto j_hparam = get_json(j, "sparse_embedding_hparam");
    auto combiner = get_value_from_json<std::string>(j_hparam, "combiner");
    EmbeddingFeatureCombiner_t feature_combiner_type;
    if (combiner == "sum") {
      feature_combiner_type = EmbeddingFeatureCombiner_t::Sum;
    } else if (combiner == "mean") {
      feature_combiner_type = EmbeddingFeatureCombiner_t::Mean;
    } else {
      HCTR_OWN_THROW(Error_t::WrongInput, "combiner need to be 0 or 1");
    }
    size_t embedding_vec_size = get_value_from_json<size_t>(j_hparam, "embedding_vec_size");

    size_t prefix_slot_num = embedding_table_slot_size.back();
    embedding_table_slot_size.push_back(prefix_slot_num + slot_num);

    std::vector<size_t> row_dims = {
        static_cast<size_t>(inference_params.max_batchsize * slot_num + 1)};
    std::vector<size_t> embeddingvecs_dims = {
        static_cast<size_t>(inference_params.max_batchsize * max_feature_num_per_sample),
        static_cast<size_t>(embedding_vec_size)};
    std::shared_ptr<Tensor2<int>> row_tensor = std::make_shared<Tensor2<int>>();
    std::shared_ptr<Tensor2<float>> embeddingvecs_tensor = std::make_shared<Tensor2<float>>();
    blobs_buff->reserve(row_dims, row_tensor.get());
    blobs_buff->reserve(embeddingvecs_dims, embeddingvecs_tensor.get());
    rows.push_back(row_tensor);
    embeddingvecs.push_back(embeddingvecs_tensor);
    Tensor2<TypeFP> embedding_output;
    embeddings->push_back(std::make_shared<EmbeddingFeatureCombinerCPU<TypeFP>>(
        embeddingvecs[0], rows[0], embedding_output, inference_params.max_batchsize, slot_num,
        feature_combiner_type, blobs_buff));
    tensor_entries->push_back({layer_top, embedding_output.shrink()});
  }
  HCTR_LOG(INFO, ROOT, "create cpu embedding for inference success\n");
}

template struct create_embedding_cpu<float>;
template struct create_embedding_cpu<__half>;

}  // namespace HugeCTR
