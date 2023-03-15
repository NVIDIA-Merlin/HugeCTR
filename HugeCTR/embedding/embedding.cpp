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

#include <embedding/data_parallel_embedding.hpp>
#include <embedding/embedding.hpp>
#include <embedding/hier_model_parallel_embedding.hpp>
#include <embedding/model_parallel_embedding.hpp>
namespace embedding {

std::vector<std::unique_ptr<IGroupedEmbeddingOp>> create_grouped_embeddings(
    std::shared_ptr<CoreResourceManager> core, const EmbeddingCollectionParam &ebc_param) {
  std::vector<std::unique_ptr<IGroupedEmbeddingOp>> embeddings;

  for (size_t emb_id = 0; emb_id < ebc_param.grouped_emb_params.size(); ++emb_id) {
    switch (ebc_param.grouped_emb_params[emb_id].table_placement_strategy) {
      case TablePlacementStrategy::DataParallel:
        embeddings.push_back(std::make_unique<UniformDPEmbedding>(core, ebc_param, emb_id));
        break;
      case TablePlacementStrategy::ModelParallel:
        switch (ebc_param.comm_strategy_) {
          case CommunicationStrategy::Uniform:
            embeddings.push_back(
                std::make_unique<UniformModelParallelEmbedding>(core, ebc_param, emb_id));
            break;
          case CommunicationStrategy::Hierarchical:
            embeddings.push_back(
                std::make_unique<HierModelParallelEmbedding>(core, ebc_param, emb_id));
            break;
          default:
            HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError, "grouped embedding create fail.");
        }
        break;
      default:
        HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError, "grouped embedding create fail.");
    }
  }
  return embeddings;
}
}  // namespace embedding
