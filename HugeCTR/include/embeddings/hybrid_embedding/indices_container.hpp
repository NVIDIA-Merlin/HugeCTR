#pragma once

#include <cuda_runtime.h>

#include <map>
#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/data_readers/async_reader/async_reader_common.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/frequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_indices.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/infrequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/gpu_resource.hpp"
#include "HugeCTR/include/graph_wrapper.hpp"
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {
namespace hybrid_embedding {

template <typename dtype>
class BatchIndices {
 public:
  BatchIndices(std::vector<Model<dtype>>& models, std::vector<SparseTensor<dtype>> data_source,
               std::shared_ptr<ResourceManager>& resource_manager, size_t batch_size,
               std::vector<size_t>& slot_size_array, size_t max_num_frequent_categories,
               CommunicationType communication_type);

  void compute(int raw_device_id, size_t batch_size, cudaStream_t stream);

  FrequentEmbeddingCompression<dtype>& get_frequent(int raw_device_id) {
    return frequent_compression_[raw_device_id];
  }

  InfrequentEmbeddingSelection<dtype>& get_infrequent(int raw_device_id) {
    return infrequent_selection_[raw_device_id];
  }

 private:
  size_t num_slots_ = 0;
  std::shared_ptr<ResourceManager> resource_manager_;
  CommunicationType communication_type_;
  std::vector<Data<dtype>> data_;
  std::vector<FrequentEmbeddingCompression<dtype>> frequent_compression_;
  std::vector<InfrequentEmbeddingSelection<dtype>> infrequent_selection_;
};

}  // namespace hybrid_embedding
}  // namespace HugeCTR
