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
struct IndexContainer {
  std::vector<TensorBag2> label_tensors;
  std::vector<TensorBag2> dense_tensors;

  std::vector<Data<dtype>> datas;
  std::vector<FrequentEmbeddingCompression<dtype>> frequent_compressions;
  std::vector<InfrequentEmbeddingSelection<dtype>> infrequent_selections;
};

template <typename dtype>
class IndexProcessor {
  using LabelType = float;
  using InputType = int;
  using SparseType = dtype;

 public:
  IndexProcessor(std::vector<Model<dtype>>& models,
                 std::vector<FrequentEmbeddingBase<dtype>*> frequent_embeddings,
                 std::vector<InfrequentEmbeddingBase<dtype>*> infrequent_embeddings,
                 std::shared_ptr<ResourceManager>& resource_manager, size_t queue_size,
                 size_t batch_size, std::vector<size_t>& slot_size_array,
                 size_t max_num_frequent_categories, bool mixed_precision,
                 CommunicationType communication_type, size_t label_dim, size_t dense_dim,
                 size_t sparse_dim, size_t sample_size_items);
  void calculate_indices(BatchDesc batch, size_t queue_id, int raw_device_id, cudaStream_t stream);
  void split3way(BatchDesc batch, size_t queue_id, int raw_device_id, cudaStream_t stream);
  void assign_sparse_indices(size_t queue_id, int raw_device_id, cudaStream_t stream);
  void assign_dense_and_label_tensors(TensorBag2& label_tensor, TensorBag2& dense_tensor,
                                      size_t queue_id, int raw_device_id, cudaStream_t stream);
  size_t get_queue_size();

 private:
  std::vector<IndexContainer<dtype>> containers_;

  std::vector<FrequentEmbeddingBase<dtype>*> frequent_embeddings_;
  std::vector<InfrequentEmbeddingBase<dtype>*> infrequent_embeddings_;

  std::shared_ptr<ResourceManager> resource_manager_;
  size_t batch_size_, batch_size_per_dev_;
  size_t sparse_dim_, sample_size_items_;
  bool mixed_precision_;
  CommunicationType communication_type_;
};

}  // namespace hybrid_embedding
}  // namespace HugeCTR
