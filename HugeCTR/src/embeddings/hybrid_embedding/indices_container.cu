#include "HugeCTR/include/data_readers/async_reader/split_label_dense_sparse.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/indices_container.hpp"
#include "HugeCTR/include/tensor2.hpp"
#include "HugeCTR/include/utils.cuh"

namespace HugeCTR {
namespace hybrid_embedding {

template <typename dtype>
IndexProcessor<dtype>::IndexProcessor(
    std::vector<Model<dtype>>& models,
    std::vector<FrequentEmbeddingBase<dtype>*> frequent_embeddings,
    std::vector<InfrequentEmbeddingBase<dtype>*> infrequent_embeddings,
    std::shared_ptr<ResourceManager>& resource_manager, size_t queue_size, size_t batch_size,
    std::vector<size_t>& slot_size_array, size_t max_num_frequent_categories, bool mixed_precision,
    CommunicationType communication_type, size_t label_dim, size_t dense_dim, size_t sparse_dim,
    size_t sample_size_items)

    : containers_(queue_size),
      frequent_embeddings_(frequent_embeddings),
      infrequent_embeddings_(infrequent_embeddings),
      resource_manager_(resource_manager),
      batch_size_(batch_size),
      mixed_precision_(mixed_precision),
      communication_type_(communication_type) {
  auto local_gpu_count = resource_manager->get_local_gpu_count();
  batch_size_per_dev_ = batch_size / resource_manager_->get_global_gpu_count();
  sparse_dim_ = sparse_dim;
  sample_size_items_ = sample_size_items;

  for (size_t qid = 0; qid < queue_size; qid++) {
    auto& container = containers_[qid];

    for (size_t raw_device_id = 0; raw_device_id < local_gpu_count; raw_device_id++) {
      CudaDeviceContext ctx(resource_manager_->get_local_gpu(raw_device_id)->get_device_id());
      container.datas.emplace_back(slot_size_array, batch_size, 1);
    }

    for (size_t raw_device_id = 0; raw_device_id < local_gpu_count; raw_device_id++) {
      auto& local_gpu = resource_manager_->get_local_gpu(raw_device_id);
      auto local_device_id = local_gpu->get_device_id();
      CudaDeviceContext ctx(local_device_id);

      container.frequent_compressions.reserve(local_gpu_count);
      container.infrequent_selections.reserve(local_gpu_count);
      container.label_tensors.reserve(local_gpu_count);
      container.dense_tensors.reserve(local_gpu_count);

      container.frequent_compressions.emplace_back(
          max_num_frequent_categories, container.datas[raw_device_id], models[raw_device_id]);
      container.infrequent_selections.emplace_back(container.datas[raw_device_id],
                                                   models[raw_device_id]);

      auto label_dense_buffer = std::make_shared<RawPtrBuffer>(
          batch_size_per_dev_ * (label_dim * sizeof(LabelType) +
                                 dense_dim * (mixed_precision_ ? sizeof(__half) : sizeof(float))));
      auto dense_buffer = std::make_shared<RawPtrWrapper>(
          (LabelType*)(label_dense_buffer->get_ptr()) + batch_size_per_dev_ * label_dim);

      container.label_tensors.emplace_back(
          Tensor2<LabelType>({batch_size_per_dev_, label_dim}, label_dense_buffer).shrink());
      if (mixed_precision_) {
        container.dense_tensors.emplace_back(
            Tensor2<__half>({batch_size_per_dev_, dense_dim}, dense_buffer).shrink());
      } else {
        container.dense_tensors.emplace_back(
            Tensor2<float>({batch_size_per_dev_, dense_dim}, dense_buffer).shrink());
      }

      // Zero-initialize in case we're aligning dense tensor
      auto& dense_tensor_bag = container.dense_tensors.back();
      HCTR_LIB_THROW(
          cudaMemset(dense_tensor_bag.get_ptr(), 0, dense_tensor_bag.get_size_in_bytes()));
    }
  }
}

template <typename dtype>
void IndexProcessor<dtype>::calculate_indices(BatchDesc batch, size_t queue_id, int raw_device_id,
                                              cudaStream_t stream) {
  auto& local_gpu = resource_manager_->get_local_gpu(raw_device_id);
  auto& container = containers_[queue_id];
  auto& my_data = container.datas[raw_device_id];

  my_data.data_to_unique_categories(my_data.samples, stream);

  compute_indices(container.frequent_compressions[raw_device_id],
                  container.infrequent_selections[raw_device_id], communication_type_, true, stream,
                  local_gpu->get_sm_count());
}

template <typename dtype>
void IndexProcessor<dtype>::split3way(BatchDesc batch, size_t queue_id, int raw_device_id,
                                      cudaStream_t stream) {
  auto global_dev_id = resource_manager_->get_gpu_global_id_from_local_id(raw_device_id);
  auto& container = containers_[queue_id];
  auto& my_data = container.datas[raw_device_id];
  auto ptr_wrap =
      std::make_shared<RawPtrWrapper>(reinterpret_cast<InputType*>(batch.dev_data[raw_device_id]));

  // To save memory we're going to use the space in the Data for the unprocessed
  //  sparse features, and then run to_unique_categories essentially in place
  auto in_place_tensor = my_data.samples;
  in_place_tensor.reset_shape({batch_size_, sparse_dim_});
  if (mixed_precision_) {
    split_3_way<__half, SparseType>(
        Tensor2<LabelType>::stretch_from(container.label_tensors[raw_device_id]),
        Tensor2<__half>::stretch_from(container.dense_tensors[raw_device_id]), in_place_tensor,
        Tensor2<InputType>({batch_size_, sample_size_items_}, ptr_wrap),
        global_dev_id * batch_size_per_dev_, (global_dev_id + 1) * batch_size_per_dev_, stream);
  } else {
    split_3_way<float, SparseType>(
        Tensor2<LabelType>::stretch_from(container.label_tensors[raw_device_id]),
        Tensor2<float>::stretch_from(container.dense_tensors[raw_device_id]), in_place_tensor,
        Tensor2<InputType>({batch_size_, sample_size_items_}, ptr_wrap),
        global_dev_id * batch_size_per_dev_, (global_dev_id + 1) * batch_size_per_dev_, stream);
  }
}

template <typename dtype>
void IndexProcessor<dtype>::finalize(TensorBag2& label_tensor, TensorBag2& dense_tensor,
                                     SparseTensorBag& sparse_tensor, size_t queue_id,
                                     int raw_device_id, cudaStream_t stream) {
  auto& container = containers_[queue_id];

  if ((char*)label_tensor.get_ptr() + label_tensor.get_size_in_bytes() ==
      (char*)dense_tensor.get_ptr()) {
    HCTR_LIB_THROW(
        cudaMemcpyAsync(label_tensor.get_ptr(), container.label_tensors[raw_device_id].get_ptr(),
                        label_tensor.get_size_in_bytes() + dense_tensor.get_size_in_bytes(),
                        cudaMemcpyDeviceToDevice, stream));
  } else {
    HCTR_LIB_THROW(
        cudaMemcpyAsync(label_tensor.get_ptr(), container.label_tensors[raw_device_id].get_ptr(),
                        label_tensor.get_size_in_bytes(), cudaMemcpyDeviceToDevice, stream));

    HCTR_LIB_THROW(
        cudaMemcpyAsync(dense_tensor.get_ptr(), container.dense_tensors[raw_device_id].get_ptr(),
                        dense_tensor.get_size_in_bytes(), cudaMemcpyDeviceToDevice, stream));
  }

  // We don't copy the sparse tensor since all the required data are already in the
  // Data type and indices
  frequent_embeddings_[raw_device_id]->set_current_indices(
      &container.frequent_compressions[raw_device_id], stream);
  infrequent_embeddings_[raw_device_id]->set_current_indices(
      &container.infrequent_selections[raw_device_id], stream);
}

template <typename dtype>
size_t IndexProcessor<dtype>::get_queue_size() {
  return containers_.size();
}

template class IndexProcessor<uint32_t>;
template class IndexProcessor<long long>;

}  // namespace hybrid_embedding
}  // namespace HugeCTR
