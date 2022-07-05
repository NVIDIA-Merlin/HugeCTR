#include <cub/cub.cuh>

#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_indices.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.cuh"
#include "HugeCTR/include/utils.cuh"

namespace indices_kernels {

template <typename dtype>
__global__ void fused_cache_masks(const dtype* __restrict__ samples,
                                  const dtype* __restrict__ category_location,
                                  bool* __restrict__ model_cache_mask,
                                  bool* __restrict__ network_cache_mask, uint32_t offset,
                                  uint32_t samples_size, uint32_t local_samples_size,
                                  uint32_t num_frequent, uint32_t num_frequent_per_model,
                                  uint32_t model_id, uint32_t num_instances) {
  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < samples_size) {
    dtype category = __ldg(samples + tid);
    dtype frequent_loc = __ldg(category_location + 2 * category);
    dtype frequent_index = __ldg(category_location + (2 * category + 1));

    if (frequent_loc == num_instances && frequent_index / num_frequent_per_model == model_id)
      model_cache_mask[(tid / local_samples_size) * num_frequent_per_model +
                       frequent_index % num_frequent_per_model] = true;
  }

  if (tid < local_samples_size) {
    dtype category = __ldg(samples + offset + tid);
    dtype frequent_loc = __ldg(category_location + 2 * category);
    dtype frequent_index = __ldg(category_location + (2 * category + 1));

    if (frequent_loc == num_instances) network_cache_mask[frequent_index] = true;
  }
}

__global__ void mask_indices_to_buffer_indices(
    uint32_t* __restrict__ model_cache_indices,
    const uint32_t* __restrict__ model_cache_indices_offsets, uint32_t num_instances,
    uint32_t num_frequent_per_model, uint32_t model_id) {
  const uint32_t num_selected = __ldg(model_cache_indices_offsets + num_instances);

  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_selected;
       i += blockDim.x * gridDim.x)
    model_cache_indices[i] =
        model_cache_indices[i] % num_frequent_per_model + num_frequent_per_model * model_id;
}

template <typename dtype>
__global__ void calculate_network_indices_mask(const dtype* __restrict__ local_samples,
                                               const dtype* __restrict__ category_location,
                                               bool* mask, uint32_t local_samples_size,
                                               uint32_t num_instances) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < local_samples_size;
       i += gridDim.x * blockDim.x) {
    dtype category = local_samples[i];
    uint32_t model_id = static_cast<uint32_t>(category_location[2 * category]);
    for (uint32_t section_id = 0; section_id < num_instances; section_id++) {
      mask[local_samples_size * section_id + i] = (model_id == section_id);
    }
  }
}

}  // namespace indices_kernels

namespace HugeCTR {
namespace hybrid_embedding {

// ===========================================================================================
// Frequent Compression
// ===========================================================================================

template <typename dtype>
FrequentEmbeddingCompression<dtype>::FrequentEmbeddingCompression(
    size_t max_num_frequent_categories, const Data<dtype>& data, const Model<dtype>& model)
    : data_(data), model_(model) {
  const int num_tables = data_.table_sizes.size();

  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf = GeneralBuffer2<CudaAllocator>::create();
  buf->reserve({max_num_frequent_categories, 1}, &model_cache_indices_);
  buf->reserve({model.num_instances + 1, 1}, &model_cache_indices_offsets_);
  buf->reserve({max_num_frequent_categories, 1}, &network_cache_indices_);
  buf->reserve({model.num_instances + 1, 1}, &network_cache_indices_offsets_);
  buf->reserve({2 * max_num_frequent_categories, 1}, &cache_masks_);
  buf->reserve({ceildiv<size_t>(data_.batch_size, model.num_instances) * num_tables, 1},
               &frequent_sample_indices_);
  buf->reserve({1}, &d_num_frequent_sample_indices_);

  // Temporary storage
  calculate_frequent_sample_indices_temp_storage_bytes((data_.batch_size / model.num_instances) *
                                                       num_tables);
  calculate_model_cache_indices_temp_storage_bytes(max_num_frequent_categories);
  calculate_network_cache_indices_temp_storage_bytes(max_num_frequent_categories);
  buf->reserve({frequent_sample_indices_temp_storage_bytes_, 1},
               &frequent_sample_indices_temp_storage_);
  buf->reserve({model_cache_indices_temp_storage_bytes_, 1}, &model_cache_indices_temp_storage_);
  buf->reserve({network_cache_indices_temp_storage_bytes_, 1},
               &network_cache_indices_temp_storage_);
  buf->allocate();

  FrequentEmbeddingCompressionView<dtype> view = {data_.samples.get_ptr(),
                                                  cache_masks_.get_ptr(),
                                                  model_cache_indices_.get_ptr(),
                                                  model_cache_indices_offsets_.get_ptr(),
                                                  network_cache_indices_.get_ptr(),
                                                  network_cache_indices_offsets_.get_ptr(),
                                                  d_num_frequent_sample_indices_.get_ptr(),
                                                  frequent_sample_indices_.get_ptr()};

  HCTR_LIB_THROW(cudaMalloc(&device_indices_view_, sizeof(view)));
  HCTR_LIB_THROW(cudaMemcpy(device_indices_view_, &view, sizeof(view), cudaMemcpyHostToDevice));
}

template <typename dtype>
struct FrequentSampleIndicesSelectOp {
  const dtype* samples;
  const dtype* category_location;
  uint32_t offset;
  dtype num_instances;
  __host__ __device__ __forceinline__ FrequentSampleIndicesSelectOp(const dtype* samples,
                                                                    const dtype* category_location,
                                                                    uint32_t offset,
                                                                    dtype num_instances)
      : samples(samples),
        category_location(category_location),
        offset(offset),
        num_instances(num_instances) {}
  __device__ __forceinline__ bool operator()(const uint32_t& idx) const {
    dtype category = __ldg(samples + offset + idx);
    dtype frequent_location = __ldg(category_location + 2 * category);
    return frequent_location == num_instances;
  }
};

template <typename dtype>
void FrequentEmbeddingCompression<dtype>::calculate_frequent_sample_indices_temp_storage_bytes(
    const size_t local_samples_size) {
  cub::CountingInputIterator<uint32_t> counting(0);
  FrequentSampleIndicesSelectOp<dtype> select_op(nullptr, nullptr, 0, 0);
  cub::DeviceSelect::If(nullptr, frequent_sample_indices_temp_storage_bytes_, counting,
                        (uint32_t*)nullptr, (uint32_t*)nullptr, local_samples_size, select_op, 0);
}

template <typename dtype>
void FrequentEmbeddingCompression<dtype>::calculate_model_cache_indices_temp_storage_bytes(
    const size_t num_frequent) {
  size_t select_bytes = 0;
  cub::CountingInputIterator<uint32_t> counting(0);
  cub::DeviceSelect::Flagged(nullptr, select_bytes, counting, (bool*)nullptr, (uint32_t*)nullptr,
                             (uint32_t*)nullptr, num_frequent, 0);

  constexpr uint32_t align = 256;
  model_cache_indices_temp_storage_bytes_ = alignTo<size_t>(num_frequent, align) + select_bytes;
}

template <typename dtype>
void FrequentEmbeddingCompression<dtype>::calculate_network_cache_indices_temp_storage_bytes(
    const size_t num_frequent) {
  size_t select_bytes = (size_t)0;
  cub::CountingInputIterator<uint32_t> counting(0);
  cub::DeviceSelect::Flagged(nullptr, select_bytes, counting, (bool*)nullptr, (uint32_t*)nullptr,
                             (uint32_t*)nullptr, num_frequent, 0);

  network_cache_indices_temp_storage_bytes_ = select_bytes;
}

template <typename dtype>
void FrequentEmbeddingCompression<dtype>::calculate_frequent_sample_indices(cudaStream_t stream) {
  const size_t num_networks = model_.num_instances;
  size_t local_samples_size = (data_.batch_size / num_networks) * data_.table_sizes.size();

  // Select indices of frequent categories appearing in the local MLP batch
  cub::CountingInputIterator<uint32_t> counting(0);
  FrequentSampleIndicesSelectOp<dtype> select_op(
      data_.samples.get_ptr(), model_.category_location.get_ptr(),
      model_.global_instance_id * local_samples_size, model_.num_instances);
  cub::DeviceSelect::If(
      reinterpret_cast<void*>(frequent_sample_indices_temp_storage_.get_ptr()),
      frequent_sample_indices_temp_storage_bytes_, counting, frequent_sample_indices_.get_ptr(),
      d_num_frequent_sample_indices_.get_ptr(), local_samples_size, select_op, stream);
}

template <typename dtype>
void FrequentEmbeddingCompression<dtype>::calculate_model_cache_indices(size_t sm_count,
                                                                        cudaStream_t stream) {
  const size_t num_instances = model_.num_instances;
  const size_t num_frequent = model_.num_frequent;
  const size_t samples_size = data_.batch_size * data_.table_sizes.size();
  size_t local_samples_size =
      ceildiv<size_t>(data_.batch_size, num_instances) * data_.table_sizes.size();

  // Note: we assume that the number of frequent categories is a
  // multiple of the number of models!
  const size_t num_frequent_per_model = num_frequent / num_instances;

  /**
   * Explanation of the mask:
   * The model owns num_frequent_per_model categories. For each network,
   * we want to know the categories that appear in their local batch and
   * belong to this model. The mask is the concatenation of num_network
   * sections of size num_frequent_per_model.
   * It has a size num_frequent but does not represent all the frequent
   * categories, only num_networks repetitions of the same categories.
   */

  // Temporary storage
  char* scratch_ptr = model_cache_indices_temp_storage_.get_ptr();
  void* d_temp_storage = reinterpret_cast<void*>(scratch_ptr);
  size_t temp_storage_bytes = model_cache_indices_temp_storage_bytes_;

  const bool* d_model_cache_mask = cache_masks_.get_ptr() + num_frequent;

  /* Select categories according to the mask */
  cub::CountingInputIterator<uint32_t> counting(0);
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, counting, d_model_cache_mask,
                             model_cache_indices_.get_ptr(),
                             model_cache_indices_offsets_.get_ptr() + num_instances, num_frequent,
                             stream);

  /* Compute offsets */
  constexpr size_t TPB_offsets = 256;
  size_t n_blocks = ceildiv<size_t>(num_instances, TPB_offsets);
  offsets_kernel<<<n_blocks, TPB_offsets, 0, stream>>>(model_cache_indices_.get_ptr(),
                                                       model_cache_indices_offsets_.get_ptr(),
                                                       num_instances, num_frequent_per_model);
  HCTR_LIB_THROW(cudaPeekAtLastError());

  /* Convert to buffer indices */

  constexpr size_t TPB_convert = 256;
  n_blocks = sm_count;
  indices_kernels::mask_indices_to_buffer_indices<<<n_blocks, TPB_convert, 0, stream>>>(
      model_cache_indices_.get_ptr(), model_cache_indices_offsets_.get_ptr(), num_instances,
      num_frequent_per_model, model_.global_instance_id);
  HCTR_LIB_THROW(cudaPeekAtLastError());
}

template <typename dtype>
void FrequentEmbeddingCompression<dtype>::calculate_cache_masks(cudaStream_t stream) {
  const size_t num_instances = model_.num_instances;
  const size_t num_frequent = model_.num_frequent;
  size_t samples_size = data_.batch_size * data_.table_sizes.size();
  size_t local_samples_size = ceildiv<size_t>(samples_size, num_instances);
  const size_t num_frequent_per_model = num_frequent / num_instances;

  bool* d_network_cache_mask = cache_masks_.get_ptr();
  bool* d_model_cache_mask = cache_masks_.get_ptr() + num_frequent;

  /* Initialize the masks to false */
  // // PROFILE_RECORD("fre_calculate_cache_masks.memset.start", stream);
  HCTR_LIB_THROW(cudaMemsetAsync(cache_masks_.get_ptr(), 0, 2 * num_frequent, stream));
  // // PROFILE_RECORD("fre_calculate_cache_masks.memset.stop", stream);

  /* Compute the model cache mask */
  constexpr size_t TPB_mask = 256;
  size_t n_blocks = ceildiv<size_t>(samples_size, TPB_mask);
  // // PROFILE_RECORD("fre_calculate_cache_masks.start", stream);
  indices_kernels::fused_cache_masks<<<n_blocks, TPB_mask, 0, stream>>>(
      data_.samples.get_ptr(), model_.category_location.get_ptr(), d_model_cache_mask,
      d_network_cache_mask, model_.global_instance_id * local_samples_size, samples_size,
      local_samples_size, num_frequent, num_frequent_per_model, model_.global_instance_id,
      model_.num_instances);
  HCTR_LIB_THROW(cudaPeekAtLastError());
  // // PROFILE_RECORD("fre_calculate_cache_masks.stop", stream);
}

template <typename dtype>
void FrequentEmbeddingCompression<dtype>::calculate_network_cache_indices(cudaStream_t stream) {
  const size_t num_instances = model_.num_instances;
  const size_t num_frequent = model_.num_frequent;
  size_t local_samples_size =
      ceildiv<size_t>(data_.batch_size, num_instances) * data_.table_sizes.size();

  // Note: we assume that the number of frequent categories is a
  // multiple of the number of models!
  const size_t num_frequent_per_model = num_frequent / num_instances;

  // Temporary storage
  char* scratch_ptr = network_cache_indices_temp_storage_.get_ptr();
  void* d_temp_storage = reinterpret_cast<void*>(scratch_ptr);
  size_t temp_storage_bytes = network_cache_indices_temp_storage_bytes_;

  const bool* d_network_cache_mask = cache_masks_.get_ptr();

  /* Select categories according to the mask */
  cub::CountingInputIterator<uint32_t> counting(0);
  // // PROFILE_RECORD("fre_calculate_network_cache_indices.device_select_flagged.start", stream);
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, counting, d_network_cache_mask,
                             network_cache_indices_.get_ptr(),
                             network_cache_indices_offsets_.get_ptr() + num_instances, num_frequent,
                             stream);
  // // PROFILE_RECORD("fre_calculate_network_cache_indices.device_select_flagged.stop", stream);

  /* Compute offsets */
  constexpr size_t TPB_offsets = 256;
  size_t n_blocks = ceildiv<size_t>(num_instances, TPB_offsets);
  // // PROFILE_RECORD("fre_calculate_network_cache_indices.offsets_kernel.start", stream);
  offsets_kernel<<<n_blocks, TPB_offsets, 0, stream>>>(network_cache_indices_.get_ptr(),
                                                       network_cache_indices_offsets_.get_ptr(),
                                                       num_instances, num_frequent_per_model);
  HCTR_LIB_THROW(cudaPeekAtLastError());
  // // PROFILE_RECORD("fre_calculate_network_cache_indices.offsets_kernel.stop", stream);
}

// ===========================================================================================
// Inrequent Selection
// ===========================================================================================

template <typename dtype>
InfrequentEmbeddingSelection<dtype>::InfrequentEmbeddingSelection(const Data<dtype>& data,
                                                                  const Model<dtype>& model)
    : data_(data), model_(model) {
  const size_t num_tables = data_.table_sizes.size();

  auto buf = GeneralBuffer2<CudaAllocator>::create();

  buf->reserve({data_.batch_size, num_tables}, &model_indices_);
  buf->reserve({ceildiv<size_t>(data_.batch_size, model.num_instances), num_tables},
               &network_indices_);
  buf->reserve({ceildiv<size_t>(data_.batch_size, model.num_instances), num_tables},
               &network_indices_src_model_id_);

  // buf->reserve({model.num_instances}, &model_indices_sizes_);
  // buf->reserve({model.num_instances}, &model_indices_sizes_ptrs_);
  // buf->reserve({model.num_instances}, &network_indices_sizes_);
  // buf->reserve({model.num_instances}, &network_indices_sizes_ptrs_);

  // Temporary storage
  calculate_model_indices_temp_storage_bytes(data_.batch_size, num_tables);
  calculate_network_indices_temp_storage_bytes(data_.batch_size, num_tables, model.num_instances);
  buf->reserve({model_indices_temp_storage_bytes_, 1}, &model_indices_temp_storage_);
  buf->reserve({network_indices_temp_storage_bytes_, 1}, &network_indices_temp_storage_);

  buf->allocate();

  auto managed_buf = GeneralBuffer2<CudaManagedAllocator>::create();
  managed_buf->reserve({model.num_instances + 1, 1}, &model_indices_offsets_);
  managed_buf->reserve({model.num_instances + 1, 1}, &network_indices_offsets_);
  managed_buf->allocate();
  //int current_device;
  //HCTR_LIB_THROW(cudaGetDevice(&current_device));
  //HCTR_LIB_THROW(cudaMemAdvise(managed_buf->get_ptr(), managed_buf->get_size_in_bytes(),
  //                             cudaMemAdviseSetReadMostly, current_device));

  InfrequentEmbeddingSelectionView<dtype> view = {data_.samples.get_ptr(),
                                                  model_indices_.get_ptr(),
                                                  model_indices_offsets_.get_ptr(),
                                                  network_indices_.get_ptr(),
                                                  network_indices_offsets_.get_ptr(),
                                                  network_indices_src_model_id_.get_ptr()};

  HCTR_LIB_THROW(cudaMalloc(&device_indices_view_, sizeof(view)));
  HCTR_LIB_THROW(cudaMemcpy(device_indices_view_, &view, sizeof(view), cudaMemcpyHostToDevice));
}

template <typename dtype>
struct ModelIndicesSelectOp {
  const dtype* samples;
  const dtype* category_location;
  uint32_t my_model_id;
  __host__ __device__ __forceinline__ ModelIndicesSelectOp(const dtype* samples,
                                                           const dtype* category_location,
                                                           uint32_t my_model_id)
      : samples(samples), category_location(category_location), my_model_id(my_model_id) {}
  __device__ __forceinline__ bool operator()(const uint32_t& idx) const {
    dtype category = __ldg(samples + idx);
    dtype model_id = __ldg(category_location + 2 * category);
    return model_id == my_model_id;
  }
};

template <typename dtype>
void InfrequentEmbeddingSelection<dtype>::calculate_model_indices_temp_storage_bytes(
    size_t max_batch_size, size_t table_size) {
  cub::CountingInputIterator<uint32_t> counting(0);
  ModelIndicesSelectOp<dtype> select_op(nullptr, nullptr, 0);
  cub::DeviceSelect::If(nullptr, model_indices_temp_storage_bytes_, counting, (uint32_t*)nullptr,
                        (uint32_t*)nullptr, max_batch_size * table_size, select_op, 0);
}

template <typename dtype>
void InfrequentEmbeddingSelection<dtype>::calculate_network_indices_temp_storage_bytes(
    size_t max_batch_size, size_t table_size, const uint32_t num_instances) {
  uint32_t samples_size = max_batch_size * table_size;
  uint32_t local_samples_size = ceildiv<uint32_t>(samples_size, num_instances);

  // Calculate select bytes
  size_t select_bytes = 0;
  cub::CountingInputIterator<uint32_t> counting(0);
  cub::DeviceSelect::Flagged(nullptr, select_bytes, counting, (bool*)nullptr, (uint32_t*)nullptr,
                             (uint32_t*)nullptr, samples_size, 0);

  // Total size
  constexpr uint32_t align = 256;
  network_indices_temp_storage_bytes_ =
      alignTo<size_t>(sizeof(bool) * samples_size, align) + select_bytes;
}

template <typename dtype>
void InfrequentEmbeddingSelection<dtype>::calculate_model_indices(cudaStream_t stream) {
  const uint32_t& num_instances = model_.num_instances;

  size_t local_batch_size = ceildiv<size_t>(data_.batch_size, num_instances);

  // Select indices of infrequent categories belonging to this model
  cub::CountingInputIterator<uint32_t> counting(0);
  ModelIndicesSelectOp<dtype> select_op(data_.samples.get_ptr(), model_.category_location.get_ptr(),
                                        model_.global_instance_id);
  // // PROFILE_RECORD("inf_calculate_model_indices.device_select_if.start", stream);
  cub::DeviceSelect::If(reinterpret_cast<void*>(model_indices_temp_storage_.get_ptr()),
                        model_indices_temp_storage_bytes_, counting, model_indices_.get_ptr(),
                        model_indices_offsets_.get_ptr() + num_instances,
                        data_.batch_size * data_.table_sizes.size(), select_op, stream);
  // // PROFILE_RECORD("inf_calculate_model_indices.device_select_if.stop", stream);

  // Compute offsets
  constexpr size_t TPB = 256;
  const size_t n_blocks = ceildiv<size_t>(num_instances, TPB);
  // // PROFILE_RECORD("inf_calculate_model_indices.offsets_kernel.start", stream);
  offsets_kernel<<<n_blocks, TPB, 0, stream>>>(model_indices_.get_ptr(),
                                               model_indices_offsets_.get_ptr(), num_instances,
                                               local_batch_size * data_.table_sizes.size());
  // // PROFILE_RECORD("inf_calculate_model_indices.offsets_kernel.stop", stream);
  HCTR_LIB_THROW(cudaPeekAtLastError());
}

template <typename dtype>
void InfrequentEmbeddingSelection<dtype>::calculate_network_indices(size_t sm_count,
                                                                    cudaStream_t stream) {
  const uint32_t num_instances = model_.num_instances;
  uint32_t samples_size = data_.batch_size * data_.table_sizes.size();
  uint32_t local_samples_size = ceildiv<uint32_t>(samples_size, num_instances);

  // Temporary storage
  constexpr uint32_t align = 256;
  char* scratch_ptr = network_indices_temp_storage_.get_ptr();
  size_t scratch_offset = 0;
  bool* d_mask = reinterpret_cast<bool*>(scratch_ptr + scratch_offset);
  scratch_offset += alignTo<size_t>(sizeof(bool) * samples_size, align);
  void* d_temp_storage = reinterpret_cast<void*>(scratch_ptr + scratch_offset);
  size_t temp_storage_bytes = network_indices_temp_storage_bytes_ - scratch_offset;

  // Compute mask (for each source GPU, whether each element in the batch is located there)
  constexpr uint32_t TPB_mask = 256;
  uint32_t n_blocks_mask = ceildiv<uint32_t>(local_samples_size, TPB_mask);
  // // PROFILE_RECORD("inf_calculate_network_indices.calculate_network_indices_mask.start",
  // stream);
  indices_kernels::calculate_network_indices_mask<<<n_blocks_mask, TPB_mask, 0, stream>>>(
      data_.samples.get_ptr() + model_.global_instance_id * local_samples_size,
      model_.category_location.get_ptr(), d_mask, local_samples_size, num_instances);
  HCTR_LIB_THROW(cudaPeekAtLastError());
  // // PROFILE_RECORD("inf_calculate_network_indices.calculate_network_indices_mask.stop", stream);

  // Select indices according to the mask
  cub::CountingInputIterator<uint32_t> counting(0);
  // // PROFILE_RECORD("inf_calculate_network_indices.device_select_flagged.start", stream);
  cub::DeviceSelect::Flagged(
      d_temp_storage, temp_storage_bytes, counting, d_mask, network_indices_.get_ptr(),
      network_indices_offsets_.get_ptr() + num_instances, samples_size, stream);
  // // PROFILE_RECORD("inf_calculate_network_indices.device_select_flagged.stop", stream);

  // Compute offsets
  constexpr uint32_t TPB_offsets = 256;
  uint32_t n_blocks_offsets = ceildiv<uint32_t>(num_instances, TPB_offsets);
  // // PROFILE_RECORD("inf_calculate_network_indices.offsets_kernel.start", stream);
  offsets_kernel<<<n_blocks_offsets, TPB_offsets, 0, stream>>>(network_indices_.get_ptr(),
                                                               network_indices_offsets_.get_ptr(),
                                                               num_instances, local_samples_size);
  HCTR_LIB_THROW(cudaPeekAtLastError());
  // // PROFILE_RECORD("inf_calculate_network_indices.offsets_kernel.stop", stream);

  // Re-map indices between 0 and local_samples_size - 1
  uint32_t TPB_remap = 256;
  uint32_t n_blocks_remap = sm_count;
  // // PROFILE_RECORD("inf_calculate_network_indices.modulo_kernel.start", stream);
  modulo_kernel<<<n_blocks_remap, TPB_remap, 0, stream>>>(
      network_indices_.get_ptr(), network_indices_offsets_.get_ptr() + num_instances,
      local_samples_size);
  HCTR_LIB_THROW(cudaPeekAtLastError());
  // // PROFILE_RECORD("inf_calculate_network_indices.modulo_kernel.stop", stream);

  // Figure out the model id for each indices
  model_id_kernel<<<n_blocks_remap, TPB_remap, 0, stream>>>(
      network_indices_offsets_.get_ptr(), network_indices_src_model_id_.get_ptr(),
      network_indices_offsets_.get_ptr() + num_instances);
  HCTR_LIB_THROW(cudaPeekAtLastError());
}

// template <typename dtype>
// void InfrequentEmbeddingSelection<dtype>::calculate_model_indices_sizes_from_offsets(
//     size_t embedding_vec_bytes, cudaStream_t stream) {
//   constexpr size_t TPB = 256;
//   const size_t n_blocks = ceildiv<size_t>(model_.num_instances, TPB);
//   offsets_to_sizes<<<n_blocks, TPB, 0, stream>>>(
//       model_indices_sizes_.get_ptr(), model_indices_offsets_.get_ptr(),
//       embedding_vec_bytes, model_.num_instances);
// }

// template <typename dtype>
// void InfrequentEmbeddingSelection<dtype>::calculate_network_indices_sizes_from_offsets(
//     size_t embedding_vec_bytes, cudaStream_t stream) {
//   constexpr size_t TPB = 256;
//   const size_t n_blocks = ceildiv<size_t>(model_.num_instances, TPB);
//   offsets_to_sizes<<<n_blocks, TPB, 0, stream>>>(
//       network_indices_sizes_.get_ptr(), network_indices_offsets_.get_ptr(),
//       embedding_vec_bytes, model_.num_instances);
// }

template <typename dtype>
void compute_indices(FrequentEmbeddingCompression<dtype>& compression,
                     InfrequentEmbeddingSelection<dtype>& selection,
                     CommunicationType communication_type, bool compute_network_cache_indices,
                     cudaStream_t stream, int sm_count) {
  compression.calculate_frequent_sample_indices(stream);
  selection.calculate_model_indices(stream);

  if (communication_type != CommunicationType::NVLink_SingleNode) {
    selection.calculate_network_indices(sm_count, stream);
  } else {
    compression.calculate_cache_masks(stream);
    if (compute_network_cache_indices) {
      compression.calculate_network_cache_indices(stream);
    }
    compression.calculate_model_cache_indices(sm_count, stream);
  }
}

template void compute_indices<uint32_t>(FrequentEmbeddingCompression<uint32_t>& compression,
                                        InfrequentEmbeddingSelection<uint32_t>& selection,
                                        CommunicationType communication_type,
                                        bool compute_network_cache_indices, cudaStream_t stream,
                                        int sm_count);

template void compute_indices<long long>(FrequentEmbeddingCompression<long long>& compression,
                                         InfrequentEmbeddingSelection<long long>& selection,
                                         CommunicationType communication_type,
                                         bool compute_network_cache_indices, cudaStream_t stream,
                                         int sm_count);

template class FrequentEmbeddingCompression<uint32_t>;
template class FrequentEmbeddingCompression<long long>;
template class InfrequentEmbeddingSelection<uint32_t>;
template class InfrequentEmbeddingSelection<long long>;

}  // namespace hybrid_embedding
}  // namespace HugeCTR
