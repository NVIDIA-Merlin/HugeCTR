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

#include <embedding/operators/generic_lookup.cuh>
#include <embedding/operators/model_forward.hpp>
#include <utils.cuh>
#include <utils.hpp>

namespace embedding {
using HugeCTR::CudaDeviceContext;

DPModelForward::DPModelForward(std::shared_ptr<CoreResourceManager> core) : core_(core) {}

namespace {

template <typename SrcType, typename DstType, typename offset_t>
struct DPForwardToFeatureMajorOutputMultiToOneDesc {
  using SrcT = SrcType;
  using DstT = DstType;

  HOST_DEVICE_INLINE int get_offset(int i) { return bucket_range_ptr[i]; }
  HOST_DEVICE_INLINE int get_vec_length(int i) {
    int lookup_id = local_lookup_ids_ptr[i / batch_size_per_gpu];
    return dst_id_to_ev_size_ptr[lookup_id];
  }
  HOST_DEVICE_INLINE int get_average_pooling_factor(int i) {
    int pooling_factor = static_cast<int>(bucket_range_ptr[i + 1] - bucket_range_ptr[i]);
    int lookup_id = local_lookup_ids_ptr[i / batch_size_per_gpu];
    return dst_id_to_combiner_ptr[lookup_id] == static_cast<char>(Combiner::Average)
               ? pooling_factor
               : 1;
  }
  HOST_DEVICE_INLINE const SrcType *get_src_ptr(int i) { return lookup_res_ptr[i]; }
  HOST_DEVICE_INLINE DstType *get_dst_ptr(int i) {
    int lookup_id = local_lookup_ids_ptr[i / batch_size_per_gpu];
    int bid = i % batch_size_per_gpu;
    int ev_size = dst_id_to_ev_size_ptr[lookup_id];
    return output_buffer_ptr + batch_size_per_gpu * dst_id_to_ev_start_indices_ptr[lookup_id] +
           bid * ev_size;
  }

  int num_vec_;
  int batch_size_per_gpu;
  const int *__restrict__ local_lookup_ids_ptr;
  const offset_t *__restrict__ bucket_range_ptr;
  const int *__restrict__ dst_id_to_ev_size_ptr;
  const int *__restrict__ dst_id_to_ev_start_indices_ptr;
  const char *__restrict__ dst_id_to_combiner_ptr;
  const float **__restrict__ lookup_res_ptr;
  DstType *output_buffer_ptr;
};

void dp_forward_to_feature_major_output(const core23::Tensor &lookup_res,
                                        const core23::Tensor &dp_feature_major_bucket_range,
                                        const core23::Tensor &local_lookup_ids,
                                        EmbeddingOutput &embedding_output, int batch_size_per_gpu,
                                        int gpu_id, int num_gpus, cudaStream_t stream) {
  int num_local_lookup = local_lookup_ids.num_elements();
  int batch_size = num_gpus * batch_size_per_gpu;
  auto &output_buffer = embedding_output.data;
  const auto &embedding_output_attr = embedding_output.attr;

  DISPATCH_INTEGRAL_FUNCTION_CORE23(
      dp_feature_major_bucket_range.data_type().type(), offset_t, [&] {
        DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(output_buffer.data_type().type(), emb_t, [&] {
          using CopyDesc = DPForwardToFeatureMajorOutputMultiToOneDesc<float, emb_t, offset_t>;
          CopyDesc multi_to_one_desc{batch_size_per_gpu * num_local_lookup,
                                     batch_size_per_gpu,
                                     local_lookup_ids.data<int>(),
                                     dp_feature_major_bucket_range.data<offset_t>(),
                                     embedding_output_attr.id_to_ev_size.data<int>(),
                                     embedding_output_attr.id_to_ev_start_indices.data<int>(),
                                     embedding_output_attr.id_to_combiner.data<char>(),
                                     (const float **)lookup_res.data(),
                                     output_buffer.data<emb_t>()};
          copy_multi_to_one(multi_to_one_desc, embedding_output_attr.max_ev_size, stream);
        });
      });
}

void dp_forward_to_batch_major_output(const core23::Tensor &lookup_res,
                                      const core23::Tensor &dp_feature_major_bucket_range,
                                      const core23::Tensor &local_lookup_ids,
                                      EmbeddingOutput &embedding_output, int batch_size_per_gpu,
                                      int gpu_id, int num_gpus, cudaStream_t stream) {
  int num_local_lookup = local_lookup_ids.num_elements();
  int batch_size = num_gpus * batch_size_per_gpu;
  auto &output_buffer = embedding_output.data;
  const auto &embedding_output_attr = embedding_output.attr;
  int num_lookup = embedding_output_attr.id_to_ev_size.num_elements();

  DISPATCH_INTEGRAL_FUNCTION_CORE23(
      dp_feature_major_bucket_range.data_type().type(), offset_t, [&] {
        DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(output_buffer.data_type().type(), emb_t, [&] {
          const int *local_lookup_ids_ptr = local_lookup_ids.data<int>();
          const offset_t *bucket_range_ptr = dp_feature_major_bucket_range.data<offset_t>();
          const int *dst_id_to_ev_size_ptr = embedding_output_attr.id_to_ev_size.data<int>();
          const int *dst_id_to_ev_start_indices_ptr =
              embedding_output_attr.id_to_ev_start_indices.data<int>();
          const char *dst_id_to_combiner_ptr = embedding_output_attr.id_to_combiner.data<char>();

          const float **lookup_res_ptr = (const float **)lookup_res.data();
          emb_t *output_buffer_ptr = output_buffer.data<emb_t>();

          auto multi_to_one_desc = make_MultiToOne<float, emb_t>(
              batch_size_per_gpu * num_local_lookup,
              [=] __device__(int i) { return bucket_range_ptr[i]; },
              [=] __device__(int i) {
                int pooling_factor =
                    static_cast<int>(bucket_range_ptr[i + 1] - bucket_range_ptr[i]);

                int lookup_id = local_lookup_ids_ptr[i / batch_size_per_gpu];
                return dst_id_to_combiner_ptr[lookup_id] == static_cast<char>(Combiner::Average)
                           ? pooling_factor
                           : 1;
              },
              [=] __device__(int i) {
                int lookup_id = local_lookup_ids_ptr[i / batch_size_per_gpu];
                return dst_id_to_ev_size_ptr[lookup_id];
              },
              [=] __device__(int i) { return lookup_res_ptr[i]; },
              [=] __device__(int i) {
                int lookup_id = local_lookup_ids_ptr[i / batch_size_per_gpu];
                int bid = i % batch_size_per_gpu;
                return output_buffer_ptr + bid * dst_id_to_ev_start_indices_ptr[num_lookup] +
                       dst_id_to_ev_start_indices_ptr[lookup_id];
              });
          copy_multi_to_one(multi_to_one_desc, embedding_output_attr.max_ev_size, stream);
        });
      });
}
}  // namespace

void DPModelForward::sparse_forward(const core23::Tensor &lookup_res,
                                    const core23::Tensor &dp_bucket_range,
                                    const core23::Tensor &local_lookup_ids,
                                    EmbeddingOutput &embedding_output, int batch_size_per_gpu) {
  CudaDeviceContext ctx(core_->get_device_id());
  auto stream = core_->get_local_gpu()->get_stream();
  int gpu_id = core_->get_global_gpu_id();
  int num_gpus = core_->get_global_gpu_count();

  if (embedding_output.attr.layout == EmbeddingLayout::FeatureMajor) {
    dp_forward_to_feature_major_output(lookup_res, dp_bucket_range, local_lookup_ids,
                                       embedding_output, batch_size_per_gpu, gpu_id, num_gpus,
                                       stream);
  } else {
    dp_forward_to_batch_major_output(lookup_res, dp_bucket_range, local_lookup_ids,
                                     embedding_output, batch_size_per_gpu, gpu_id, num_gpus,
                                     stream);
  }
}

void ModelForward::sparse_forward(const core23::Tensor &mp_ev, const core23::Tensor &bucket_range,
                                  ModelCommBuffer &model_comm_buffer, int batch_size) {
  CudaDeviceContext ctx(core_->get_device_id());
  int batch_size_per_gpu = batch_size / model_comm_buffer.attr.num_gpus;
  auto stream = core_->get_local_gpu()->get_stream();

  int num_lookup = model_comm_buffer.attr.num_lookup;
  if (num_lookup > 0) {
    DISPATCH_INTEGRAL_FUNCTION_CORE23(bucket_range.data_type().type(), offset_t, [&] {
      DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(model_comm_buffer.attr.type.type(), emb_t, [&] {
        const offset_t *bucket_range_ptr = bucket_range.data<offset_t>();
        const int *id_to_ev_size_ptr = model_comm_buffer.attr.id_to_ev_size.data<int>();
        const int *id_to_ev_start_indices_ptr =
            model_comm_buffer.attr.id_to_ev_start_indices.data<int>();
        const float **mp_ev_ptr = (const float **)mp_ev.data();
        emb_t **model_comm_buffer_ptr = (emb_t **)model_comm_buffer.data.data();

        auto multi_to_one_desc = make_MultiToOne<float, emb_t>(
            batch_size * num_lookup, [=] __device__(int i) { return bucket_range_ptr[i]; },
            [=] __device__(int i) { return 1; },
            [=] __device__(int i) {
              int i_lookup = i / batch_size;
              return id_to_ev_size_ptr[i_lookup];
            },
            [=] __device__(int i) { return mp_ev_ptr[i]; },
            [=] __device__(int i) {
              int i_lookup = i / batch_size;
              int batch_id = i % batch_size;
              int gpu_id = batch_id / batch_size_per_gpu;
              int ev_size = id_to_ev_size_ptr[i_lookup];
              int local_batch_id = batch_id % batch_size_per_gpu;
              return model_comm_buffer_ptr[gpu_id] +
                     batch_size_per_gpu * id_to_ev_start_indices_ptr[i_lookup] +
                     local_batch_id * ev_size;
            });
        copy_multi_to_one(multi_to_one_desc, core_->get_kernel_param(),
                          model_comm_buffer.attr.max_ev_size, stream);
      });
    });
  }
}

template <typename src_emb_t, typename dst_emb_t, typename offset_t>
struct DenseModelForwardOneToOneDesc {
  using SrcT = src_emb_t;
  using DstT = dst_emb_t;

  HOST_DEVICE_INLINE bool need_copy(int i) { return true; }

  HOST_DEVICE_INLINE int get_vec_length(int i) { return ev_size; }
  // we need a transform to src id use num_model_revers_idx
  HOST_DEVICE_INLINE const SrcT *get_src_ptr(int i) { return mp_ev_ptr[reverse_id_ptr[i]]; }
  HOST_DEVICE_INLINE DstT *get_dst_ptr(int i) { return dst_ptr + i * ev_size; }

  size_t num_vec_;
  int ev_size;
  const offset_t *__restrict__ reverse_id_ptr;
  const src_emb_t **__restrict__ mp_ev_ptr;
  dst_emb_t *__restrict__ dst_ptr;
};

void ModelForward::dense_forward(const core23::Tensor &mp_ev, const core23::Tensor &reverse_idx,
                                 DenseModelCommBuffer &model_comm_buffer, int batch_size,
                                 size_t num_key) {
  CudaDeviceContext ctx(core_->get_device_id());
  // int batch_size_per_gpu = batch_size / model_comm_buffer.attr.num_gpus;
  auto stream = core_->get_local_gpu()->get_stream();

  int num_lookup = model_comm_buffer.attr.num_local_lookup;
  if (num_lookup > 0) {
    DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(model_comm_buffer.attr.type.type(), emb_t, [&] {
      DISPATCH_INTEGRAL_FUNCTION_CORE23(reverse_idx.data_type().type(), offset_t, [&] {
        const float **mp_ev_ptr = (const float **)mp_ev.data();
        emb_t *model_comm_buffer_ptr = (emb_t *)model_comm_buffer.data.data();
        offset_t *reverse_idx_ptr = (offset_t *)reverse_idx.data();
        int ev_size = model_comm_buffer.attr.ev_size;
        using CopyDesc = DenseModelForwardOneToOneDesc<float, emb_t, offset_t>;

        CopyDesc one_to_one_desc = {num_key, ev_size, reverse_idx_ptr, mp_ev_ptr,
                                    model_comm_buffer_ptr};
        copy_one_to_one(one_to_one_desc, core_->get_kernel_param(), ev_size, stream);
      });
    });
  }
}

void ModelCommBufferAttr::init(std::shared_ptr<CoreResourceManager> core,
                               const EmbeddingCollectionParam &ebc_param, size_t grouped_id) {
  HugeCTR::CudaDeviceContext context(core->get_device_id());
  h_id_to_ev_size.clear();
  h_id_to_ev_start_indices = {0};

  int gpu_id = core->get_global_gpu_id();

  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    if (!ebc_param.has_table_shard(gpu_id, grouped_id, lookup_id)) continue;

    const auto &lookup_params = ebc_param.lookup_params;
    int ev_size = lookup_params[lookup_id].ev_size;
    h_id_to_ev_size.push_back(ev_size);
  }
  std::partial_sum(h_id_to_ev_size.begin(), h_id_to_ev_size.end(),
                   std::back_inserter(h_id_to_ev_start_indices));

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::BufferParams buffer_params;
  buffer_params.unitary = false;
  core23::TensorParams params = core23::TensorParams().device(device).buffer_params(buffer_params);

  this->id_to_ev_size = core23::Tensor(params.shape({static_cast<int64_t>(h_id_to_ev_size.size())})
                                           .data_type(core23::ScalarType::Int32));
  core23::copy_sync(this->id_to_ev_size, h_id_to_ev_size);
  this->id_to_ev_start_indices =
      core23::Tensor(params.shape({static_cast<int64_t>(h_id_to_ev_start_indices.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(this->id_to_ev_start_indices, h_id_to_ev_start_indices);

  this->num_lookup = static_cast<int>(h_id_to_ev_size.size());
  this->num_gpus = static_cast<int>(core->get_global_gpu_count());
  this->max_ev_elements = std::accumulate(h_id_to_ev_size.begin(), h_id_to_ev_size.end(), 0);

  this->layout = EmbeddingLayout::FeatureMajor;
  this->max_ev_size = h_id_to_ev_size.empty()
                          ? 0
                          : *std::max_element(h_id_to_ev_size.begin(), h_id_to_ev_size.end());
  this->type = ebc_param.emb_type;
}

void DenseModelCommBufferAttr::init(std::shared_ptr<CoreResourceManager> core,
                                    const EmbeddingCollectionParam &ebc_param, size_t grouped_id) {
  HugeCTR::CudaDeviceContext context(core->get_device_id());
  int gpu_id = core->get_global_gpu_id();
  this->ev_size = -1;
  this->num_local_lookup = 0;
  this->max_hotness_sum = 0;
  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    if (!ebc_param.has_table_shard(gpu_id, grouped_id, lookup_id)) continue;
    this->num_local_lookup++;
    const auto &lookup_params = ebc_param.lookup_params;
    this->max_hotness_sum += lookup_params[lookup_id].max_hotness;
    if (this->ev_size == -1) {
      this->ev_size = lookup_params[lookup_id].ev_size;
    } else {
      if (this->ev_size != lookup_params[lookup_id].ev_size) {
        HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                       "dense lookup don't support different ev_size between different tables");
      }
    }
  }

  if (this->num_local_lookup > 0 && this->ev_size == -1) {
    HCTR_OWN_THROW(HugeCTR::Error_t::IllegalCall,
                   "number of local lookup > 0 and don't get ev_size,please check input");
  }

  this->num_gpus = static_cast<int>(core->get_global_gpu_count());
  this->layout = EmbeddingLayout::FeatureMajor;
  this->type = ebc_param.emb_type;
}

void ModelCommBuffer::init(std::shared_ptr<CoreResourceManager> core,
                           const ModelCommBufferAttr &attr, int batch_size) {
  HugeCTR::CudaDeviceContext context(core->get_device_id());
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  this->data_list.clear();
  for (int gpu_id = 0; gpu_id < attr.num_gpus; ++gpu_id) {
    // We can not create size 0 Tensor
    this->data_list.emplace_back(
        params.shape({batch_size * attr.max_ev_elements / attr.num_gpus}).data_type(attr.type));
  }
  DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(attr.type.type(), emb_t, [&] {
    this->data = core23::init_tensor_list<emb_t>(this->data_list, core->get_device_id());
  });

  this->attr = attr;
}

void ModelCommBuffer::init_from_device_buffer(std::shared_ptr<CoreResourceManager> core,
                                              const std::vector<core23::Tensor> &data_buffer_list,
                                              const ModelCommBufferAttr &attr) {
  HugeCTR::CudaDeviceContext context(core->get_device_id());
  HCTR_CHECK(attr.max_ev_elements > 0);
  HCTR_CHECK(attr.num_lookup > 0);

  this->data_list.clear();
  this->data_list = data_buffer_list;
  this->data = core23::init_tensor_list<float>(this->data_list, core->get_device_id(),
                                               core->get_local_gpu()->get_stream());
  DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(attr.type.type(), emb_t, [&] {
    this->data = core23::init_tensor_list<emb_t>(this->data_list, core->get_device_id(),
                                                 core->get_local_gpu()->get_stream());
  });
  this->attr = attr;
}

void DenseModelCommBuffer::init(std::shared_ptr<CoreResourceManager> core,
                                const DenseModelCommBufferAttr &attr, int batch_size) {
  HugeCTR::CudaDeviceContext context(core->get_device_id());
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  double dense_unique_ratio = get_dense_unique_ratio();

  // FIX:when tensor dimension is , num local lookup is 0??
  // We can not create size 0 Tensor
  int64_t max_num_elements = static_cast<int64_t>(batch_size) * attr.max_hotness_sum * attr.ev_size;
  int64_t num_elements =
      static_cast<int64_t>(dense_unique_ratio * static_cast<double>(max_num_elements));
  this->data = core23::Tensor(params.shape({num_elements}).data_type(attr.type));

  this->attr = attr;
}

}  // namespace embedding
