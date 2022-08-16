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

#include <hps/lookup_session.hpp>
#include <utils.hpp>

namespace HugeCTR {

LookupSessionBase::~LookupSessionBase() = default;

std::shared_ptr<LookupSessionBase> LookupSessionBase::create(
    const InferenceParams& inference_params,
    const std::shared_ptr<EmbeddingCacheBase>& embedding_cache) {
  return std::make_shared<LookupSession>(inference_params, embedding_cache);
}

LookupSession::LookupSession(const InferenceParams& inference_params,
                             const std::shared_ptr<EmbeddingCacheBase>& embedding_cache)
    : LookupSessionBase(), embedding_cache_(embedding_cache), inference_params_(inference_params) {
  try {
    if (inference_params_.use_gpu_embedding_cache &&
        embedding_cache_->get_device_id() != inference_params_.device_id) {
      HCTR_OWN_THROW(
          Error_t::WrongInput,
          "The device id of inference_params is not consistent with that of embedding cache.");
    }
    HCTR_LOG(INFO, ROOT, "Creating lookup session for %s on device: %d\n",
             inference_params_.model_name.c_str(), inference_params_.device_id);

    CudaDeviceContext context(inference_params_.device_id);
    for (size_t idx = 0; idx < inference_params_.sparse_model_files.size(); ++idx) {
      cudaStream_t stream;
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      lookup_streams_.push_back(stream);
    }
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
  return;
}

LookupSession::~LookupSession() {
  CudaDeviceContext context(inference_params_.device_id);
  for (auto stream : lookup_streams_) cudaStreamDestroy(stream);
}

void LookupSession::lookup(const void* const h_keys, float* const d_vectors, const size_t num_keys,
                           const size_t table_id) {
  CudaDeviceContext context(inference_params_.device_id);
  // embedding_cache lookup
  embedding_cache_->lookup(table_id, d_vectors, h_keys, num_keys,
                           inference_params_.hit_rate_threshold, lookup_streams_[table_id]);
  HCTR_LIB_THROW(cudaStreamSynchronize(lookup_streams_[table_id]));
}

void LookupSession::lookup(const std::vector<const void*>& h_keys_per_table,
                           const std::vector<float*>& d_vectors_per_table,
                           const std::vector<size_t>& num_keys_per_table) {
  CudaDeviceContext context(inference_params_.device_id);
  HCTR_CHECK_HINT(h_keys_per_table.size() == inference_params_.sparse_model_files.size(),
                  "The h_keys_per_table.size() should be equal to the number of embedding tables");
  HCTR_CHECK_HINT(
      d_vectors_per_table.size() == inference_params_.sparse_model_files.size(),
      "The d_vectors_per_table.size() should be equal to the number of embedding tables");
  for (size_t table_id{0}; table_id < h_keys_per_table.size(); ++table_id) {
    embedding_cache_->lookup(table_id, d_vectors_per_table[table_id], h_keys_per_table[table_id],
                             num_keys_per_table[table_id], inference_params_.hit_rate_threshold,
                             lookup_streams_[table_id]);
  }
  for (size_t table_id{0}; table_id < h_keys_per_table.size(); ++table_id) {
    HCTR_LIB_THROW(cudaStreamSynchronize(lookup_streams_[table_id]));
  }
}

}  // namespace HugeCTR