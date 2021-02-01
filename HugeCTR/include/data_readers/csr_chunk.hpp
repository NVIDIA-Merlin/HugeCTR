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

#pragma once

#include <data_readers/csr.hpp>
#include <vector>

namespace HugeCTR {

/**
 * @brief A wrapper of CSR objects.
 *
 * For each iteration DataReader will get one chunk of CSR objects from Heap
 * Such a chunk contains the training input data (sample + label) required by
 * this iteration
 */
template <typename CSR_Type>
class CSRChunk {
 private:
  std::vector<CSR<CSR_Type>>
      csr_buffers_;               /**< A vector of CSR objects, should be same number as devices. */
  Tensors2<float> label_buffers_; /**< A vector of label buffers */
  int label_dense_dim_;           /**< dimension of label + dense (for one sample) */
  int batchsize_;                 /**< batch size of training */
  int num_params_;
  int num_devices_;
  long long current_batchsize_;

 public:
  void set_current_batchsize(long long current_batchsize) {
    current_batchsize_ = current_batchsize;
  }
  long long get_current_batchsize() { return current_batchsize_; }
  /**
   * Ctor of CSRChunk.
   * Create and initialize the CSRChunk
   * @param num_csr_buffers the number of CSR object it will have.
   *        the number usually equal to num devices will be used.
   * @param batchsize batch size.
   * @param label_dense_dim dimension of label (for one sample).
   * @param slot_num slot num.
   * @param max_value_size the number of element of values the CSR matrix will have
   *        for num_rows rows (See csr.hpp).
   */
  CSRChunk(int num_devices, int batchsize, int label_dense_dim,
           const std::vector<DataReaderSparseParam>& params) {
    if (num_devices <= 0 || batchsize % num_devices != 0 || label_dense_dim <= 0) {
      CK_THROW_(Error_t::WrongInput,
                "num_devices <= 0 || batchsize % num_devices != 0 || label_dense_dim <= 0 ");
    }
    label_dense_dim_ = label_dense_dim;
    batchsize_ = batchsize;
    num_params_ = params.size();
    num_devices_ = num_devices;
    assert(csr_buffers_.empty() && label_buffers_.empty());

    std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> buff =
        GeneralBuffer2<CudaHostAllocator>::create();

    for (int i = 0; i < num_devices; i++) {
      for (auto& param : params) {
        int slots = 0;
        if (param.type == DataReaderSparse_t::Distributed) {
          slots = param.slot_num;
        } else if (param.type == DataReaderSparse_t::Localized) {
          int mod_slots = param.slot_num % num_devices;  // ceiling
          if (i < mod_slots) {
            slots = param.slot_num / num_devices + 1;
          } else {
            slots = param.slot_num / num_devices;
          }
        }
        csr_buffers_.emplace_back(batchsize * slots, param.max_feature_num * batchsize);
      }

      Tensor2<float> label_tensor;
      buff->reserve({static_cast<size_t>(batchsize / num_devices * label_dense_dim)},
                    &label_tensor);
      label_buffers_.push_back(label_tensor);
    }

    buff->allocate();
  }

  /**
   * Get the vector of csr objects.
   * This methord is used in collector (consumer) and data_reader (provider).
   */
  std::vector<CSR<CSR_Type>>& get_csr_buffers() { return csr_buffers_; }

  /**
   * Get the specific csr object.
   * This methord is used in collector (consumer) and data_reader (provider).
   */
  CSR<CSR_Type>& get_csr_buffer(int i) { return csr_buffers_[i]; }

  /**
   * Get the specific csr object with param_id and device_id.
   */
  CSR<CSR_Type>& get_csr_buffer(int param_id, int dev_id) {
    return csr_buffers_[dev_id * num_params_ + param_id];
  }

  /**
   * Call member function of all csr objects.
   * This methord is used in collector (consumer) and data_reader (provider).
   */
  template <typename MemberFunctionPointer>
  void apply_to_csr_buffers(const MemberFunctionPointer& fp) {
    for (auto& csr_buffer : csr_buffers_) {
      (csr_buffer.*fp)();
    }
  }

  /**
   * Get labels
   * This methord is used in collector (consumer) and data_reader (provider).
   */
  Tensors2<float>& get_label_buffers() { return label_buffers_; }
  int get_label_dense_dim() const { return label_dense_dim_; }
  int get_batchsize() const { return batchsize_; }
  int get_num_devices() const { return num_devices_; }
  int get_num_params() const { return num_params_; }

  /**
   * A copy Ctor but allocating new resources.
   * This Ctor is used in Heap (Ctor) to make several
   * copies of the object in heap.
   * @param C prototype of the Ctor.
   */
  CSRChunk(const CSRChunk&) = delete;
  CSRChunk& operator=(const CSRChunk&) = delete;
  CSRChunk(CSRChunk&&) = default;
};

}  // namespace HugeCTR
