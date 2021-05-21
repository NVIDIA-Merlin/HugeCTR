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

#pragma once
#include <omp.h>

#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"

namespace HugeCTR {

/* This implements inter-node portion of hierarchical all-to-all for node first device layout. */

#if defined(NCCL_A2A) && defined(ENABLE_MPI)

template <typename TypeEmbeddingComp>
class InterNodeHierarchicalAlltoAll {
 private:
  /* NCCL inter node communicators */
  std::vector<ncclComm_t> inter_comms_;

  enum TrainEval { TRAIN = 0, EVAL = 1, TE_COUNT = 2 };

  struct TrainEvalWrap {
    /* Inter send/recv offsets */
    std::vector<size_t> inter_send_offsets_;
    std::vector<size_t> inter_send_counts_;
    std::vector<size_t> inter_recv_offsets_;
    std::vector<size_t> inter_recv_counts_;

    /* Batch size */
    size_t batch_size_;
    size_t batch_size_per_gpu_;
  };

  /* internal variables */
  TrainEvalWrap te_vars[TE_COUNT];

  const ResourceManager* resource_manager_;

  size_t local_gpu_count_;
  size_t ngpus_;
  size_t slot_num_;
  size_t slot_num_per_node_;
  size_t embedding_vec_size_;
  int pid_;
  int num_procs_;
  ncclDataType_t nccl_dtype_;

  // Helper functions
  int create_nccl_comms();
  void calculate_offsets();
  size_t get_slot_num_per_gpu(size_t device) const;
  size_t get_slot_num_per_node(size_t my_node) const;
  ncclDataType_t get_nccl_type() const;

 public:
  // interface
  int init(const ResourceManager* resource_manager, size_t slot_num, size_t train_batch_size,
           size_t eval_batch_size, size_t embedding_vec_size);

  int fprop(bool is_train, Tensors2<TypeEmbeddingComp>& input, Tensors2<TypeEmbeddingComp>& output);

  int bprop(Tensors2<TypeEmbeddingComp>& input, Tensors2<TypeEmbeddingComp>& output);
};

/* Implementations */

// Create NCCL communicators
template <typename TypeEmbeddingComp>
int InterNodeHierarchicalAlltoAll<TypeEmbeddingComp>::create_nccl_comms() {
  inter_comms_.resize(local_gpu_count_, ncclComm_t());

  for (size_t g = 0; g < local_gpu_count_; g++) {
    ncclUniqueId inter_id;
    int device_id = resource_manager_->get_local_gpu(g)->get_device_id();
    if (pid_ == 0) {
      CK_NCCL_THROW_(ncclGetUniqueId(&inter_id));
    }

    CK_MPI_THROW_(MPI_Bcast(&inter_id, sizeof(inter_id), MPI_BYTE, 0, MPI_COMM_WORLD));
    CK_NCCL_THROW_(ncclGroupStart());
    CK_CUDA_THROW_(cudaSetDevice(device_id));
    CK_NCCL_THROW_(ncclCommInitRank(&inter_comms_[g], num_procs_, inter_id, pid_));
    CK_NCCL_THROW_(ncclGroupEnd());
  }
  return 0;
}

// calculate offsets
template <typename TypeEmbeddingComp>
void InterNodeHierarchicalAlltoAll<TypeEmbeddingComp>::calculate_offsets() {
  for (int te = 0; te < TE_COUNT; te++) {
    auto& var = te_vars[te];

    var.inter_send_offsets_.resize(num_procs_);
    var.inter_send_counts_.resize(num_procs_);
    var.inter_recv_offsets_.resize(num_procs_);
    var.inter_recv_counts_.resize(num_procs_);

    {
      size_t num_elems = var.batch_size_per_gpu_ * slot_num_per_node_ * embedding_vec_size_;
      for (int n = 0; n < num_procs_; n++) {
        var.inter_send_counts_[n] = num_elems;
        var.inter_send_offsets_[n] = n * num_elems;
      }
    }

    size_t offset = 0;
    for (int n = 0; n < num_procs_; n++) {
      size_t slot_num_per_node = get_slot_num_per_node(n);
      size_t num_elems = var.batch_size_per_gpu_ * slot_num_per_node * embedding_vec_size_;
      var.inter_recv_counts_[n] = num_elems;
      var.inter_recv_offsets_[n] = offset;
      offset += num_elems;
    }
  }
}

template <typename TypeEmbeddingComp>
size_t InterNodeHierarchicalAlltoAll<TypeEmbeddingComp>::get_slot_num_per_gpu(size_t device) const {
  size_t gid = resource_manager_->get_local_gpu(device)->get_global_id();
  size_t slot_num_per_gpu = slot_num_ / ngpus_;
  slot_num_per_gpu += (gid < (slot_num_ % ngpus_)) ? 1 : 0;
  return slot_num_per_gpu;
}

template <typename TypeEmbeddingComp>
size_t InterNodeHierarchicalAlltoAll<TypeEmbeddingComp>::get_slot_num_per_node(
    size_t my_node) const {
  // Note: Assumes nodeFirst device layout
  size_t slot_num_per_node = slot_num_ / num_procs_;
  slot_num_per_node += (my_node < (slot_num_ % num_procs_)) ? 1 : 0;
  return slot_num_per_node;
}

template <>
inline ncclDataType_t InterNodeHierarchicalAlltoAll<float>::get_nccl_type() const {
  return ncclFloat32;
}

template <>
inline ncclDataType_t InterNodeHierarchicalAlltoAll<__half>::get_nccl_type() const {
  return ncclFloat16;
}

// Initialization
template <typename TypeEmbeddingComp>
int InterNodeHierarchicalAlltoAll<TypeEmbeddingComp>::init(const ResourceManager* resource_manager,
                                                           size_t slot_num, size_t train_batch_size,
                                                           size_t eval_batch_size,
                                                           size_t embedding_vec_size) {
  // Init vars

  resource_manager_ = resource_manager;
  local_gpu_count_ = resource_manager->get_local_gpu_count();
  ngpus_ = resource_manager->get_global_gpu_count();
  pid_ = resource_manager->get_process_id();
  num_procs_ = resource_manager->get_num_process();

  slot_num_ = slot_num;
  slot_num_per_node_ = get_slot_num_per_node(pid_);
  embedding_vec_size_ = embedding_vec_size;
  nccl_dtype_ = get_nccl_type();

  te_vars[TRAIN].batch_size_ = train_batch_size;
  te_vars[EVAL].batch_size_ = eval_batch_size;

  te_vars[TRAIN].batch_size_per_gpu_ = train_batch_size / ngpus_;
  te_vars[EVAL].batch_size_per_gpu_ = eval_batch_size / ngpus_;

  create_nccl_comms();
  calculate_offsets();
  return 0;
}

// fprop
template <typename TypeEmbeddingComp>
int InterNodeHierarchicalAlltoAll<TypeEmbeddingComp>::fprop(bool is_train,
                                                            Tensors2<TypeEmbeddingComp>& input,
                                                            Tensors2<TypeEmbeddingComp>& output) {
  int te = is_train ? TRAIN : EVAL;
  auto& vars = te_vars[te];
  for (size_t g = 0; g < local_gpu_count_; g++) {
    int device_id = resource_manager_->get_local_gpu(g)->get_device_id();
    CK_CUDA_THROW_(cudaSetDevice(device_id));
    CK_NCCL_THROW_(ncclGroupStart());
    for (int p = 0; p < num_procs_; p++) {
      CK_NCCL_THROW_(ncclSend(input[g].get_ptr() + vars.inter_send_offsets_[p],
                              vars.inter_send_counts_[p], nccl_dtype_, p, inter_comms_[g],
                              resource_manager_->get_local_gpu(g)->get_stream()));

      CK_NCCL_THROW_(ncclRecv(output[g].get_ptr() + vars.inter_recv_offsets_[p],
                              vars.inter_recv_counts_[p], nccl_dtype_, p, inter_comms_[g],
                              resource_manager_->get_local_gpu(g)->get_stream()));
    }
    CK_NCCL_THROW_(ncclGroupEnd());
  }
  return 0;
}

template <typename TypeEmbeddingComp>
int InterNodeHierarchicalAlltoAll<TypeEmbeddingComp>::bprop(
    Tensors2<TypeEmbeddingComp>& bprop_input, Tensors2<TypeEmbeddingComp>& bprop_output) {
  auto& vars = te_vars[TRAIN];
  for (size_t g = 0; g < local_gpu_count_; g++) {
    int device_id = resource_manager_->get_local_gpu(g)->get_device_id();
    CK_CUDA_THROW_(cudaSetDevice(device_id));
    CK_NCCL_THROW_(ncclGroupStart());
    for (int p = 0; p < num_procs_; p++) {
      // send/recv offsets will be reverse for bprop
      CK_NCCL_THROW_(ncclSend(bprop_input[g].get_ptr() + vars.inter_recv_offsets_[p],
                              vars.inter_recv_counts_[p], nccl_dtype_, p, inter_comms_[g],
                              resource_manager_->get_local_gpu(g)->get_stream()));

      CK_NCCL_THROW_(ncclRecv(bprop_output[g].get_ptr() + vars.inter_send_offsets_[p],
                              vars.inter_send_counts_[p], nccl_dtype_, p, inter_comms_[g],
                              resource_manager_->get_local_gpu(g)->get_stream()));
    }
    CK_NCCL_THROW_(ncclGroupEnd());
  }
  return 0;
}
#endif
}  // namespace HugeCTR
