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

#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"

namespace HugeCTR {

#ifndef NCCL_A2A
#ifdef ENABLE_MPI

template <typename Type>
void SparseEmbeddingFunctors::all2all_exec(
    const std::unique_ptr<FasterGossipCommMulti::FasterGossipCommMulti<
        Type, FasterGossipCommMulti::FasterGossipCommMultiAll2AllTraits<Type>>> &all2all) {
  all2all->exec();
  return;
}

template void SparseEmbeddingFunctors::all2all_exec<float>(
    const std::unique_ptr<FasterGossipCommMulti::FasterGossipCommMulti<
        float, FasterGossipCommMulti::FasterGossipCommMultiAll2AllTraits<float>>> &all2all);

template void SparseEmbeddingFunctors::all2all_exec<__half>(
    const std::unique_ptr<FasterGossipCommMulti::FasterGossipCommMulti<
        __half, FasterGossipCommMulti::FasterGossipCommMultiAll2AllTraits<__half>>> &all2all);

#else

template <typename Type>
void SparseEmbeddingFunctors::all2all_exec(
    const std::unique_ptr<FasterGossipComm::FasterGossipComm<
        Type, FasterGossipComm::FasterGossipCommAll2AllTraits<Type>>> &all2all) {
  all2all->execAsync();
  all2all->sync();

  return;
}

template void SparseEmbeddingFunctors::all2all_exec<float>(
    const std::unique_ptr<FasterGossipComm::FasterGossipComm<
        float, FasterGossipComm::FasterGossipCommAll2AllTraits<float>>> &all2all);

template void SparseEmbeddingFunctors::all2all_exec<__half>(
    const std::unique_ptr<FasterGossipComm::FasterGossipComm<
        __half, FasterGossipComm::FasterGossipCommAll2AllTraits<__half>>> &all2all);

#endif
#endif

}  // namespace HugeCTR