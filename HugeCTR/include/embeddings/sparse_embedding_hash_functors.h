/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/utils.hpp"
#include "HugeCTR/include/gpu_resource.hpp"
#include "HugeCTR/include/hashtable/nv_hashtable.cuh"
#include "HugeCTR/include/embeddings/sparse_embedding_kernels.cuh"

namespace HugeCTR {

class SparseEmbeddingHashFunctors {

public:
  /**
   * Ctor of SparseEmbeddingHashFunctors. Copy construction and assigment are disabled.
   */
  SparseEmbeddingHashFunctors() {}
  SparseEmbeddingHashFunctors(SparseEmbeddingHashFunctors & obj) = delete;
  SparseEmbeddingHashFunctors& operator=(const SparseEmbeddingHashFunctors&) = delete;

  /**
   * Dtor of SparseEmbeddingHashFunctors.
   */
  ~SparseEmbeddingHashFunctors() {}

  // stream sync on multi GPUs 
  void sync_all_gpus(const std::shared_ptr<GPUResourceGroup> device_resources,
            const CudaDeviceContext& context) {

    int local_gpu_count = device_resources->size();

    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      CK_CUDA_THROW_(cudaStreamSynchronize((*device_resources)[id]->get_stream()));
    }
  }

  // forward computation 
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void forward(const int batch_size, 
              const int slot_num,
              const int embedding_vec_size, 
              const Tensors<TypeHashKey>& row_offsets_tensors,
              const Tensors<TypeHashKey>& value_tensors,
              const std::vector<std::unique_ptr<nv::HashTable<TypeHashKey, TypeHashValueIndex,
                std::numeric_limits<TypeHashKey>::max()>>>& hash_tables,
              const Tensors<float>& hash_table_value_tensors, 
              const Tensors<TypeHashValueIndex>& hash_value_index_tensors,
              Tensors<float>& embedding_feature_tensors,
              const CudaDeviceContext& context,
              const std::shared_ptr<GPUResourceGroup> device_resources) {

    int local_gpu_count = device_resources->size();

    // launch kernels on GPUs: do embedding lookup on multi GPUs
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());

      const auto &row_offset = row_offsets_tensors[id]->get_ptr();
      const auto &hash_key = value_tensors[id]->get_ptr();
      const auto &hash_table = hash_tables[id].get();
      const auto &hash_table_value = hash_table_value_tensors[id]->get_ptr();
      const auto &hash_value_index = hash_value_index_tensors[id]->get_ptr();
      auto embedding_feature = embedding_feature_tensors[id]->get_ptr();
      const cudaStream_t stream = (*device_resources)[id]->get_stream();

      try {
        // get hash_value_index from hash_table by hash_key
        size_t num;
        CK_CUDA_THROW_(cudaMemcpyAsync(&num, &row_offset[batch_size * slot_num], sizeof(TypeHashKey),
                                      cudaMemcpyDeviceToHost, stream));
        hash_table->get_insert(hash_key, hash_value_index, num, stream);
        // hash_table->get(hash_key, hash_value_index, num, stream);

        // do sum reduction
        dim3 blockSize(embedding_vec_size, 1,
                      1);  // each thread corresponds to one element in a embedding vector
        dim3 gridSize(batch_size, 1, 1);  // each block corresponds to a sample
        forward_sum_kernel<TypeHashKey, TypeHashValueIndex>
            <<<gridSize, blockSize, 0, stream>>>(batch_size, slot_num, embedding_vec_size, row_offset,
                                                hash_value_index, hash_table_value, embedding_feature);
        // for combiner=mean, call forward_scale() after this forward() and NCCL all-reduce
        // operation
      } catch (const std::runtime_error &rt_err) {
        std::cerr << rt_err.what() << std::endl;
        throw;
      }
    }

    return;
  }

  // this is an additional function for combiner=mean
  template <typename TypeHashKey>
  void forward_scale(const int batch_size, 
                    const int slot_num,
                    const int embedding_vec_size, 
                    const Tensors<TypeHashKey>& row_offset_allreduce_tensors,
                    Tensors<float>& output_tensors,
                    const std::shared_ptr<GPUResourceGroup> device_resources,
                    const CudaDeviceContext& context) {

    int local_gpu_count = device_resources->size();
    int total_gpu_count = device_resources->get_total_gpu_count();

    int batchsize_per_gpu = (int)(batch_size / total_gpu_count);
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());

      const auto &row_offset =
          row_offset_allreduce_tensors[id]->get_ptr() + id * batchsize_per_gpu;
      auto embedding_feature = output_tensors[id]->get_ptr();
      const auto &stream = (*device_resources)[id]->get_stream();

      try {
        dim3 blockSize(embedding_vec_size, 1, 1);
        dim3 gridSize(batch_size, 1, 1);

        forward_scale_kernel<<<gridSize, blockSize, 0, stream>>>(
            batch_size, slot_num, embedding_vec_size, row_offset, embedding_feature);

      } catch (const std::runtime_error &rt_err) {
        std::cerr << rt_err.what() << std::endl;
        throw;
      }
    }


    return;
  }

  // calculate wgrad
  template <typename TypeHashKey>
  void backward(const int batch_size, 
                const int slot_num,
                const int embedding_vec_size, 
                const int combiner, 
                const Tensors<TypeHashKey> row_offset_allreduce_tensors,
                const Tensors<float> embedding_feature_tensors,
                Tensors<float> wgrad_tensors,
                const std::shared_ptr<GPUResourceGroup> device_resources,
                const CudaDeviceContext& context) {

    int local_gpu_count = device_resources->size();

    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      const auto &stream = (*device_resources)[id]->get_stream();
      const auto &top_grad = embedding_feature_tensors[id]->get_ptr();
      const auto &row_offset = row_offset_allreduce_tensors[id]->get_ptr();
      auto wgrad = wgrad_tensors[id]->get_ptr();

      try {
        dim3 blockSize(embedding_vec_size, 1,
                      1);                // each thread corresponds to one element in an embedding vetor
        dim3 gridSize(batch_size, 1, 1);  // each block corresponds to a sample

        if (combiner == 0)  // sum
        {
          backward_sum_kernel<TypeHashKey><<<gridSize, blockSize, 0, stream>>>(
              batch_size, slot_num, embedding_vec_size, top_grad, wgrad);
        } else if (combiner == 1)  // mean
        {
          backward_mean_kernel<<<gridSize, blockSize, 0, stream>>>(
              batch_size, slot_num, embedding_vec_size, row_offset, top_grad, wgrad);
        } else {
          CK_THROW_(Error_t::WrongInput, "Invalid combiner type ");
        }
      } catch (const std::runtime_error &rt_err) {
        std::cerr << rt_err.what() << std::endl;
        throw;
      }
    }

    return;
  }

  // update embedding table(weights)
  template <typename TypeHashKey, typename TypeHashValueIndex>
  void update_params(const cudaStream_t stream, 
                    const int batch_size, 
                    const int slot_num,
                    const int embedding_vec_size, 
                    const long long max_vocabulary_size_per_gpu, 
                    OptParams opt_params,
                    const TypeHashKey *row_offset, 
                    const TypeHashKey *hash_key,
                    const nv::HashTable<TypeHashKey, TypeHashValueIndex, std::numeric_limits<TypeHashKey>::max()>
                        *hash_table,
                    TypeHashValueIndex *hash_value_index, 
                    TypeHashKey *sample_id, 
                    TypeHashKey *sample_id_sort,
                    TypeHashValueIndex *hash_value_index_sort, 
                    uint32_t *hash_value_index_count,
                    uint32_t *hash_value_index_count_offset, 
                    uint32_t *hash_value_index_count_counter,
                    void *temp_storage_sort, 
                    size_t temp_storage_sort_bytes, 
                    const float *wgrad,
                    TypeHashValueIndex *deltaw_hash_value_index, 
                    float *deltaw, 
                    float *hash_table_value) {
    try {
      // step1: expand sample IDs
      dim3 blockSize(64, 1, 1);
      dim3 gridSize((batch_size * slot_num + blockSize.x - 1) / blockSize.x, 1, 1);
      sample_id_expand_kernel<<<gridSize, blockSize, 0, stream>>>(batch_size, slot_num, row_offset,
                                                                  sample_id);

      int nnz;
      // this async memcpy will not perform as a async operation because the host memory is not a
      // pinned memory
      CK_CUDA_THROW_(cudaMemcpyAsync(&nnz, row_offset + batch_size * slot_num, sizeof(TypeHashKey),
                                    cudaMemcpyDeviceToHost, stream));

      // step2: get hash_value_index by hash_key
      hash_table->get_insert(hash_key, hash_value_index, nnz, stream);

      // step3: sort by hash_value_index
      int end_bit = (int)log2((float)max_vocabulary_size_per_gpu) + 1;
      CK_CUDA_THROW_(cub::DeviceRadixSort::SortPairs(
          (void *)temp_storage_sort, temp_storage_sort_bytes, hash_value_index, hash_value_index_sort,
          sample_id, sample_id_sort, nnz, 0, end_bit, stream, false));

      // step4: count the number for each unduplicated hash_value_index
      CK_CUDA_THROW_(cudaMemsetAsync(hash_value_index_count_counter, 0, sizeof(uint32_t), stream));
      gridSize.x = (nnz + (blockSize.x - 1)) / blockSize.x;
      value_count_kernel<<<gridSize, blockSize, 0, stream>>>(
          nnz, hash_value_index_sort, hash_value_index_count, hash_value_index_count_offset,
          hash_value_index_count_counter);

      uint32_t hash_hash_value_index_count_num = 0;
      // this async memcpy will not perform as a async operation because the host memory is not a
      // pinned memroy
      CK_CUDA_THROW_(cudaMemcpyAsync(&hash_hash_value_index_count_num, hash_value_index_count_counter,
                                    sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));

      // step5: use optimizer method to compute deltaw, and record corresponding
      // deltaw_hash_value_index
      blockSize.x = embedding_vec_size;
      gridSize.x = max(1, hash_hash_value_index_count_num);
      switch (opt_params.optimizer) {
        case 0:  // adam
          opt_params.hyperparams.adam.alpha_t =
              opt_params.lr *
              sqrt(1 - pow(opt_params.hyperparams.adam.beta2, opt_params.hyperparams.adam.times)) /
              (1 - pow(opt_params.hyperparams.adam.beta1, opt_params.hyperparams.adam.times));

          opt_adam_kernel<<<gridSize, blockSize, 0, stream>>>(
              hash_hash_value_index_count_num, embedding_vec_size, opt_params.hyperparams.adam,
              sample_id_sort, hash_value_index_sort, hash_value_index_count,
              hash_value_index_count_offset, wgrad, deltaw_hash_value_index, (float *)deltaw);
          break;
        case 1:  // momentum sgd
          opt_momentum_sgd_kernel<<<gridSize, blockSize, 0, stream>>>(
              hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
              opt_params.hyperparams.momentum, sample_id_sort, hash_value_index_sort,
              hash_value_index_count, hash_value_index_count_offset, wgrad, deltaw_hash_value_index,
              (float *)deltaw);
          break;
        case 2:  // nesterov
          opt_nesterov_kernel<<<gridSize, blockSize, 0, stream>>>(
              hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
              opt_params.hyperparams.nesterov, sample_id_sort, hash_value_index_sort,
              hash_value_index_count, hash_value_index_count_offset, wgrad, deltaw_hash_value_index,
              (float *)deltaw);
          break;
        default:
          CK_THROW_(Error_t::WrongInput, "Error: Invalid opitimizer type");
      }

      // step6: update hash_table_value by deltaw
      blockSize.x = embedding_vec_size;
      gridSize.x = max(1, hash_hash_value_index_count_num);
      update_kernel<TypeHashValueIndex>
          <<<gridSize, blockSize, 0, stream>>>(hash_hash_value_index_count_num, embedding_vec_size,
                                              deltaw_hash_value_index, deltaw, hash_table_value);
    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }

    return;
  }

  // collection communication: reduce_scatter
  void reduce_scatter(const int recv_count,
                      const Tensors<float>& send_tensors,
                      Tensors<float>& recv_tensors,
                      const std::shared_ptr<GPUResourceGroup> device_resources,
                      const CudaDeviceContext& context) {
    int local_gpu_count = device_resources->size();
    int total_gpu_count = device_resources->get_total_gpu_count();

    // for multi GPUs, use NCCL to do Reduce-Scatter(supporting multi-node GPU servers)
    if (total_gpu_count > 1) { 
      CK_NCCL_THROW_(ncclGroupStart());
      for (int id = 0; id < local_gpu_count; id++) {

        CK_NCCL_THROW_(ncclReduceScatter(send_tensors[id]->get_ptr(),   // send buf
                                        recv_tensors[id]->get_ptr(),  // recv buff
                                        recv_count, ncclFloat, ncclSum,
                                        *(*device_resources)[id]->get_nccl_ptr(),
                                        (*device_resources)[id]->get_stream()));
      }
      CK_NCCL_THROW_(ncclGroupEnd());
    } 
    // for single GPU, just copy the embedding_features to the output_tensor
    else {  // total_gpu_count == 1
      context.set_device((*device_resources)[0]->get_device_id());
      CK_CUDA_THROW_(cudaMemcpyAsync(recv_tensors[0]->get_ptr(),
                                    send_tensors[0]->get_ptr(),
                                    recv_count * sizeof(float), cudaMemcpyDeviceToDevice,
                                    (*device_resources)[0]->get_stream()));
    }

    return;
  }

  // collection communication: all_reduce
  template<typename TypeHashKey>
  void all_reduce(const int send_count,
                  const Tensors<TypeHashKey>& send_tensors,
                  Tensors<TypeHashKey>& recv_tensors,
                  const std::shared_ptr<GPUResourceGroup> device_resources,
                  const CudaDeviceContext& context) {

    int local_gpu_count = device_resources->size();
    int total_gpu_count = device_resources->get_total_gpu_count();

    // for multi GPUs, use NCCL to do all_reduce (supporting multi-node GPU servers)
    if (total_gpu_count > 1) {
      // need to know the type of TypeHashKey here
      ncclDataType_t type;
      switch (sizeof(TypeHashKey)) {
        case 4:
          type = ncclUint32;
          break;
        case 8:
          type = ncclUint64;
          break;
        default:
          CK_THROW_(Error_t::WrongInput, "Error: TypeHashKey not support by now");
      }

      CK_NCCL_THROW_(ncclGroupStart());
      for (int id = 0; id < local_gpu_count; id++) {
        CK_NCCL_THROW_(ncclAllReduce(send_tensors[id]->get_ptr(),
                                     recv_tensors[id]->get_ptr(), send_count, type,
                                     ncclSum, *(*device_resources)[id]->get_nccl_ptr(),
                                     (*device_resources)[id]->get_stream()));
      }

      CK_NCCL_THROW_(ncclGroupEnd());
    } else {  // gpu == 1
      context.set_device((*device_resources)[0]->get_device_id());
      CK_CUDA_THROW_(cudaMemcpyAsync(recv_tensors[0]->get_ptr(),
                                     send_tensors[0]->get_ptr(),
                                     send_count * sizeof(TypeHashKey), cudaMemcpyDeviceToDevice,
                                     (*device_resources)[0]->get_stream()));
    }

    return;
  }

  // collection communication: all_gather 
  void all_gather(int send_count,
                  const Tensors<float> send_tensors,
                  Tensors<float> recv_tensors,
                  const std::shared_ptr<GPUResourceGroup> device_resources,
                  const CudaDeviceContext& context) {
    int local_gpu_count = device_resources->size();
    int total_gpu_count = device_resources->get_total_gpu_count();

    // for multi GPUs, use NCCL to do All-Gather
    if (total_gpu_count > 1) {
      CK_NCCL_THROW_(ncclGroupStart());
      for (int id = 0; id < local_gpu_count; id++) {

        CK_NCCL_THROW_(ncclAllGather(send_tensors[id]->get_ptr(), // send buff
                                    recv_tensors[id]->get_ptr(),  // recv buff
                                    send_count, ncclFloat,
                                    *(*device_resources)[id]->get_nccl_ptr(),
                                    (*device_resources)[id]->get_stream()));
      }
      CK_NCCL_THROW_(ncclGroupEnd());
    }
    // for single GPU, just copy the grad from output_tensor to embedding_feature_tensor 
    else {  // total_gpu_count == 1
      context.set_device((*device_resources)[0]->get_device_id());
      CK_CUDA_THROW_(cudaMemcpyAsync(recv_tensors[0]->get_ptr(),
                                    send_tensors[0]->get_ptr(), 
                                    send_count * sizeof(float),
                                    cudaMemcpyDeviceToDevice,
                                    (*device_resources)[0]->get_stream()));
    }
  }

  // set liner data for a buffer
  template <typename Type>
  void memset_liner(const cudaStream_t stream, 
                        Type *data, 
                        const Type start_value, 
                        const Type stride_value,
                        const long long n) {
    try {
      int blockSize = 256;
      int gridSize = (n + blockSize - 1) / blockSize;

      memset_liner_kernel<Type><<<gridSize, blockSize, 0, stream>>>(data, start_value, stride_value, n);
    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
  }

  // get hash table value by value_index
  template <typename TypeHashValueIndex>
  void get_hash_table_value(const cudaStream_t stream, 
                                const long long count,
                                const int embedding_vec_size, 
                                const TypeHashValueIndex *value_index,
                                const float *hash_table_value, 
                                float *value_retrieved) {
    try {
      int blockSize = embedding_vec_size;
      int gridSize = count;

      get_hash_table_value_kernel<<<gridSize, blockSize, 0, stream>>>(count, embedding_vec_size, value_index,
                                                              hash_table_value, value_retrieved);

    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
  }

  // upload hash_table to GPUs
  template<typename TypeHashKey, typename TypeHashValueIndex>
  void upload_params_to_device(std::ifstream &weight_stream,
                                  long long vocabulary_size,
                                  int embedding_vec_size,
                                  int max_vocabulary_size_per_gpu,
                                  std::shared_ptr<GPUResourceGroup>& device_resources,
                                  Tensors<float>& hash_table_value_tensors,
                                  std::vector<std::unique_ptr<nv::HashTable<TypeHashKey, 
                                    TypeHashValueIndex, std::numeric_limits<TypeHashKey>::max()>>>& 
                                    hash_tables) {
    // check file size and vocabulary_size (file size <=　hash_table_size)
    weight_stream.seekg(0, weight_stream.end);
    long long file_size_in_B = weight_stream.tellg();
    weight_stream.seekg(0, weight_stream.beg);
    long long hash_table_size_in_B =
        vocabulary_size *
        ((long long)embedding_vec_size * sizeof(float) +
        sizeof(TypeHashKey));  // hash_key size + hash_value size
    if (file_size_in_B > hash_table_size_in_B) {
      CK_THROW_(Error_t::WrongInput,
                "Error: hash table file size is larger than hash table vocabulary_size");
    }

    int my_rank = 0;
    int n_ranks = 1;
  #ifdef ENABLE_MPI
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));
  #endif

    CudaDeviceContext context((*device_resources)[0]->get_device_id());

    // define size
    int gpu_count = device_resources->size();
    int chunk_loop = 1000;
    int hash_table_key_tile_size =
        1;  // must be 1, because we need to cal (key&gpu_count) to decide gpu_id for each <key,value>
    int hash_table_key_tile_size_in_B = hash_table_key_tile_size * sizeof(TypeHashKey);
    int hash_table_key_chunk_size = hash_table_key_tile_size * chunk_loop;
    int hash_table_key_chunk_size_in_B = hash_table_key_chunk_size * sizeof(TypeHashKey);
    int hash_table_value_tile_size = embedding_vec_size;
    int hash_table_value_tile_size_in_B = hash_table_value_tile_size * sizeof(float);
    int hash_table_value_chunk_size = hash_table_value_tile_size * chunk_loop;
    int hash_table_value_chunk_size_in_B = hash_table_value_chunk_size * sizeof(float);
    int hash_table_tile_size_in_B = hash_table_key_tile_size_in_B + hash_table_value_tile_size_in_B;
    int hash_table_chunk_size_in_B = hash_table_tile_size_in_B * chunk_loop;

    // CAUSION: can not decide how many values for each GPU, so need to allocate enough memory for
    // each GPU allocate GPU memory for hash_table_value_index
    std::unique_ptr<long long[]> hash_table_value_index_count_per_gpu(
        new long long[gpu_count]);  // <= hash_table_value_index_per_gpu_size
    memset(hash_table_value_index_count_per_gpu.get(), 0, sizeof(long long) * gpu_count);
    std::unique_ptr<long long[]> hash_table_value_index_count_chunk_per_gpu(new long long[gpu_count]);
    memset(hash_table_value_index_count_chunk_per_gpu.get(), 0, sizeof(long long) * gpu_count);
    std::unique_ptr<TypeHashKey *[]> hash_table_value_index_chunk_per_gpu_d(
        new TypeHashKey *[gpu_count]);
    for (int id = 0; id < gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      CK_CUDA_THROW_(
          cudaMalloc(&hash_table_value_index_chunk_per_gpu_d[id], hash_table_key_chunk_size_in_B));
      // initalize to zeros
      CK_CUDA_THROW_(cudaMemsetAsync(hash_table_value_index_chunk_per_gpu_d[id], 0,
                                    hash_table_key_chunk_size_in_B,
                                    (*device_resources)[id]->get_stream()));
    }

    // sync wait
    sync_all_gpus(device_resources, context);

    // CAUSION: can not decide how many values for each GPU, so need to allocate enough memory for
    // each GPU allocate CPU/GPU memory for hash_table/key/value chunk
    char *hash_table_chunk;
    CK_CUDA_THROW_(cudaMallocHost(&hash_table_chunk, hash_table_chunk_size_in_B));
    std::unique_ptr<TypeHashKey *[]> hash_table_key_chunk_per_gpu(new TypeHashKey *[gpu_count]);
    for (int id = 0; id < gpu_count; id++) {
      CK_CUDA_THROW_(
          cudaMallocHost(&hash_table_key_chunk_per_gpu[id], hash_table_key_chunk_size_in_B));
    }
    std::unique_ptr<TypeHashKey *[]> hash_table_key_chunk_per_gpu_d(new TypeHashKey *[gpu_count]);
    for (int id = 0; id < gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      CK_CUDA_THROW_(cudaMalloc(&hash_table_key_chunk_per_gpu_d[id], hash_table_key_chunk_size_in_B));
    }
    std::unique_ptr<float *[]> hash_table_value_chunk_per_gpu(new float *[gpu_count]);
    for (int id = 0; id < gpu_count; id++) {
      CK_CUDA_THROW_(
          cudaMallocHost(&hash_table_value_chunk_per_gpu[id], hash_table_value_chunk_size_in_B));
    }

    // do upload
    int loop_num = file_size_in_B / hash_table_chunk_size_in_B;
    for (int i = 0; i < loop_num; i++) {
      // read a chunk of data from file
      weight_stream.read(hash_table_chunk, hash_table_chunk_size_in_B);

      // memcpy from CPU to CPU
      char *src_buf = hash_table_chunk;
      TypeHashKey *key_dst_buf;
      float *value_dst_buf;
      for (int k = 0; k < chunk_loop; k++) {
        TypeHashKey key = *((TypeHashKey *)src_buf);
        int gid = key % device_resources->get_total_gpu_count();  // global GPU ID
        int id = device_resources->get_local_device_id(gid);      // local GPU ID
        int dst_rank = device_resources->get_pid(gid);

        if (my_rank == dst_rank) {
          // memcpy hash_table_key to corresponding GPU
          key_dst_buf =
              hash_table_key_chunk_per_gpu[id] + hash_table_value_index_count_chunk_per_gpu[id];
          CK_CUDA_THROW_(cudaMemcpyAsync(key_dst_buf, src_buf, hash_table_key_tile_size_in_B,
                                        cudaMemcpyHostToHost,
                                        (*device_resources)[id]->get_stream()));

          src_buf += hash_table_key_tile_size_in_B;

          // memcpy hash_table_value to corresponding GPU
          value_dst_buf =
              hash_table_value_chunk_per_gpu[id] +
              hash_table_value_index_count_chunk_per_gpu[id] * embedding_vec_size;
          CK_CUDA_THROW_(cudaMemcpyAsync(value_dst_buf, src_buf, hash_table_value_tile_size_in_B,
                                        cudaMemcpyHostToHost,
                                        (*device_resources)[id]->get_stream()));

          src_buf += hash_table_value_tile_size_in_B;

          hash_table_value_index_count_chunk_per_gpu[id] += hash_table_key_tile_size;
        } else {
          break;
        }
      }  // end of for(int k = 0; k < (chunk_loop * gpu_count); k++)

      // do HashTable insert <key,value_index>
      for (int id = 0; id < gpu_count; id++) {
        context.set_device((*device_resources)[id]->get_device_id());

        long long value_index_chunk_size = hash_table_value_index_count_chunk_per_gpu[id];

        // memcpy hash_table_key from CPU to GPU
        CK_CUDA_THROW_(
            cudaMemcpyAsync(hash_table_key_chunk_per_gpu_d[id], hash_table_key_chunk_per_gpu[id],
                            value_index_chunk_size * sizeof(TypeHashKey), cudaMemcpyHostToDevice,
                            (*device_resources)[id]->get_stream()));

        long long value_index_offset = hash_table_value_index_count_per_gpu[id];
        TypeHashKey *value_index_buf = hash_table_value_index_chunk_per_gpu_d[id];
        // set hash_table_value_index on GPU
        memset_liner((*device_resources)[id]->get_stream(),
                        value_index_buf, 
                        (TypeHashKey)value_index_offset,
                        (TypeHashKey)1, 
                        value_index_chunk_size);

        // do hash table insert <key, value_index> on GPU
        hash_tables[id]->insert(hash_table_key_chunk_per_gpu_d[id], value_index_buf,
                                value_index_chunk_size,
                                (*device_resources)[id]->get_stream());
        unsigned long long value_head = hash_tables[id]->add_value_head(value_index_chunk_size);
      }

      // memcpy hash_table_value from CPU to GPU
      for (int id = 0; id < gpu_count; id++) {
        context.set_device((*device_resources)[id]->get_device_id());
        long long value_chunk_size =
            hash_table_value_index_count_chunk_per_gpu[id] * embedding_vec_size;
        long long value_chunk_offset =
            hash_table_value_index_count_per_gpu[id] * embedding_vec_size;
        float *src_buf = hash_table_value_chunk_per_gpu[id];
        float *dst_buf = hash_table_value_tensors[id]->get_ptr() + value_chunk_offset;
        CK_CUDA_THROW_(cudaMemcpyAsync(dst_buf, src_buf, value_chunk_size * sizeof(float),
                                      cudaMemcpyHostToDevice,
                                      (*device_resources)[id]->get_stream()));
      }

      sync_all_gpus(device_resources, context);

      // set counter value
      for (int id = 0; id < gpu_count; id++) {
        hash_table_value_index_count_per_gpu[id] += hash_table_value_index_count_chunk_per_gpu[id];
        hash_table_value_index_count_chunk_per_gpu[id] = 0;  // reset chunk counter to zero

        if (hash_table_value_index_count_per_gpu[id] > max_vocabulary_size_per_gpu) {
          char msg[100];
          sprintf(msg, "The size of hash table on GPU%d is out of range %d\n", id,
                  max_vocabulary_size_per_gpu);
          CK_THROW_(Error_t::OutOfBound, msg);
        }
      }
    }  // end of for(int i = 0; i < loop_num; i++)

    // process the remaining data(less than a chunk)
    int remain_size_in_B = file_size_in_B - loop_num * hash_table_chunk_size_in_B;
    int remain_loop_num = remain_size_in_B / hash_table_tile_size_in_B;
    if (remain_loop_num) {
      // read all the remaining data
      weight_stream.read((char *)hash_table_chunk, remain_size_in_B);

      char *src_buf = hash_table_chunk;
      TypeHashKey *key_dst_buf;
      TypeHashKey *value_index_buf;
      float *value_dst_buf;
      for (int i = 0; i < remain_loop_num; i++) {
        TypeHashKey key = *((TypeHashKey *)src_buf);
        int gid = key % device_resources->get_total_gpu_count();  // global GPU ID
        int id = device_resources->get_local_device_id(gid);      // local GPU ID
        int dst_rank = device_resources->get_pid(gid);

        if (my_rank == dst_rank) {
          context.set_device((*device_resources)[id]->get_device_id());

          // memcpy hash_table_key from CPU to GPU
          key_dst_buf = hash_table_key_chunk_per_gpu_d[id];
          CK_CUDA_THROW_(cudaMemcpyAsync(key_dst_buf, src_buf, hash_table_key_tile_size_in_B,
                                        cudaMemcpyHostToDevice,
                                        (*device_resources)[id]->get_stream()));

          src_buf += hash_table_key_tile_size_in_B;

          // set value_index
          long long value_index_offset = hash_table_value_index_count_per_gpu[id];
          value_index_buf = hash_table_value_index_chunk_per_gpu_d[id];
          memset_liner((*device_resources)[id]->get_stream(), 
                          value_index_buf,
                          (TypeHashKey)value_index_offset, 
                          (TypeHashKey)1, 
                          1);

          // do hash table insert <key, value_index> on GPU
          hash_tables[id]->insert(hash_table_key_chunk_per_gpu_d[id], value_index_buf,
                                  hash_table_key_tile_size,
                                  (*device_resources)[id]->get_stream());
          unsigned long long value_head = hash_tables[id]->add_value_head(hash_table_key_tile_size);

          // memcpy hash_table_value from CPU to GPU
          long long value_offset =
              hash_table_value_index_count_per_gpu[id] * embedding_vec_size;
          value_dst_buf = hash_table_value_tensors[id]->get_ptr() + value_offset;
          CK_CUDA_THROW_(cudaMemcpyAsync(value_dst_buf, src_buf, hash_table_value_tile_size_in_B,
                                        cudaMemcpyHostToDevice,
                                        (*device_resources)[id]->get_stream()));
          src_buf += hash_table_value_tile_size_in_B;

          // set counter
          hash_table_value_index_count_per_gpu[id] += hash_table_key_tile_size;
        } else {
          break;
        }
      }

      // sync wait
      sync_all_gpus(device_resources, context);

    }  // end of if(remain_loop_num)

    // release resources
    for (int id = 0; id < gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      CK_CUDA_THROW_(cudaFree(hash_table_value_index_chunk_per_gpu_d[id]));
      CK_CUDA_THROW_(cudaFree(hash_table_key_chunk_per_gpu_d[id]));
    }
    CK_CUDA_THROW_(cudaFreeHost(hash_table_chunk));
    for (int id = 0; id < gpu_count; id++) {
      CK_CUDA_THROW_(cudaFreeHost(hash_table_key_chunk_per_gpu[id]));
      CK_CUDA_THROW_(cudaFreeHost(hash_table_value_chunk_per_gpu[id]));
    }

  }

  // download hash_table from GPUs to CPU
  template<typename TypeHashKey, typename TypeHashValueIndex>
  void download_params_to_host(std::ofstream &weight_stream,
                              long long vocabulary_size,
                              int embedding_vec_size,
                              int max_vocabulary_size_per_gpu,
                              std::shared_ptr<GPUResourceGroup>& device_resources,
                              Tensors<float>& hash_table_value_tensors,
                              std::vector<std::unique_ptr<nv::HashTable<TypeHashKey, 
                                TypeHashValueIndex, std::numeric_limits<TypeHashKey>::max()>>>& 
                                hash_tables) {

    CudaDeviceContext context((*device_resources)[0]->get_device_id());

    int gpu_count = device_resources->size();

    // memory allocation
    std::unique_ptr<unsigned long long[]> count(new unsigned long long[gpu_count]);
    unsigned long long max_count = 0;
    unsigned long long total_count = 0;
    for (int id = 0; id < gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      auto count_tmp = hash_tables[id]->get_size((*device_resources)[id]->get_stream());
      if (count_tmp != hash_tables[id]->get_value_head()) {
        CK_THROW_(Error_t::WrongInput,
                  "Error: hash_table get_value_head() size not equal to get_size()");
      }
      count[id] = count_tmp;
      max_count = max(max_count, count[id]);
      total_count += count[id];
    }

  #ifdef ENABLE_MPI
    CK_MPI_THROW_(MPI_Allreduce(MPI_IN_PLACE, &max_count, sizeof(unsigned long long), MPI_CHAR,
                                MPI_MAX, MPI_COMM_WORLD));
  #endif

    if (total_count > (unsigned long long)vocabulary_size) {
      CK_THROW_(Error_t::WrongInput,
                "Error: required download size is larger than hash table vocabulary_size");
    }

    std::unique_ptr<TypeHashKey *[]> h_hash_table_key(new TypeHashKey *[gpu_count]);
    std::unique_ptr<TypeHashKey *[]> d_hash_table_key(new TypeHashKey *[gpu_count]);
    std::unique_ptr<TypeHashValueIndex *[]> d_hash_table_value_index(
        new TypeHashValueIndex *[gpu_count]);
    std::unique_ptr<float *[]> h_hash_table_value(new float *[gpu_count]);
    std::unique_ptr<float *[]> d_hash_table_value(new float *[gpu_count]);
    std::unique_ptr<size_t *[]> d_dump_counter(new size_t *[gpu_count]);
    for (int id = 0; id < gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());

      cudaMallocHost(&h_hash_table_key[id], count[id] * sizeof(TypeHashKey));
      cudaMalloc(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey));
      cudaMalloc(&d_hash_table_value_index[id], count[id] * sizeof(TypeHashValueIndex));
      cudaMallocHost(&h_hash_table_value[id],
                    count[id] * embedding_vec_size * sizeof(float));
      cudaMalloc(&d_hash_table_value[id],
                count[id] * embedding_vec_size * sizeof(float));
      cudaMalloc(&d_dump_counter[id], count[id] * sizeof(size_t));
    }

    // dump hash table on GPU
    for (int id = 0; id < gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());

      hash_tables[id]->dump(d_hash_table_key[id], d_hash_table_value_index[id], 0,
                            max_vocabulary_size_per_gpu, d_dump_counter[id],
                            (*device_resources)[id]->get_stream());

      CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_key[id], d_hash_table_key[id],
                                    count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost,
                                    (*device_resources)[id]->get_stream()));

      get_hash_table_value((*device_resources)[id]->get_stream(), count[id],
                              embedding_vec_size, d_hash_table_value_index[id],
                              hash_table_value_tensors[id]->get_ptr(), d_hash_table_value[id]);

      CK_CUDA_THROW_(cudaMemcpyAsync(h_hash_table_value[id], d_hash_table_value[id],
                                    count[id] * embedding_vec_size * sizeof(float),
                                    cudaMemcpyDeviceToHost,
                                    (*device_resources)[id]->get_stream()));
    }

    // sync wait
    sync_all_gpus(device_resources, context);

    int my_rank = 0;
    int n_ranks = 1;
  #ifdef ENABLE_MPI
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));

  #endif

    const int master_node = 0;
    const int base_tag = 0xed;
    // TODO: could be optimized ???
    unsigned long long max_size_in_B =
        max_count * (sizeof(TypeHashKey) + sizeof(float) * embedding_vec_size);
    std::unique_ptr<char[]> file_buf(new char[max_size_in_B]);
    size_t key_size = sizeof(TypeHashKey);
    size_t value_size = sizeof(float) * embedding_vec_size;
    for (int id = 0; id < gpu_count; id++) {
      unsigned long long size_in_B =
          count[id] * (sizeof(TypeHashKey) + sizeof(float) * embedding_vec_size);
      unsigned long long offset = 0;
      for (unsigned int k = 0; k < count[id]; k++) {
        memcpy(file_buf.get() + offset, h_hash_table_key[id] + k, key_size);
        offset += key_size;
        memcpy(file_buf.get() + offset,
              h_hash_table_value[id] + k * embedding_vec_size, value_size);
        offset += value_size;
      }
      if (my_rank == master_node) {
        weight_stream.write(file_buf.get(), size_in_B);
      }
  #ifdef ENABLE_MPI
      else {
        int tag = (id << 8) | base_tag;
        CK_MPI_THROW_(
            MPI_Send(file_buf.get(), size_in_B, MPI_CHAR, master_node, tag, MPI_COMM_WORLD));
      }
  #endif
    }

  #ifdef ENABLE_MPI
    if (my_rank == master_node) {
      for (int r = 1; r < n_ranks; r++) {
        for (int id = 0; id < gpu_count; id++) {
          int tag = (id << 8) | base_tag;
          MPI_Status status;
          CK_MPI_THROW_(MPI_Probe(r, tag, MPI_COMM_WORLD, &status));
          int size_in_B;
          CK_MPI_THROW_(MPI_Get_count(&status, MPI_CHAR, &size_in_B));
          CK_MPI_THROW_(MPI_Recv(file_buf.get(), size_in_B, MPI_CHAR, r, tag, MPI_COMM_WORLD,
                                MPI_STATUS_IGNORE));
          weight_stream.write(file_buf.get(), size_in_B);
        }
      }
    }
  #endif

    for (int id = 0; id < gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());

      CK_CUDA_THROW_(cudaFreeHost(h_hash_table_key[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_key[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_value_index[id]));
      CK_CUDA_THROW_(cudaFreeHost(h_hash_table_value[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_value[id]));
      CK_CUDA_THROW_(cudaFree(d_dump_counter[id]));
    }

    return;
  }  // end of download_params_to_host()

  // get forward results from GPUs to CPU
  void get_forward_results(const int memcpy_size,
                          const Tensors<float> src_tensors,
                          float * dst_buf,
                          const std::shared_ptr<GPUResourceGroup>& device_resources,
                          CudaDeviceContext& context) {
    int local_gpu_count = device_resources->size();

    int offset = 0;
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      CK_CUDA_THROW_(cudaMemcpyAsync(dst_buf + offset, src_tensors[id]->get_ptr(),
                                    memcpy_size * sizeof(float), cudaMemcpyDeviceToHost,
                                    (*device_resources)[id]->get_stream()));
      offset += memcpy_size;
    }

    return;
  }

  // get backward results from GPU to CPU
  void get_backward_results(int devId,
                            int memcpy_size,
                            const Tensors<float> wgrad_tensors,
                            float * wgrad,
                            const std::shared_ptr<GPUResourceGroup>& device_resources,
                            CudaDeviceContext& context) {

    context.set_device((*device_resources)[devId]->get_device_id());
    CK_CUDA_THROW_(cudaMemcpyAsync(wgrad, wgrad_tensors[devId]->get_ptr(), 
                                  memcpy_size * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  (*device_resources)[devId]->get_stream()));
    CK_CUDA_THROW_(cudaStreamSynchronize((*device_resources)[devId]->get_stream()));

    return;
  }

  template<typename TypeHashKey, typename TypeHashValueIndex>
  void get_update_params_results(int max_vocabulary_size_per_gpu,
                                int embedding_vec_size,
                                long long vocabulary_size,
                                Tensors<float> hash_table_value_tensors,
                                const std::vector<std::unique_ptr<nv::HashTable<TypeHashKey, TypeHashValueIndex,
                                  std::numeric_limits<TypeHashKey>::max()>>>& hash_tables,
                                TypeHashKey *hash_table_key,
                                float *hash_table_value,
                                const std::shared_ptr<GPUResourceGroup>& device_resources,
                                CudaDeviceContext& context) {

    int local_gpu_count = device_resources->size();

    // memory allocation
    std::unique_ptr<unsigned long long[]> count(new unsigned long long[local_gpu_count]);
    unsigned long long max_count = 0;
    unsigned long long total_count = 0;
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      if ((count[id] = hash_tables[id]->get_value_head()) !=
          hash_tables[id]->get_size((*device_resources)[id]->get_stream())) {
        CK_THROW_(Error_t::WrongInput,
                  "Error: hash_table get_value_head() size not equal to get_size()");
      }
      max_count = max(max_count, count[id]);
      total_count += count[id];

  #ifndef NDEBUG
      std::cout << "GPU[%d]: " << id << "number of <key,value> pairs:%d" << count[id] << std::endl;
  #endif
    }

    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());
      count[id] = hash_tables[id]->get_size((*device_resources)[id]->get_stream());
    }

    if (total_count > (unsigned long long)vocabulary_size) {
      CK_THROW_(Error_t::WrongInput,
                "Error: required download size is larger than hash table vocabulary_size");
    }

    std::unique_ptr<TypeHashKey *[]> d_hash_table_key(new TypeHashKey *[local_gpu_count]);
    std::unique_ptr<TypeHashValueIndex *[]> d_hash_table_value_index(
        new TypeHashValueIndex *[local_gpu_count]);
    std::unique_ptr<float *[]> d_hash_table_value(new float *[local_gpu_count]);
    std::unique_ptr<size_t *[]> d_dump_counter(new size_t *[local_gpu_count]);
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());

      cudaMalloc(&d_hash_table_key[id], count[id] * sizeof(TypeHashKey));
      cudaMalloc(&d_hash_table_value_index[id], count[id] * sizeof(TypeHashValueIndex));
      cudaMalloc(&d_hash_table_value[id],
                count[id] * embedding_vec_size * sizeof(float));
      cudaMalloc(&d_dump_counter[id], count[id] * sizeof(size_t));
    }

    // dump hash table on GPU
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());

      hash_tables[id]->dump(d_hash_table_key[id], d_hash_table_value_index[id], 0,
                            max_vocabulary_size_per_gpu, d_dump_counter[id],
                            (*device_resources)[id]->get_stream());

      get_hash_table_value((*device_resources)[id]->get_stream(), count[id],
                              embedding_vec_size, d_hash_table_value_index[id],
                              hash_table_value_tensors[id]->get_ptr(), d_hash_table_value[id]);
    }

    // sync wait
    sync_all_gpus(device_resources, context);

    // memcpy from GPU to CPU memory
    long long key_offset = 0;
    long long value_offset = 0;
    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());

      CK_CUDA_THROW_(cudaMemcpy(hash_table_key + key_offset, d_hash_table_key[id],
                                count[id] * sizeof(TypeHashKey), cudaMemcpyDeviceToHost));
      key_offset += count[id];

      CK_CUDA_THROW_(cudaMemcpy(hash_table_value + value_offset, d_hash_table_value[id],
                                count[id] * embedding_vec_size * sizeof(float),
                                cudaMemcpyDeviceToHost));
      value_offset += count[id] * embedding_vec_size;
    }

    for (int id = 0; id < local_gpu_count; id++) {
      context.set_device((*device_resources)[id]->get_device_id());

      CK_CUDA_THROW_(cudaFree(d_hash_table_key[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_value_index[id]));
      CK_CUDA_THROW_(cudaFree(d_hash_table_value[id]));
      CK_CUDA_THROW_(cudaFree(d_dump_counter[id]));
    }

    return;
  }
}; // end of SparseEmbeddingHashFunctors

}  // end of namespace HugeCTR
