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

#include <core23/logger.hpp>
#include <embeddings/sparse_embedding_functors.hpp>
#include <io/filesystem.hpp>
#include <utils.hpp>

namespace HugeCTR {

template <typename TypeEmbeddingComp>
std::vector<Tensors2<TypeEmbeddingComp>> SparseEmbeddingFunctors::get_opt_states(
    const std::vector<OptimizerTensor<TypeEmbeddingComp>>& opt_tensors_, Optimizer_t optimizer_type,
    size_t local_gpu_count) {
  std::vector<Tensors2<TypeEmbeddingComp>> opt_states;
  opt_states.resize(local_gpu_count);

  for (size_t i = 0; i < local_gpu_count; ++i) {
    switch (optimizer_type) {
      case Optimizer_t::Ftrl:
        opt_states[i].push_back(opt_tensors_[i].opt_n_tensors_);
        opt_states[i].push_back(opt_tensors_[i].opt_z_tensors_);
        break;

      case Optimizer_t::Adam:
        opt_states[i].push_back(opt_tensors_[i].opt_m_tensors_);
        opt_states[i].push_back(opt_tensors_[i].opt_v_tensors_);
        break;

      case Optimizer_t::AdaGrad:
        opt_states[i].push_back(opt_tensors_[i].opt_accm_tensors_);
        break;

      case Optimizer_t::MomentumSGD:
        opt_states[i].push_back(opt_tensors_[i].opt_momentum_tensors_);
        break;

      case Optimizer_t::Nesterov:
        opt_states[i].push_back(opt_tensors_[i].opt_accm_tensors_);
        break;

      case Optimizer_t::SGD:
        break;

      default:
        throw std::runtime_error("[HCDEBUG][ERROR] Runtime error: Invalid optimizer type\n");
    }
  }

  std::vector<Tensors2<TypeEmbeddingComp>> transpose_opt_states;
  if (opt_states[0].size() > 0) {
    transpose_opt_states.resize(opt_states[0].size());
    for (size_t i = 0; i < opt_states[0].size(); ++i) {
      transpose_opt_states[i].resize(opt_states.size());
      for (size_t j = 0; j < opt_states.size(); ++j) {
        transpose_opt_states[i][j] = opt_states[j][i];
      }
    }
  }
  return transpose_opt_states;
}

template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::dump_opt_states(
    std::string& write_path, const ResourceManager& resource_manager,
    std::vector<Tensors2<TypeEmbeddingComp>>& opt_states) {
  size_t local_gpu_count = resource_manager.get_local_gpu_count();

  bool append_flag = false;
  CudaDeviceContext context;
  for (auto& opt_state : opt_states) {
    size_t total_size = 0;
    for (size_t id = 0; id < local_gpu_count; id++) {
      total_size += opt_state[id].get_size_in_bytes();
    }
    size_t max_size = total_size;

#ifdef ENABLE_MPI
    bool is_master_process = resource_manager.is_master_process();
    HCTR_MPI_THROW(MPI_Reduce(is_master_process ? MPI_IN_PLACE : &max_size, &max_size,
                              sizeof(size_t), MPI_CHAR, MPI_MAX,
                              resource_manager.get_master_process_id(), MPI_COMM_WORLD));
#endif

    std::unique_ptr<char[]> h_opt_state(new char[max_size]);
    size_t offset = 0;
    for (size_t id = 0; id < local_gpu_count; id++) {
      size_t local_size = opt_state[id].get_size_in_bytes();
      auto& local_gpu = resource_manager.get_local_gpu(id);
      context.set_device(local_gpu->get_device_id());
      HCTR_LIB_THROW(cudaMemcpyAsync(h_opt_state.get() + offset, opt_state[id].get_ptr(),
                                     local_size, cudaMemcpyDeviceToHost, local_gpu->get_stream()));
      offset += local_size;
    }
    sync_all_gpus(resource_manager);
    int pid = resource_manager.get_process_id();
    if (resource_manager.is_master_process()) {
      HCTR_LOG_S(INFO, WORLD) << "Rank" << pid << ": Write optimzer state to file" << std::endl;
      auto fs = FileSystemBuilder::build_unique_by_path(write_path);
      if (!append_flag) {
        fs->write(write_path, h_opt_state.get(), total_size, true);
        append_flag = true;
      } else {
        fs->write(write_path, h_opt_state.get(), total_size, false);
      }
    }
#ifdef ENABLE_MPI
    else {
      HCTR_LOG_S(INFO, WORLD) << "Rank" << pid << ": Send optimzer state to master node"
                              << std::endl;
      int tag = (pid << 8) | 0xBA;
      HCTR_MPI_THROW(MPI_Send(h_opt_state.get(), total_size, MPI_CHAR,
                              resource_manager.get_master_process_id(), tag, MPI_COMM_WORLD));
    }

    if (resource_manager.is_master_process()) {
      for (int r = 1; r < resource_manager.get_num_process(); r++) {
        HCTR_LOG_S(INFO, WORLD) << "Rank" << pid << ": Recv optimzer state from rank" << r
                                << ", and write to file" << std::endl;
        int tag = (r << 8) | 0xBA;
        int recv_size = 0;
        MPI_Status status;
        HCTR_MPI_THROW(MPI_Probe(r, tag, MPI_COMM_WORLD, &status));
        HCTR_MPI_THROW(MPI_Get_count(&status, MPI_CHAR, &recv_size));
        HCTR_MPI_THROW(MPI_Recv(h_opt_state.get(), recv_size, MPI_CHAR, r, tag, MPI_COMM_WORLD,
                                MPI_STATUS_IGNORE));

        auto fs = FileSystemBuilder::build_unique_by_path(write_path);
        if (!append_flag) {
          fs->write(write_path, h_opt_state.get(), recv_size, true);
          append_flag = true;
        } else {
          fs->write(write_path, h_opt_state.get(), recv_size, false);
        }
      }
    }
#endif
    HCTR_LOG(INFO, ROOT, "Done\n");
  }
}

template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::load_opt_states(
    std::string& read_path, const ResourceManager& resource_manager,
    std::vector<Tensors2<TypeEmbeddingComp>>& opt_states) {
  size_t local_gpu_count = resource_manager.get_local_gpu_count();

  CudaDeviceContext context;
  size_t fs_cursor = 0;
  for (auto& opt_state : opt_states) {
    size_t total_size = 0;
    for (size_t id = 0; id < local_gpu_count; id++) {
      total_size += opt_state[id].get_size_in_bytes();
    }
    int pid = resource_manager.get_process_id();

    auto h2d_op = [&opt_state, &resource_manager, &context](char* h_opt_state) {
      size_t offset = 0;
      for (size_t id = 0; id < resource_manager.get_local_gpu_count(); id++) {
        size_t local_size = opt_state[id].get_size_in_bytes();
        auto& local_gpu = resource_manager.get_local_gpu(id);
        context.set_device(local_gpu->get_device_id());
        HCTR_LIB_THROW(cudaMemcpyAsync(opt_state[id].get_ptr(), h_opt_state + offset, local_size,
                                       cudaMemcpyHostToDevice, local_gpu->get_stream()));
        offset += local_size;
      }
    };

    std::unique_ptr<size_t[]> proc_sizes(new size_t[resource_manager.get_num_process()]);
    proc_sizes[0] = total_size;
#ifdef ENABLE_MPI
    HCTR_MPI_THROW(MPI_Gather(&total_size, sizeof(size_t), MPI_CHAR, proc_sizes.get(),
                              sizeof(size_t), MPI_CHAR, 0, MPI_COMM_WORLD));
#endif

    if (resource_manager.is_master_process()) {
      size_t sum_sizes = 0;
      size_t max_size = 0;
      for (int i = 0; i < resource_manager.get_num_process(); ++i) {
        sum_sizes += proc_sizes[i];
        if (proc_sizes[i] > max_size) {
          max_size = proc_sizes[i];
        }
      }
      std::unique_ptr<char[]> h_opt_state(new char[max_size]);
      HCTR_LOG_S(INFO, WORLD) << "Rank" << pid << ": Read optimzer state from file" << std::endl;

      auto fs = FileSystemBuilder::build_unique_by_path(read_path);
      fs->read(read_path, h_opt_state.get(), total_size, fs_cursor);
      fs_cursor += total_size;

      h2d_op(h_opt_state.get());
      sync_all_gpus(resource_manager);

#ifdef ENABLE_MPI
      for (int r = 1; r < resource_manager.get_num_process(); r++) {
        HCTR_LOG_S(INFO, WORLD) << "Rank" << pid << ": Read from file"
                                << ", and send optimzer state to rank" << r << std::endl;
        fs->read(read_path, h_opt_state.get(), proc_sizes[r], fs_cursor);
        fs_cursor += proc_sizes[r];
        int tag = (r << 8) | 0xAB;
        HCTR_MPI_THROW(
            MPI_Send(h_opt_state.get(), proc_sizes[r], MPI_CHAR, r, tag, MPI_COMM_WORLD));
      }
#endif
    }
#ifdef ENABLE_MPI
    else {
      HCTR_LOG_S(INFO, WORLD) << "Rank" << pid
                              << ": Recv optimzer state from master node, and write to GPUs"
                              << std::endl;
      int mid = resource_manager.get_master_process_id();
      int tag = (pid << 8) | 0xAB;
      int recv_size = 0;
      MPI_Status status;
      HCTR_MPI_THROW(MPI_Probe(mid, tag, MPI_COMM_WORLD, &status));
      HCTR_MPI_THROW(MPI_Get_count(&status, MPI_CHAR, &recv_size));
      std::unique_ptr<char[]> h_opt_state(new char[recv_size]);
      auto fs = FileSystemBuilder::build_unique_by_path(read_path);
      fs->read(read_path, h_opt_state.get(), recv_size, fs_cursor);
      HCTR_MPI_THROW(MPI_Recv(h_opt_state.get(), recv_size, MPI_CHAR, mid, tag, MPI_COMM_WORLD,
                              MPI_STATUS_IGNORE));
      h2d_op(h_opt_state.get());
      sync_all_gpus(resource_manager);
    }
#endif
    HCTR_LOG(INFO, ROOT, "Done\n");
  }
}
template std::vector<Tensors2<float>> SparseEmbeddingFunctors::get_opt_states(
    const std::vector<OptimizerTensor<float>>& opt_tensors_, Optimizer_t optimizer_type,
    size_t local_gpu_count);

template std::vector<Tensors2<__half>> SparseEmbeddingFunctors::get_opt_states(
    const std::vector<OptimizerTensor<__half>>& opt_tensors_, Optimizer_t optimizer_type,
    size_t local_gpu_count);

template void SparseEmbeddingFunctors::dump_opt_states<float>(
    std::string& write_path, const ResourceManager& resource_manager,
    std::vector<Tensors2<float>>& opt_states);

template void SparseEmbeddingFunctors::dump_opt_states<__half>(
    std::string& write_path, const ResourceManager& resource_manager,
    std::vector<Tensors2<__half>>& opt_states);

template void SparseEmbeddingFunctors::load_opt_states<float>(
    std::string& read_path, const ResourceManager& resource_manager,
    std::vector<Tensors2<float>>& opt_states);

template void SparseEmbeddingFunctors::load_opt_states<__half>(
    std::string& read_path, const ResourceManager& resource_manager,
    std::vector<Tensors2<__half>>& opt_states);

}  // namespace HugeCTR
