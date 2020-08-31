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

#include <nvToolsExt.h>
#include <algorithm>
#include <embedding.hpp>
#include <random>
#include <session.hpp>
#include <string>
#include <utils.hpp>

// #define DATA_READING_TEST

namespace HugeCTR {

namespace {

std::string generate_random_file_name() {
  std::string ch_set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<> ch_dist(0, ch_set.size() - 1);
  std::uniform_int_distribution<> len_dist(ch_set.size() / 5, ch_set.size() / 3);

  int length = len_dist(rng);
  auto get_ch = [&ch_set, &ch_dist, &rng]() { return ch_set[ch_dist(rng)]; };

  std::string ret(length, 0);
  std::generate_n(ret.begin(), length, get_ch);
  return ret;
}

/**
 * check if device is avaliable.
 * lowest avaliable CC is min_major.min_minor
 * @param device_id gpu id
 * @param min_major minimum compute compatibility required
 * @param min_minor minimum compute compatibility required
 */
static void check_device(int device_id, int min_major, int min_minor) {
  int device_count = 0;
  CK_CUDA_THROW_(cudaGetDeviceCount(&device_count));
  if (device_id >= device_count) {
    CK_THROW_(Error_t::WrongInput, "device is not avaliable");
  }
  CudaDeviceContext context(device_id);
  cudaDeviceProp deviceProp;
  if (cudaGetDeviceProperties(&deviceProp, device_id) != cudaSuccess) {
    CK_THROW_(Error_t::InvalidEnv, "Invalid device:" + std::to_string(device_id));
    return;
  }
  std::cout << "Device " << device_id << ": " << deviceProp.name << std::endl;
  int major = deviceProp.major;
  int minor = deviceProp.minor;
  if (major < min_major) {
    CK_THROW_(Error_t::InvalidEnv, "Device Compute Compacity is low");
  } else if (major == min_major && minor < min_minor) {
    CK_THROW_(Error_t::InvalidEnv, "Device Compute Compacity is low");
  }
  return;
}

}  // end namespace

template <typename TypeKey>
SessionImpl<TypeKey>::SessionImpl(const SolverParser& solver_config)
    : resource_manager_(ResourceManager::create(solver_config.vvgpu, solver_config.seed)) {
  for (auto dev : resource_manager_->get_local_gpu_device_id_list()) {
    if (solver_config.use_mixed_precision) {
      check_device(dev, 7,
                   0);  // to support mixed precision training earliest supported device is CC=70
    } else {
      check_device(dev, 6, 0);  // earliest supported device is CC=60
    }
  }

  Parser parser(solver_config.configure_file, solver_config.batchsize, solver_config.batchsize_eval,
                solver_config.use_mixed_precision, solver_config.scaler,
                solver_config.use_algorithm_search);

  parser.create_pipeline(data_reader_, data_reader_eval_, embedding_, networks_, resource_manager_);

  // init networks.
  std::string TMP_DENSE_NAME;
  if (resource_manager_->get_pid() == 0) {
    TMP_DENSE_NAME = "./" + generate_random_file_name();
    networks_[0]->init_params(TMP_DENSE_NAME);
  }
#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  int length = (resource_manager_->get_pid() == 0) ? TMP_DENSE_NAME.length() : 0;
  MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (resource_manager_->get_pid() != 0) {
    TMP_DENSE_NAME.resize(length);
  }
  MPI_Bcast(const_cast<char*>(TMP_DENSE_NAME.data()), length, MPI_CHAR, 0, MPI_COMM_WORLD);
  MESSAGE_("tmp dense file name: " + TMP_DENSE_NAME);
#endif
  for (auto& network : networks_) {
    network->upload_params_to_device(TMP_DENSE_NAME);
  }
#ifdef ENABLE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  if (resource_manager_->get_pid() == 0) {
    if (std::remove(TMP_DENSE_NAME.c_str()) != 0) {
      CK_THROW_(Error_t::WrongInput,
                TMP_DENSE_NAME + " cannot be removed. (reason: " + std::strerror(errno) + ")");
    }
  }

  load_params_for_dense_(solver_config.model_file);
  init_or_load_params_for_sparse_(solver_config.embedding_files);

  int num_total_gpus = resource_manager_->get_global_gpu_count();
  for (const auto& metric : solver_config.metrics_spec) {
    metrics_.emplace_back(
        std::move(metrics::Metric::Create(metric.first, solver_config.use_mixed_precision,
                                          solver_config.batchsize_eval / num_total_gpus,
                                          solver_config.eval_batches, resource_manager_)));
  }
}

/**
 * load the model (binary) from model_file.
 * In model file, model should be saved as the sequence as discribed in configure file.
 **/
template <typename TypeKey>
Error_t SessionImpl<TypeKey>::load_params_for_dense_(const std::string& model_file) {
  try {
    if (!model_file.empty()) {
      std::ifstream model_stream(model_file, std::ifstream::binary);
      if (!model_stream.is_open()) {
        CK_THROW_(Error_t::WrongInput, "Cannot open dense model file");
      }
      std::unique_ptr<float[]> weight(new float[networks_[0]->get_params_num()]());
      model_stream.read(reinterpret_cast<char*>(weight.get()),
                        networks_[0]->get_params_num() * sizeof(float));

      std::cout << "Loading dense model: " << model_file << std::endl;
      for (auto& network : networks_) {
        network->upload_params_to_device(weight.get());
      }
      model_stream.close();
    }
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    return rt_err.get_error();
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

/**
 * load the model (binary) from model_file.
 * In model file, model should be saved as
 * the sequence as discribed in configure file.
 **/
template <typename TypeKey>
Error_t SessionImpl<TypeKey>::init_or_load_params_for_sparse_(
    const std::vector<std::string>& embedding_model_files) {
  try {
    for (size_t i = 0; i < embedding_.size(); i++) {
      if (i < embedding_model_files.size()) {
        std::ifstream embedding_stream(embedding_model_files[i], std::ifstream::binary);
        if (!embedding_stream.is_open()) {
          CK_THROW_(Error_t::WrongInput, "Cannot open sparse model file");
        }
        std::cout << "Loading sparse model: " << embedding_model_files[i] << std::endl;
        embedding_[i]->upload_params_to_device(embedding_stream);
        embedding_stream.close();
      } else {
        embedding_[i]->init_params();
      }
    }
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    return rt_err.get_error();
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

void network_train_helper(Network* n) {
  try {
    n->train();
    n->exchange_wgrad();
    n->update_params();
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
  return;
}

template <typename TypeKey>
void SessionImpl<TypeKey>::train() {
  try {
#ifndef DATA_READING_TEST
    data_reader_->read_a_batch_to_device_delay_release();
    for (auto& one_embedding : embedding_) {
      one_embedding->forward(true);
    }
    data_reader_->ready_to_collect();
    if (networks_.size() > 1) {
      // execute dense forward and backward with multi-cpu threads
      std::vector<std::future<void>> results(networks_.size());
      for (unsigned int i = 0; i < networks_.size(); i++) {
        results[i] = resource_manager_->get_local_cpu()->get_thread_pool()->push(
            [this, i](int id) { network_train_helper(networks_[i].get()); });
      }
      for (unsigned int i = 0; i < networks_.size(); i++) {
        results[i].get();
      }
    } else if (networks_.size() == 1) {
      networks_[0]->train();
      networks_[0]->update_params();
    } else {
      assert(!"networks_.size() should not less than 1.");
    }
    for (auto& one_embedding : embedding_) {
      one_embedding->backward();
      one_embedding->update_params();
    }
#else
    data_reader_->read_a_batch_to_device();
#endif
  } catch (const internal_runtime_error& err) {
    std::cerr << err.what() << std::endl;
    throw err;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw err;
  }
}

void network_eval_helper(int id, Network* n, metrics::Metrics& metrics) {
  try {
    n->eval();

    for (auto& metric : metrics) {
      metric->local_reduce(id, n->get_raw_metrics());
    }
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
  }
}

template <typename TypeKey>
void SessionImpl<TypeKey>::eval() {
  try {
    if (data_reader_eval_ == nullptr) return;
    long long current_batchsize = data_reader_eval_->read_a_batch_to_device();
    for (auto& metric : metrics_) {
      metric->set_current_batch_size(current_batchsize);
    }

#ifndef DATA_READING_TEST
    for (auto& one_embedding : embedding_) {
      one_embedding->forward(false);
    }

    if (networks_.size() > 1) {
      std::vector<std::future<void>> results(networks_.size());
      for (unsigned int i = 0; i < networks_.size(); i++) {
        results[i] = resource_manager_->get_local_cpu()->get_thread_pool()->push(
            [this, i](int id) { network_eval_helper(i, networks_[i].get(), metrics_); });
      }
      for (unsigned int i = 0; i < networks_.size(); i++) {
        results[i].get();
      }
    } else if (networks_.size() == 1) {
      network_eval_helper(0, networks_[0].get(), metrics_);
    } else {
      assert(!"networks_.size() should not less than 1.");
    }
#endif

    for (auto& metric : metrics_) {
      metric->global_reduce(networks_.size());
    }

  } catch (const internal_runtime_error& err) {
    std::cerr << err.what() << std::endl;
    throw err;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw err;
  }
}

template <typename TypeKey>
std::vector<std::pair<std::string, float>> SessionImpl<TypeKey>::get_eval_metrics() {
  std::vector<std::pair<std::string, float>> metrics;
  for (auto& metric : metrics_) {
    metrics.push_back(std::make_pair(metric->name(), metric->finalize_metric()));
  }
  return metrics;
}

template <typename TypeKey>
Error_t SessionImpl<TypeKey>::download_params_to_files(std::string prefix, int iter) {
  std::string snapshot_dense_name = prefix + "_dense_" + std::to_string(iter) + ".model";
  std::vector<std::string> snapshot_sparse_names;
  if (iter <= 0) {
    return Error_t::WrongInput;
  }

  for (unsigned int i = 0; i < embedding_.size(); i++) {
    snapshot_sparse_names.push_back(prefix + std::to_string(i) + "_sparse_" + std::to_string(iter) +
                                    ".model");
  }
  return download_params_to_files_(snapshot_dense_name, snapshot_sparse_names);
}

template <typename TypeKey>
Error_t SessionImpl<TypeKey>::download_params_to_files_(
    std::string weights_file, const std::vector<std::string>& embedding_files) {
  try {
    {
      int i = 0;
      for (auto& embedding_file : embedding_files) {
        std::ofstream out_stream_embedding(embedding_file, std::ofstream::binary);
        embedding_[i]->download_params_to_host(out_stream_embedding);
        out_stream_embedding.close();
        i++;
      }
    }
    int pid = 0;
#ifdef ENABLE_MPI
    int numprocs = 1;
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
#endif
    if (pid == 0) {
      std::ofstream out_stream_weight(weights_file, std::ofstream::binary);
      networks_[0]->download_params_to_host(out_stream_weight);
      std::string no_trained_params = networks_[0]->get_no_trained_params_in_string();
      if (no_trained_params.length() != 0) {
        std::string ntp_file = weights_file + ".ntp.json";
        std::ofstream out_stream_ntp(ntp_file, std::ofstream::out);
        out_stream_ntp.write(no_trained_params.c_str(), no_trained_params.length());
        out_stream_ntp.close();
      }
      out_stream_weight.close();
    }

  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    return rt_err.get_error();
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

template <typename TypeKey>
Error_t SessionImpl<TypeKey>::get_current_loss(float* loss) {
  try {
    float loss_sum = 0.f;
    float loss_reduced = 0.f;
    int numprocs = 1;
#ifdef ENABLE_MPI
    int pid = 0;
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
#endif
    // Collect all the loss from every network and average
    for (auto& network : networks_) {
      loss_sum += network->get_loss();
    }
    if (numprocs > 1) {
#ifdef ENABLE_MPI
      CK_MPI_THROW_(MPI_Reduce(&loss_sum, &loss_reduced, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));
#endif
    } else {
      loss_reduced = loss_sum;
    }
    *loss = loss_reduced / networks_.size() / numprocs;
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    return rt_err.get_error();
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

template <typename TypeKey>
void SessionImpl<TypeKey>::check_overflow() const {
  for (auto& one_embedding : embedding_) {
    one_embedding->check_overflow();
  }
}

template <typename TypeKey>
SessionImpl<TypeKey>::~SessionImpl() {
  try {
    for (auto device : resource_manager_->get_local_gpu_device_id_list()) {
      CudaDeviceContext context(device);
      CK_CUDA_THROW_(cudaDeviceSynchronize());
    }

  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
  }
}

template class SessionImpl<unsigned int>;
template class SessionImpl<long long>;

std::shared_ptr<Session> Session::Create(const SolverParser& solver_config) {
  std::shared_ptr<Session> session;
  if (solver_config.i64_input_key) {
    session.reset(new SessionImpl<long long>(solver_config));
  } else {
    session.reset(new SessionImpl<unsigned int>(solver_config));
  }
  return session;
}

}  // namespace HugeCTR
