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
#include <omp.h>

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

Session::Session(const SolverParser& solver_config, const std::string& config_file,
                 bool use_model_oversubscriber, const std::string temp_embedding_dir)
    : resource_manager_(ResourceManager::create(solver_config.vvgpu, solver_config.seed)),
    use_mixed_precision_(solver_config.use_mixed_precision), batchsize_eval(solver_config.batchsize_eval) {
  for (auto dev : resource_manager_->get_local_gpu_device_id_list()) {
    if (solver_config.use_mixed_precision) {
      check_device(dev, 7,
                   0);  // to support mixed precision training earliest supported device is CC=70
    } else {
      check_device(dev, 6, 0);  // earliest supported device is CC=60
    }
  }

  Parser parser(config_file, solver_config.batchsize, solver_config.batchsize_eval,
                solver_config.num_epochs < 1, solver_config.i64_input_key,
                solver_config.use_mixed_precision, solver_config.enable_tf32_compute,
                solver_config.scaler, solver_config.use_algorithm_search,
                solver_config.use_cuda_graph);

  parser.create_pipeline(train_data_reader_, evaluate_data_reader_, embeddings_, networks_,
                         resource_manager_);

  // init networks.
  std::string TMP_DENSE_NAME;
  if (resource_manager_->get_pid() == 0) {
    const char* TMP_DENSE_NAME_PREFIX = std::getenv("TMP_DIR");
    if (TMP_DENSE_NAME_PREFIX == nullptr) {
      TMP_DENSE_NAME = "./" + generate_random_file_name();
    } else {
      TMP_DENSE_NAME = std::string(TMP_DENSE_NAME_PREFIX) + "/" + generate_random_file_name();
    }
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
  if (use_model_oversubscriber) {
    if (solver_config.use_mixed_precision) {
      model_oversubscriber_ =
          create_model_oversubscriber_<__half>(solver_config, temp_embedding_dir);
    } else {
      model_oversubscriber_ =
          create_model_oversubscriber_<float>(solver_config, temp_embedding_dir);
    }
  } else {
    init_or_load_params_for_sparse_(solver_config.embedding_files);
  }

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
Error_t Session::load_params_for_dense_(const std::string& model_file) {
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
Error_t Session::init_or_load_params_for_sparse_(
    const std::vector<std::string>& embedding_model_files) {
  try {
    for (size_t i = 0; i < embeddings_.size(); i++) {
      if (i < embedding_model_files.size()) {
        std::ifstream embedding_stream(embedding_model_files[i], std::ifstream::binary);
        if (!embedding_stream.is_open()) {
          CK_THROW_(Error_t::WrongInput, "Cannot open sparse model file");
        }
        std::cout << "Loading sparse model: " << embedding_model_files[i] << std::endl;
        embeddings_[i]->load_parameters(embedding_stream);
        embedding_stream.close();
      } else {
        embeddings_[i]->init_params();
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

bool Session::train() {
  try {
    if (train_data_reader_->is_started() == false) {
      CK_THROW_(Error_t::IllegalCall,
                "Start the data reader first before calling Session::train()");
    }

#ifndef DATA_READING_TEST
    long long current_batchsize = train_data_reader_->read_a_batch_to_device_delay_release();
    if (!current_batchsize) {
      return false;
    }
    train_data_reader_->ready_to_collect();
    for (auto& one_embedding : embeddings_) {
      one_embedding->forward(true);
    }
    if (networks_.size() > 1) {
// execute dense forward and backward with multi-cpu threads
#pragma omp parallel num_threads(networks_.size())
      {
        size_t id = omp_get_thread_num();
        long long current_batchsize_per_device =
            train_data_reader_->get_current_batchsize_per_device(id);
        networks_[id]->train(current_batchsize_per_device);
        networks_[id]->exchange_wgrad();
        networks_[id]->update_params();
      }
    } else if (resource_manager_->get_global_gpu_count() > 1) {
      long long current_batchsize_per_device =
          train_data_reader_->get_current_batchsize_per_device(0);
      networks_[0]->train(current_batchsize_per_device);
      networks_[0]->exchange_wgrad();
      networks_[0]->update_params();
    } else {
      long long current_batchsize_per_device =
          train_data_reader_->get_current_batchsize_per_device(0);
      networks_[0]->train(current_batchsize_per_device);
      networks_[0]->update_params();
    }
    for (auto& one_embedding : embeddings_) {
      one_embedding->backward();
      one_embedding->update_params();
    }
    return true;
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

bool Session::eval() {
  try {
    if (evaluate_data_reader_ == nullptr) return true;

    if (evaluate_data_reader_->is_started() == false) {
      CK_THROW_(Error_t::IllegalCall, "Start the data reader first before calling Session::eval()");
    }

    long long current_batchsize = evaluate_data_reader_->read_a_batch_to_device();
    for (auto& metric : metrics_) {
      metric->set_current_batch_size(current_batchsize);
    }
    if (!current_batchsize) {
      return false;
    }

#ifndef DATA_READING_TEST
    for (auto& one_embedding : embeddings_) {
      one_embedding->forward(false);
    }

    if (networks_.size() > 1) {
#pragma omp parallel num_threads(networks_.size())
      {
        size_t id = omp_get_thread_num();
        networks_[id]->eval();
        for (auto& metric : metrics_) {
          metric->local_reduce(id, networks_[id]->get_raw_metrics());
        }
      }

    } else if (networks_.size() == 1) {
      networks_[0]->eval();
      for (auto& metric : metrics_) {
        metric->local_reduce(0, networks_[0]->get_raw_metrics());
      }
    } else {
      assert(!"networks_.size() should not less than 1.");
    }
#endif

    for (auto& metric : metrics_) {
      metric->global_reduce(networks_.size());
    }
    return true;
  } catch (const internal_runtime_error& err) {
    std::cerr << err.what() << std::endl;
    throw err;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw err;
  }
}

std::vector<std::pair<std::string, float>> Session::get_eval_metrics() {
  std::vector<std::pair<std::string, float>> metrics;
  for (auto& metric : metrics_) {
    metrics.push_back(std::make_pair(metric->name(), metric->finalize_metric()));
  }
  return metrics;
}

Error_t Session::download_params_to_files(std::string prefix, int iter) {
  std::string snapshot_dense_name = prefix + "_dense_" + std::to_string(iter) + ".model";
  std::vector<std::string> snapshot_sparse_names;
  if (iter <= 0) {
    return Error_t::WrongInput;
  }

  for (unsigned int i = 0; i < embeddings_.size(); i++) {
    snapshot_sparse_names.push_back(prefix + std::to_string(i) + "_sparse_" + std::to_string(iter) +
                                    ".model");
  }
  return download_params_to_files_(snapshot_dense_name, snapshot_sparse_names);
}

Error_t Session::export_predictions(const std::string& output_file_name){
  try {
    
    CudaDeviceContext context;
    const std::vector<int>& local_gpu_device_id_list = resource_manager_->get_local_gpu_device_id_list();
    const size_t global_gpu_count = resource_manager_->get_global_gpu_count();
    const size_t local_gpu_count = resource_manager_->get_local_gpu_count();
    size_t batchsize_eval_per_gpu = batchsize_eval / global_gpu_count;
    size_t total_prediction_count = batchsize_eval_per_gpu * local_gpu_count;
    std::unique_ptr<float[]> local_prediction_result(new float[total_prediction_count]);

    for(unsigned int i = 0; i < networks_.size(); ++i){
      int gpu_id = local_gpu_device_id_list[i];
      context.set_device(gpu_id);

      get_raw_metric_as_host_float_tensor(networks_[i]->get_raw_metrics(), metrics::RawType::Pred, use_mixed_precision_, local_prediction_result.get() + batchsize_eval_per_gpu * i, batchsize_eval_per_gpu);
    }

    std::unique_ptr<float[]> global_prediction_result;
    int numprocs = 1;
    int pid = 0;
#ifdef ENABLE_MPI
      CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
      CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
#endif
    if(numprocs > 1){
#ifdef ENABLE_MPI
      if(pid == 0){
        global_prediction_result.reset(new float[batchsize_eval]);
      }
      CK_MPI_THROW_(MPI_Gather(local_prediction_result.get(), total_prediction_count, MPI_FLOAT, global_prediction_result.get(), total_prediction_count, MPI_FLOAT, 0, MPI_COMM_WORLD));
#endif
    } else {
      global_prediction_result = std::move(local_prediction_result);
    }
    if(pid == 0){
      // write
      std::ofstream predictions_stream(output_file_name, std::ios::out | std::ios::app);
      if(!predictions_stream.is_open()) {
        CK_THROW_(Error_t::WrongInput, "Cannot open output prediction file " + output_file_name);
      }
      for(unsigned int i = 0; i < batchsize_eval; ++i){
        predictions_stream << global_prediction_result[i] << " ";
      }
      predictions_stream.close();
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

Error_t Session::download_params_to_files_(std::string weights_file,
                                           const std::vector<std::string>& embedding_files) {
  try {
    {
      int i = 0;
      for (auto& embedding_file : embedding_files) {
        std::ofstream out_stream_embedding(embedding_file, std::ofstream::binary);
        embeddings_[i]->dump_parameters(out_stream_embedding);
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

Error_t Session::get_current_loss(float* loss) {
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

template <typename TypeEmbeddingComp>
std::shared_ptr<ModelOversubscriber> Session::create_model_oversubscriber_(
    const SolverParser& solver_config, const std::string& temp_embedding_dir) {
  try {
    if (temp_embedding_dir.empty()) {
      CK_THROW_(Error_t::WrongInput, "must provide a directory for storing temporary embedding");
    }

    std::vector<SparseEmbeddingHashParams<TypeEmbeddingComp>> embedding_params;
    return std::shared_ptr<ModelOversubscriber>(
        new ModelOversubscriber(embeddings_, embedding_params, solver_config, temp_embedding_dir));
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw rt_err;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw err;
  }
}

void Session::check_overflow() const {
  for (auto& one_embedding : embeddings_) {
    one_embedding->check_overflow();
  }
}

Session::~Session() {
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

void Session::copy_weights_for_evaluation() {
  for (auto& network : networks_) {
    network->copy_weights_from_train_layers_to_evaluate_layers();
  }
}

}  // namespace HugeCTR
