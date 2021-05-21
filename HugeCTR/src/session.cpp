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
    CK_THROW_(Error_t::InvalidEnv, "Device Compute Capability is low");
  } else if (major == min_major && minor < min_minor) {
    CK_THROW_(Error_t::InvalidEnv, "Device Compute Capability is low");
  }
  return;
}

}  // end namespace

Session::Session(const SolverParser& solver_config, const std::string& config_file,
                 bool use_model_oversubscriber, const std::string temp_embedding_dir)
    : resource_manager_(ResourceManager::create(solver_config.vvgpu, solver_config.seed,
                                                solver_config.device_layout)),
      solver_config_(solver_config) {
  for (auto dev : resource_manager_->get_local_gpu_device_id_list()) {
    if (solver_config.use_mixed_precision) {
      check_device(dev, 7,
                   0);  // to support mixed precision training earliest supported device is CC=70
    } else {
      check_device(dev, 6, 0);  // earliest supported device is CC=60
    }
  }

  parser_.reset(new Parser(config_file, solver_config.batchsize, solver_config.batchsize_eval,
                           solver_config.num_epochs < 1, solver_config.i64_input_key,
                           solver_config.use_mixed_precision, solver_config.enable_tf32_compute, solver_config.scaler,
                           solver_config.use_algorithm_search, solver_config.use_cuda_graph));

  parser_->create_pipeline(init_data_reader_,train_data_reader_, evaluate_data_reader_, embeddings_, networks_,
                           resource_manager_, exchange_wgrad_);

#ifndef DATA_READING_TEST
  #pragma omp parallel num_threads(networks_.size())
  {
    size_t id = omp_get_thread_num();
    networks_[id]->initialize();
    if (solver_config.use_algorithm_search) {
      networks_[id]->search_algorithm();
    }
  }
#endif

  init_or_load_params_for_dense_(solver_config.model_file);
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

  if (solver_config_.use_holistic_cuda_graph) {
    train_graph_.initialized.resize(networks_.size(), false);
    train_graph_.instance.resize(networks_.size());
    for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
      auto& gpu_resource = resource_manager_->get_local_gpu(i);
      CudaCPUDeviceContext context(gpu_resource->get_device_id());
      cudaEvent_t event;
      CK_CUDA_THROW_(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      train_graph_.fork_event.push_back(event);
    }
  }

  // TODO: currently it is only for HE
  if (embeddings_.size() == 1) {
    auto lr_scheds = embeddings_[0]->get_learning_rate_schedulers();
    for (size_t i = 0; i < lr_scheds.size(); i++) {
      networks_[i]->set_learning_rate_scheduler(lr_scheds[i]);
    }
  }
}

Error_t Session::initialize() {
  try {
    parser_->initialize_pipeline(init_data_reader_, embeddings_, resource_manager_, exchange_wgrad_);
    // TODO: find out if deleting the init reader is faster than keeping it
    //init_data_reader_.reset();
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    return rt_err.get_error();
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return Error_t::UnspecificError;
  }
#ifdef ENABLE_MPI
  if (resource_manager_->get_num_process() > 1) {
    resource_manager_->set_ready_to_transfer();
  }
#endif
  return Error_t::Success;
}

void Session::exchange_wgrad(size_t device_id)
{
  auto& gpu_resource = resource_manager_->get_local_gpu(device_id);
  CudaCPUDeviceContext context(gpu_resource->get_device_id());
  PROFILE_RECORD("exchange_wgrad.start", gpu_resource->get_stream(), false);
  exchange_wgrad_->allreduce(device_id, gpu_resource->get_stream());
  PROFILE_RECORD("exchange_wgrad.stop", gpu_resource->get_stream(), false);
}

/**
 * load the model (binary) from model_file.
 * In model file, model should be saved as the sequence as discribed in configure file.
 **/
Error_t Session::init_or_load_params_for_dense_(const std::string& model_file) {
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
    } else {
      for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
        networks_[i]->init_params(i);
      }

      for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
        CK_CUDA_THROW_(cudaStreamSynchronize(resource_manager_->get_local_gpu(i)->get_stream()));
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

void Session::train_overlapped() {
  auto change_state = [] (TrainState* state) -> bool {
    switch(state->state) {
      case TrainState_t::Init:
        state->state = TrainState_t::BottomMLPFprop;
        break;
      case TrainState_t::BottomMLPFprop:
        state->state = TrainState_t::TopMLPFprop;
        break;
      case TrainState_t::TopMLPFprop:
        state->state = TrainState_t::TopMLPBprop;
        break;
      case TrainState_t::TopMLPBprop:
        state->state = TrainState_t::BottomMLPBprop;
        break;
      case TrainState_t::BottomMLPBprop:
        state->state = TrainState_t::MLPExchangeWgrad;
        break;
      case TrainState_t::MLPExchangeWgrad:
        state->state = TrainState_t::MLPUpdate;
        break;
      case TrainState_t::MLPUpdate:
        state->state = TrainState_t::Finalize;
        break;
      case TrainState_t::Finalize:
        return false;
      default:
        CK_THROW_(Error_t::InvalidEnv, "session state reached invalid status");
    }
    return true;
  };

  auto scheduled_reader = dynamic_cast<IDataReaderWithScheduling*>(train_data_reader_.get());
#pragma omp parallel num_threads(resource_manager_->get_local_gpu_count())
  {
    size_t id = omp_get_thread_num();
    auto device_id = resource_manager_->get_local_gpu(id)->get_device_id();
    auto stream = resource_manager_->get_local_gpu(id)->get_stream();
    CudaCPUDeviceContext context(device_id);
    long long current_batchsize_per_device = train_data_reader_->get_current_batchsize_per_device(id);
      
    TrainState state;
    auto sync = [&state, &stream, id] () {
      if (state.event) { CK_CUDA_THROW_(cudaStreamWaitEvent(stream, *state.event)); }
      state.event = nullptr;
    };

    auto schedule_reader = [&, this] (TrainState_t expected) {
      if (scheduled_reader && state.state == expected) {
        if (solver_config_.use_holistic_cuda_graph) {
          scheduled_reader->schedule_here_graph(stream, id);
        } else {
          scheduled_reader->schedule_here(stream, id);
        }
      }
    };

    auto do_it = [&, this](int id, int batch_size) {
      if (solver_config_.use_holistic_cuda_graph) {
        CK_CUDA_THROW_(cudaEventRecord(train_graph_.fork_event[id], stream));
        state.event = &train_graph_.fork_event[id];
      }

      // Network just runs unconditionally
      // Embedding manages events from the networks and waits if necessary
      // Session inserts a wait if it gets a non-null event from the embedding

      do {
        state = embeddings_[0]->train(true, id, state);
        sync();
        if (resource_manager_->get_num_process() == 1) {
          schedule_reader(TrainState_t::TopMLPFprop);
        }
        state = networks_[id]->train(
          batch_size,
          [this, id] () { this->exchange_wgrad(id); },
          state
        );
        if (resource_manager_->get_num_process() > 1) {
          schedule_reader(TrainState_t::TopMLPFprop);
        }
      } while(change_state(&state));
      sync();
    };

    if (solver_config_.use_holistic_cuda_graph) {
      if (!train_graph_.initialized[id]) {
        cudaGraph_t graph;
        CK_CUDA_THROW_(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
        do_it(id, current_batchsize_per_device);
        CK_CUDA_THROW_(cudaStreamEndCapture(stream, &graph));
        CK_CUDA_THROW_(cudaGraphInstantiate(&train_graph_.instance[id], graph, NULL, NULL, 0));
        train_graph_.initialized[id] = true;
      }
      CK_CUDA_THROW_(cudaGraphLaunch(train_graph_.instance[id], stream));
      if (scheduled_reader) {
        scheduled_reader->update_schedule_graph(id);
      }
    }
    else {
      do_it(id, current_batchsize_per_device);
    }
  }
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
    #pragma omp parallel num_threads(networks_.size())
    {
      size_t id = omp_get_thread_num();
      CudaCPUDeviceContext ctx(resource_manager_->get_local_gpu(id)->get_device_id());
      cudaStreamSynchronize(resource_manager_->get_local_gpu(id)->get_stream());
    }
    train_data_reader_->ready_to_collect();
#ifdef ENABLE_PROFILING
    global_profiler.iter_check();
#endif


    // If true we're gonna use overlaping, if false we use default
    if (solver_config_.use_overlapped_pipeline) {
      train_overlapped();
    } else {
      for (const auto& one_embedding : embeddings_) {
        one_embedding->forward(true);
      }

      // Network forward / backward
      if (networks_.size() > 1) {
        // execute dense forward and backward with multi-cpu threads
        #pragma omp parallel num_threads(networks_.size())
        {
          size_t id = omp_get_thread_num();
          long long current_batchsize_per_device =
              train_data_reader_->get_current_batchsize_per_device(id);
          networks_[id]->train(current_batchsize_per_device);
          const auto& local_gpu = resource_manager_->get_local_gpu(id);
          local_gpu->set_compute_event_sync(local_gpu->get_stream());
          local_gpu->wait_on_compute_event(local_gpu->get_comp_overlap_stream());
        }
      } else if (resource_manager_->get_global_gpu_count() > 1) {
        long long current_batchsize_per_device =
            train_data_reader_->get_current_batchsize_per_device(0);
        networks_[0]->train(current_batchsize_per_device);
        const auto& local_gpu = resource_manager_->get_local_gpu(0);
        local_gpu->set_compute_event_sync(local_gpu->get_stream());
        local_gpu->wait_on_compute_event(local_gpu->get_comp_overlap_stream());
      } else {
        long long current_batchsize_per_device =
            train_data_reader_->get_current_batchsize_per_device(0);
        networks_[0]->train(current_batchsize_per_device);
        const auto& local_gpu = resource_manager_->get_local_gpu(0);
        local_gpu->set_compute_event_sync(local_gpu->get_stream());
        local_gpu->wait_on_compute_event(local_gpu->get_comp_overlap_stream());
        networks_[0]->update_params();
      }

      // Embedding backward
      for (const auto& one_embedding : embeddings_) {
        one_embedding->backward();
      }

      // Exchange wgrad and update params
      if (networks_.size() > 1) {
        #pragma omp parallel num_threads(networks_.size())
        {
          size_t id = omp_get_thread_num();
          exchange_wgrad(id);
          networks_[id]->update_params();
        }
      } else if (resource_manager_->get_global_gpu_count() > 1) {
        exchange_wgrad(0);
        networks_[0]->update_params();
      } 
      for (const auto& one_embedding : embeddings_) {
        one_embedding->update_params();
      }

      // Join streams
      if (networks_.size() > 1) {
        #pragma omp parallel num_threads(networks_.size())
        {
          size_t id = omp_get_thread_num();
          const auto& local_gpu = resource_manager_->get_local_gpu(id);
          local_gpu->set_compute2_event_sync(local_gpu->get_comp_overlap_stream());
          local_gpu->wait_on_compute2_event(local_gpu->get_stream());
        }
      }
      else {
        const auto& local_gpu = resource_manager_->get_local_gpu(0);
        local_gpu->set_compute2_event_sync(local_gpu->get_comp_overlap_stream());
        local_gpu->wait_on_compute2_event(local_gpu->get_stream());
      }
      return true;
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
  return true;
}

bool Session::eval(int eval_batch) {
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
    for (const auto& one_embedding : embeddings_) {
      one_embedding->forward(false, eval_batch);
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

    if (resource_manager_->is_master_process()) {
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

    // Collect all the loss from every network and average
    for (auto& network : networks_) {
      loss_sum += network->get_loss();
    }
    if (resource_manager_->get_num_process() > 1) {
#ifdef ENABLE_MPI
      CK_MPI_THROW_(MPI_Reduce(&loss_sum, &loss_reduced, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));
#endif
    } else {
      loss_reduced = loss_sum;
    }
    *loss = loss_reduced / resource_manager_->get_global_gpu_count();
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
