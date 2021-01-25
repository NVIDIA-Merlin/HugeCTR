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

#include <HugeCTR/pybind/model.hpp>
#include <iomanip>
namespace HugeCTR {

namespace {
std::map<Layer_t, std::string> LAYER_TYPE_TO_STRING = {
  {Layer_t::BatchNorm, "BatchNorm"},
  {Layer_t::BinaryCrossEntropyLoss, "BinaryCrossEntropyLoss"},
  {Layer_t::Concat, "Concat"},
  {Layer_t::CrossEntropyLoss, "CrossEntropyLoss"},
  {Layer_t::Dropout, "Dropout"},
  {Layer_t::ELU, "ELU"},
  {Layer_t::InnerProduct, "InnerProduct"},
  {Layer_t::Interaction, "Interaction"},
  {Layer_t::MultiCrossEntropyLoss, "MultiCrossEntropyLoss"},
  {Layer_t::ReLU, "ReLU"},
  {Layer_t::Reshape, "Reshape"},
  {Layer_t::Sigmoid, "Sigmoid"},
  {Layer_t::Slice, "Slice"},
  {Layer_t::WeightMultiply, "WeightMultiply"},
  {Layer_t::FmOrder2, "FmOrder2"},
  {Layer_t::Add, "Add"},
  {Layer_t::ReduceSum, "ReduceSum"},
  {Layer_t::MultiCross, "MultiCross"},
  {Layer_t::DotProduct, "DotProduct"}};

std::map<Embedding_t, std::string> EMBEDDING_TYPE_TO_STRING = {
    {Embedding_t::DistributedSlotSparseEmbeddingHash, "DistributedHash"},
    {Embedding_t::LocalizedSlotSparseEmbeddingHash, "LocalizedHash"},
    {Embedding_t::LocalizedSlotSparseEmbeddingOneHot, "LocalizedOneHot"}};
/**
 * check if device is avaliable.
 * lowest avaliable CC is min_major.min_minor
 * @param device_id gpu id
 * @param min_major minimum compute compatibility required
 * @param min_minor minimum compute compatibility required
 */

static std::string join(std::vector<std::string>& strs, std::string delim) {
  std::string str;
  const std::vector<std::string>::iterator itlast = strs.end()-1;
  for (auto it = strs.begin(); it != strs.end(); it++) {
    str += *it;
    if (it != itlast) {
        str += delim;
    }
  }
  return str;
}

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

} // end namespace

Input::Input(DataReaderType_t data_reader_type,
       std::string source,
       std::string eval_source,
       Check_t check_type,
       int cache_eval_data,
       int label_dim,
       std::string label_name,
       int dense_dim,
       std::string dense_name,
       long long num_samples,
       long long eval_num_samples,
       bool float_label_dense,
       int num_workers,
       std::vector<long long>& slot_size_array,
       std::vector<DataReaderSparseParam>& data_reader_sparse_param_array,
       std::vector<std::string>& sparse_names)
    : data_reader_type(data_reader_type), source(source), eval_source(eval_source),
      check_type(check_type), cache_eval_data(cache_eval_data), label_dim(label_dim),
      label_name(label_name), dense_dim(dense_dim), dense_name(dense_name),
      num_samples(num_samples), eval_num_samples(eval_num_samples), float_label_dense(float_label_dense),
      num_workers(num_workers), slot_size_array(slot_size_array),
      data_reader_sparse_param_array(data_reader_sparse_param_array), sparse_names(sparse_names) {
  if (data_reader_sparse_param_array.size() != sparse_names.size()) {
    CK_THROW_(Error_t::WrongInput, "Inconsistent size of sparse hyperparameters and sparse names!");
  }
}

SparseEmbedding::SparseEmbedding(Embedding_t embedding_type,
       size_t max_vocabulary_size_per_gpu,
       size_t embedding_vec_size,
       int combiner,
       std::string sparse_embedding_name,
       std::string bottom_name,
       std::vector<size_t>& slot_size_array)
    : embedding_type(embedding_type), max_vocabulary_size_per_gpu(max_vocabulary_size_per_gpu),
      embedding_vec_size(embedding_vec_size), combiner(combiner),
      sparse_embedding_name(sparse_embedding_name), bottom_name(bottom_name),
      slot_size_array(slot_size_array) {}

DenseLayer::DenseLayer(Layer_t layer_type,
            std::vector<std::string>& bottom_names,
            std::vector<std::string>& top_names,
            float factor,
            float eps,
            Initializer_t gamma_init_type,
            Initializer_t beta_init_type,
            float dropout_rate,
            float elu_alpha,
            size_t num_output,
            Initializer_t weight_init_type,
            Initializer_t bias_init_type,
            int num_layers,
            size_t leading_dim,
            bool selected,
            std::vector<int>& selected_slots,
            std::vector<std::pair<int, int>>& ranges,
            std::vector<size_t>& weight_dims,
            size_t out_dim,
            int axis,
            std::vector<float>& target_weight_vec,
            bool use_regularizer,
            Regularizer_t regularizer_type,
            float lambda) 
    : layer_type(layer_type), bottom_names(bottom_names), top_names(top_names),
      factor(factor), eps(eps), gamma_init_type(gamma_init_type),
      beta_init_type(beta_init_type), dropout_rate(dropout_rate), elu_alpha(elu_alpha),
      num_output(num_output), weight_init_type(weight_init_type), bias_init_type(bias_init_type),
      num_layers(num_layers), leading_dim(leading_dim), selected(selected),
      selected_slots(selected_slots), ranges(ranges), weight_dims(weight_dims),
      out_dim(out_dim), axis(axis), target_weight_vec(target_weight_vec),
      use_regularizer(use_regularizer), regularizer_type(regularizer_type), lambda(lambda) {}

Model::Model(const SolverParser& solver, std::shared_ptr<OptParamsBase>& opt_params)
  : solver_(solver) {
  std::cout << "===================================Model Init====================================" << std::endl;
  resource_manager_ = ResourceManager::create(solver.vvgpu, solver.seed);
  for (auto dev : resource_manager_->get_local_gpu_device_id_list()) {
    if (solver_.use_mixed_precision) {
      check_device(dev, 7, 0);  // to support mixed precision training earliest supported device is CC=70
    } else {
      check_device(dev, 6, 0);  // earliest supported device is CC=60
    }
  }
  if (solver.use_mixed_precision != opt_params->use_mixed_precision) {
    CK_THROW_(Error_t::WrongInput, "whether to use mixed precision should be consistent across solver and optimizer");
  }
  int total_gpu_count = resource_manager_->get_global_gpu_count();
  if (0 != solver_.batchsize % total_gpu_count) {
    CK_THROW_(Error_t::WrongInput, "0 != batch_size\%total_gpu_count");
  }
  // reserve networks to be created
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    networks_.emplace_back(new Network(resource_manager_->get_local_cpu(),
                                    resource_manager_->get_local_gpu(i), 
                                    solver_.use_mixed_precision,
                                    solver_.use_cuda_graph));
    blobs_buff_list_.emplace_back(GeneralBuffer2<CudaAllocator>::create());
  }

  
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    train_weight_buff_list_.emplace_back(blobs_buff_list_[i]->create_block<float>());
    train_weight_buff_half_list_.emplace_back(blobs_buff_list_[i]->create_block<__half>());
    wgrad_buff_list_.emplace_back(blobs_buff_list_[i]->create_block<float>());
    wgrad_buff_half_list_.emplace_back(blobs_buff_list_[i]->create_block<__half>());
    evaluate_weight_buff_list_.emplace_back(blobs_buff_list_[i]->create_block<float>());
    evaluate_weight_buff_half_list_.emplace_back(blobs_buff_list_[i]->create_block<__half>());
    wgrad_buff_placeholder_list_.emplace_back(blobs_buff_list_[i]->create_block<float>());
    wgrad_buff_half_placeholder_list_.emplace_back(blobs_buff_list_[i]->create_block<__half>());
  }

  // resize train_tensor_entries_list_ and evaluate_tensor_entries_list_
  train_tensor_entries_list_.resize(resource_manager_->get_local_gpu_count());
  evaluate_tensor_entries_list_.resize(resource_manager_->get_local_gpu_count());

  // initialize optimizer
  if (solver_.use_mixed_precision) {
    OptParamsPy<__half>* opt_params_py = dynamic_cast<OptParamsPy<__half>*>(opt_params.get());
    opt_params_16_.optimizer = opt_params_py->optimizer;
    opt_params_16_.lr = opt_params_py->lr;
    opt_params_16_.update_type = opt_params_py->update_type;
    opt_params_16_.scaler = solver_.scaler;
    opt_params_16_.hyperparams = opt_params_py->hyperparams;
    lr_sch_.reset(new LearningRateScheduler(opt_params_py->lr, opt_params_py->warmup_steps, opt_params_py->decay_start,
                                            opt_params_py->decay_steps, opt_params_py->decay_power, opt_params_py->end_lr));
  } else {
    OptParamsPy<float>* opt_params_py = dynamic_cast<OptParamsPy<float>*>(opt_params.get());
    opt_params_32_.optimizer = opt_params_py->optimizer;
    opt_params_32_.lr = opt_params_py->lr;
    opt_params_32_.update_type = opt_params_py->update_type;
    opt_params_32_.scaler = solver_.scaler;
    opt_params_32_.hyperparams = opt_params_py->hyperparams;
    lr_sch_.reset(new LearningRateScheduler(opt_params_py->lr, opt_params_py->warmup_steps, opt_params_py->decay_start,
                                            opt_params_py->decay_steps, opt_params_py->decay_power, opt_params_py->end_lr));
  }
}

Model::~Model() {
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

void Model::add(Input& input) {
  if (solver_.i64_input_key) {
    add_input<long long>(input, sparse_input_map_64_, train_tensor_entries_list_,
                        evaluate_tensor_entries_list_, train_data_reader_,
                        evaluate_data_reader_, solver_.batchsize,
                        solver_.batchsize_eval, solver_.use_mixed_precision, 
                        solver_.num_epochs < 1, resource_manager_);
  } else {
    add_input<unsigned int>(input, sparse_input_map_32_, train_tensor_entries_list_,
                        evaluate_tensor_entries_list_, train_data_reader_,
                        evaluate_data_reader_, solver_.batchsize,
                        solver_.batchsize_eval, solver_.use_mixed_precision, 
                        solver_.num_epochs < 1, resource_manager_);
  }
}

void Model::add(SparseEmbedding& sparse_embedding) {
  input_output_info_.push_back(std::make_pair(sparse_embedding.bottom_name, sparse_embedding.sparse_embedding_name));
  layer_info_.push_back(EMBEDDING_TYPE_TO_STRING[sparse_embedding.embedding_type]);
  if (solver_.i64_input_key && !solver_.use_mixed_precision) {
    add_sparse_embedding<long long, float>(sparse_embedding, sparse_input_map_64_, train_tensor_entries_list_,
                        evaluate_tensor_entries_list_, embeddings_,
                        resource_manager_, solver_.batchsize,
                        solver_.batchsize_eval, opt_params_32_);
  } else if (solver_.i64_input_key && solver_.use_mixed_precision) {
    add_sparse_embedding<long long, __half>(sparse_embedding, sparse_input_map_64_, train_tensor_entries_list_,
                        evaluate_tensor_entries_list_, embeddings_,
                        resource_manager_, solver_.batchsize,
                        solver_.batchsize_eval, opt_params_16_);
  } else if (!solver_.i64_input_key && !solver_.use_mixed_precision) {
    add_sparse_embedding<unsigned int, float>(sparse_embedding, sparse_input_map_32_, train_tensor_entries_list_,
                        evaluate_tensor_entries_list_, embeddings_,
                        resource_manager_, solver_.batchsize,
                        solver_.batchsize_eval, opt_params_32_);
  } else {
    add_sparse_embedding<unsigned int, __half>(sparse_embedding, sparse_input_map_32_, train_tensor_entries_list_,
                        evaluate_tensor_entries_list_, embeddings_,
                        resource_manager_, solver_.batchsize,
                        solver_.batchsize_eval, opt_params_16_);
  }
}

void Model::add(DenseLayer& dense_layer) {
  std::string input_names = join(dense_layer.bottom_names, " ");
  std::string output_names = join(dense_layer.top_names, " ");
  input_output_info_.push_back(std::make_pair(input_names, output_names));
  layer_info_.push_back(LAYER_TYPE_TO_STRING[dense_layer.layer_type]);
  add_dense_layer(dense_layer, train_tensor_entries_list_, evaluate_tensor_entries_list_,
                resource_manager_, solver_.use_mixed_precision, solver_.enable_tf32_compute,
                solver_.scaler, solver_.use_algorithm_search, solver_.use_cuda_graph,
                networks_, blobs_buff_list_, train_weight_buff_list_, train_weight_buff_half_list_,
                wgrad_buff_list_, wgrad_buff_half_list_, evaluate_weight_buff_list_,
                evaluate_weight_buff_half_list_, wgrad_buff_placeholder_list_,
                wgrad_buff_half_placeholder_list_);
}

void Model::compile() {
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    networks_[i]->optimizer_ = std::move(Optimizer::Create(
        opt_params_32_, train_weight_buff_list_[i]->as_tensor(), wgrad_buff_list_[i]->as_tensor(),
        wgrad_buff_half_list_[i]->as_tensor(), solver_.use_mixed_precision,
        solver_.scaler, blobs_buff_list_[i], resource_manager_->get_local_gpu(i)));

    networks_[i]->train_weight_tensor_ = train_weight_buff_list_[i]->as_tensor();
    networks_[i]->train_weight_tensor_half_ = train_weight_buff_half_list_[i]->as_tensor();
    networks_[i]->wgrad_tensor_ = wgrad_buff_list_[i]->as_tensor();
    networks_[i]->wgrad_tensor_half_ = wgrad_buff_half_list_[i]->as_tensor();
    networks_[i]->evaluate_weight_tensor_ = evaluate_weight_buff_list_[i]->as_tensor();
    networks_[i]->evaluate_weight_tensor_half_ = evaluate_weight_buff_half_list_[i]->as_tensor();
    CudaDeviceContext context(resource_manager_->get_local_gpu(i)->get_device_id());
    blobs_buff_list_[i]->allocate();
  }
#ifndef DATA_READING_TEST
  for (auto& network : networks_) {
    network->initialize();
  }
  if (solver_.use_algorithm_search) {
    for (auto& network : networks_) {
      network->search_algorithm();
    }
  }
#endif

  init_or_load_params_for_dense_(solver_.model_file);
  if (solver_.use_model_oversubscriber) {
    if (solver_.use_mixed_precision) {
      model_oversubscriber_ = create_model_oversubscriber_<__half>();
    } else {
      model_oversubscriber_ = create_model_oversubscriber_<float>();
    }
  } else {
    init_or_load_params_for_sparse_(solver_.embedding_files);
  }

  int num_total_gpus = resource_manager_->get_global_gpu_count();
  for (const auto& metric : solver_.metrics_spec) {
    metrics_.emplace_back(
        std::move(metrics::Metric::Create(metric.first, solver_.use_mixed_precision,
                                          solver_.batchsize_eval / num_total_gpus,
                                          solver_.max_eval_batches, resource_manager_)));
  }
}

void Model::summary() {
  std::cout << "==================================Model Summary==================================" << std::endl;
  std::cout << "--------------------------------------------------------------------------------" << std::endl;
  std::cout << std::left << std::setw(30) << std::setfill(' ') << "Layer type"
       << std::left << std::setw(30) << std::setfill(' ') << "Input"
       << std::left << std::setw(30) << std::setfill(' ') << "Output"
       << std::endl;
  std::cout << "--------------------------------------------------------------------------------" << std::endl;
  for (size_t i = 0; i < layer_info_.size(); ++i) {
    std::cout << std::left << std::setw(30) << std::setfill(' ') << layer_info_[i]
        << std::left << std::setw(30) << std::setfill(' ') << input_output_info_[i].first
        << std::left << std::setw(30) << std::setfill(' ') << input_output_info_[i].second
        << std::endl;
  }
  std::cout << "--------------------------------------------------------------------------------" << std::endl;
}

void Model::fit() {
  HugeCTR::Timer timer_train;
  HugeCTR::Timer timer_eval;
  timer_train.start();
  bool epoch_mode = solver_.num_epochs > 0;
  std::cout << "=====================================Model Fit====================================" << std::endl;
  if (epoch_mode) {
    MESSAGE_("Use epoch mode with number of epochs: " + std::to_string(solver_.num_epochs));
  } else {
    MESSAGE_("Use non-epoch mode with number of iterations: " + std::to_string(solver_.max_iter));
  }
  MESSAGE_("Training batchsize: " + std::to_string(solver_.batchsize) 
         + ", evaluation batchsize: " + std::to_string(solver_.batchsize_eval));
  MESSAGE_("Evaluation interval: " + std::to_string(solver_.eval_interval) 
        + ", snapshot interval: " + std::to_string(solver_.snapshot));
  if (epoch_mode) {
    int iter = 0;
    auto data_reader_eval = this->get_evaluate_data_reader();
    data_reader_eval->set_source();
    for (int e = 0; e < solver_.num_epochs; e++) {
      std::cout << "--------------------Epoch " << e << "--------------------" << std::endl;
      bool train_reader_flag = false;
      auto data_reader_train = this->get_train_data_reader();
      data_reader_train->set_source();
      do {
        float lr = lr_sch_->get_next();
        this->set_learning_rate(lr);
        train_reader_flag = this->train();
        if (solver_.display > 0 && iter % solver_.display == 0 && iter != 0) {
          timer_train.stop();
          float loss = 0;
          this->get_current_loss(&loss);
          if (isnan(loss)) {
            throw std::runtime_error(std::string("Train Runtime error: Loss cannot converge") +
                                     " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n");
          }
          MESSAGE_("Iter: " + std::to_string(iter) + " Time(" +
                  std::to_string(solver_.display) +
                  " iters): " + std::to_string(timer_train.elapsedSeconds()) +
                  "s Loss: " + std::to_string(loss) + " lr:" + std::to_string(lr));
          timer_train.start();
        }
        if (solver_.eval_interval > 0 && iter % solver_.eval_interval == 0 && iter != 0) {
          this->check_overflow();
          this->copy_weights_for_evaluation();
          int batches = 0;
          bool eval_reader_flag = true;
          timer_eval.start();
          while (eval_reader_flag) {
            if (solver_.max_eval_batches == 0 || batches >= solver_.max_eval_batches) {
              break;
            }
            eval_reader_flag = this->eval();
            batches++;
          }
          if (eval_reader_flag == false) {
            data_reader_eval->set_source();
          }
          timer_eval.stop();
          auto eval_metrics = this->get_eval_metrics();
          for (auto& eval_metric : eval_metrics) {
            MESSAGE_("Evaluation, " + eval_metric.first + ": " + std::to_string(eval_metric.second));
          }
          MESSAGE_("Eval Time for " + std::to_string(solver_.max_eval_batches) +
            " iters: " + std::to_string(timer_eval.elapsedSeconds()) + "s");
        }
        if (solver_.snapshot > 0 && iter % solver_.snapshot == 0 && iter != 0) {
          this->download_params_to_files(solver_.snapshot_prefix, iter);
        }
        iter++;
      } while (train_reader_flag);
    } // end for epoch
  } else {
    this->start_data_reading();
    for (int iter = 0; iter < solver_.max_iter; iter++) {
      float lr = lr_sch_->get_next();
      this->set_learning_rate(lr);
      this->train();
      if (solver_.display > 0 && iter % solver_.display == 0 && iter != 0) {
        timer_train.stop();
        float loss = 0;
        this->get_current_loss(&loss);
        if (isnan(loss)) {
          throw std::runtime_error(std::string("Train Runtime error: Loss cannot converge") + " " +
                                   __FILE__ + ":" + std::to_string(__LINE__) + " \n");
        }
        MESSAGE_("Iter: " + std::to_string(iter) + " Time(" + std::to_string(solver_.display) +
                " iters): " + std::to_string(timer_train.elapsedSeconds()) +
                "s Loss: " + std::to_string(loss) + " lr:" + std::to_string(lr));
        timer_train.start();
      }
      if (solver_.eval_interval > 0 && iter % solver_.eval_interval == 0 && iter != 0) {
        this->check_overflow();
        this->copy_weights_for_evaluation();
        timer_eval.start();
        for (int batches = 0; batches < solver_.max_eval_batches; batches++) {
          this->eval();
        }
        timer_eval.stop();
        auto eval_metrics = this->get_eval_metrics();
        for (auto& eval_metric : eval_metrics) {
          MESSAGE_("Evaluation, " + eval_metric.first + ": " + std::to_string(eval_metric.second));
        }
        MESSAGE_("Eval Time for " + std::to_string(solver_.max_eval_batches) +
          " iters: " + std::to_string(timer_eval.elapsedSeconds()) + "s");
      }
      if (solver_.snapshot > 0 && iter % solver_.snapshot == 0 && iter != 0) {
        this->download_params_to_files(solver_.snapshot_prefix, iter);
      }
    } // end for iter
  } // end if else
}

bool Model::train() {
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
    train_data_reader_->read_a_batch_to_device();
#endif
  } catch (const internal_runtime_error& err) {
    std::cerr << err.what() << std::endl;
    throw err;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw err;
  }
}

bool Model::eval() {
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

std::vector<std::pair<std::string, float>> Model::get_eval_metrics() {
  std::vector<std::pair<std::string, float>> metrics;
  for (auto& metric : metrics_) {
    metrics.push_back(std::make_pair(metric->name(), metric->finalize_metric()));
  }
  return metrics;
}

Error_t Model::get_current_loss(float* loss) {
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

Error_t Model::download_params_to_files(std::string prefix, int iter) {
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

void Model::check_overflow() const {
  for (auto& one_embedding : embeddings_) {
    one_embedding->check_overflow();
  }
}

void Model::copy_weights_for_evaluation() {
  for (auto& network : networks_) {
    network->copy_weights_from_train_layers_to_evaluate_layers();
  }
}

Error_t Model::download_params_to_files_(std::string weights_file,
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

template <typename TypeEmbeddingComp>
std::shared_ptr<ModelOversubscriber> Model::create_model_oversubscriber_() {
  try {
    if (solver_.temp_embedding_dir.empty()) {
      CK_THROW_(Error_t::WrongInput, "must provide a directory for storing temporary embedding");
    }
    std::vector<SparseEmbeddingHashParams<TypeEmbeddingComp>> embedding_params;
    return std::shared_ptr<ModelOversubscriber>(
        new ModelOversubscriber(embeddings_, embedding_params, solver_, solver_.temp_embedding_dir));
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw rt_err;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw err;
  }
}

Error_t Model::init_or_load_params_for_dense_(const std::string& model_file) {
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

Error_t Model::init_or_load_params_for_sparse_(
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

} // namespace HugeCTR