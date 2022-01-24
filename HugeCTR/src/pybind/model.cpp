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

#include <HugeCTR/include/base/debug/logger.hpp>
#include <HugeCTR/include/resource_managers/resource_manager_ext.hpp>
#include <algorithm>
#include <data_readers/async_reader/async_reader_adapter.hpp>
#include <embeddings/hybrid_sparse_embedding.hpp>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <pybind/model.hpp>
#include <sstream>

namespace HugeCTR {

namespace {

/**
 * check if device is avaliable.
 * lowest avaliable CC is min_major.min_minor
 * @param device_id gpu id
 * @param min_major minimum compute compatibility required
 * @param min_minor minimum compute compatibility required
 */

static std::vector<std::string>& split(const std::string& s, char delim,
                                       std::vector<std::string>& elems) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

static std::string join(std::vector<std::string>& strs, std::string delim) {
  std::string str;
  const std::vector<std::string>::iterator itlast = strs.end() - 1;
  for (auto it = strs.begin(); it != strs.end(); it++) {
    str += *it;
    if (it != itlast) {
      str += delim;
    }
  }
  return str;
}

static std::string get_tensor_shape(std::string tensor_name,
                                    std::map<std::string, std::vector<size_t>> tensor_shape_info) {
  std::string shape = "";
  if (tensor_shape_info.find(tensor_name) != tensor_shape_info.end()) {
    shape += "(None";
    for (unsigned int i = 1; i < tensor_shape_info[tensor_name].size(); i++) {
      shape += ", ";
      shape += std::to_string(tensor_shape_info[tensor_name][i]);
    }
    shape += ")";
  }
  return shape;
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
  HCTR_LOG(INFO, WORLD, "Device %d: %s\n", device_id, deviceProp.name);
  int major = deviceProp.major;
  int minor = deviceProp.minor;
  if (major < min_major) {
    CK_THROW_(Error_t::InvalidEnv, "Device Compute Compacity is low");
  } else if (major == min_major && minor < min_minor) {
    CK_THROW_(Error_t::InvalidEnv, "Device Compute Compacity is low");
  }
  return;
}

template <typename TypeKey>
auto load_key_files(std::vector<std::string> const& key_files) {
  std::vector<TypeKey> keys_vec;
  for (auto const& key_file : key_files) {
    auto key_file_size = std::filesystem::file_size(key_file);
    auto num_new_keys = key_file_size / sizeof(TypeKey);
    std::ifstream key_fs(key_file, std::ifstream::binary);
    if (!key_fs.is_open()) {
      CK_THROW_(Error_t::WrongInput, "Cannot open the file: " + key_file);
    }
    auto num_exist_keys = keys_vec.size();
    keys_vec.resize(num_exist_keys + num_new_keys);
    key_fs.read(reinterpret_cast<char*>(&keys_vec[num_exist_keys]), key_file_size);
  }
  std::sort(keys_vec.begin(), keys_vec.end());
  keys_vec.erase(std::unique(keys_vec.begin(), keys_vec.end()), keys_vec.end());
  return keys_vec;
}

}  // end namespace

EmbeddingTrainingCacheParams::EmbeddingTrainingCacheParams(
    std::vector<TrainPSType_t>& _ps_types, std::vector<std::string>& _sparse_models,
    std::vector<std::string>& _local_paths, std::vector<HMemCacheConfig>& _hmem_cache_configs)
    : use_embedding_training_cache(true),
      ps_types(_ps_types),
      sparse_models(_sparse_models),
      local_paths(_local_paths),
      hmem_cache_configs(_hmem_cache_configs) {}

EmbeddingTrainingCacheParams::EmbeddingTrainingCacheParams()
    : use_embedding_training_cache(false) {}

DataReaderParams::DataReaderParams(DataReaderType_t data_reader_type,
                                   std::vector<std::string> source, std::vector<std::string> keyset,
                                   std::string eval_source, Check_t check_type, int cache_eval_data,
                                   long long num_samples, long long eval_num_samples,
                                   bool float_label_dense, int num_workers,
                                   std::vector<long long>& slot_size_array,
                                   const AsyncParam& async_param)
    : data_reader_type(data_reader_type),
      source(source),
      keyset(keyset),
      eval_source(eval_source),
      check_type(check_type),
      cache_eval_data(cache_eval_data),
      num_samples(num_samples),
      eval_num_samples(eval_num_samples),
      float_label_dense(float_label_dense),
      num_workers(num_workers),
      slot_size_array(slot_size_array),
      async_param(async_param) {}

DataReaderParams::DataReaderParams(DataReaderType_t data_reader_type, std::string source,
                                   std::string keyset, std::string eval_source, Check_t check_type,
                                   int cache_eval_data, long long num_samples,
                                   long long eval_num_samples, bool float_label_dense,
                                   int num_workers, std::vector<long long>& slot_size_array,
                                   const AsyncParam& async_param)
    : data_reader_type(data_reader_type),
      eval_source(eval_source),
      check_type(check_type),
      cache_eval_data(cache_eval_data),
      num_samples(num_samples),
      eval_num_samples(eval_num_samples),
      float_label_dense(float_label_dense),
      num_workers(num_workers),
      slot_size_array(slot_size_array),
      async_param(async_param) {
  this->source.push_back(source);
  this->keyset.push_back(keyset);
}

Input::Input(int label_dim, std::string label_name, int dense_dim, std::string dense_name,
             std::vector<DataReaderSparseParam>& data_reader_sparse_param_array)
    : label_dim(label_dim),
      label_name(label_name),
      dense_dim(dense_dim),
      dense_name(dense_name),
      data_reader_sparse_param_array(data_reader_sparse_param_array) {}

SparseEmbedding::SparseEmbedding(Embedding_t embedding_type, size_t workspace_size_per_gpu_in_mb,
                                 size_t embedding_vec_size, const std::string& combiner_str,
                                 std::string sparse_embedding_name, std::string bottom_name,
                                 std::vector<size_t>& slot_size_array,
                                 std::shared_ptr<OptParamsPy>& embedding_opt_params,
                                 const HybridEmbeddingParam& hybrid_embedding_param)
    : embedding_type(embedding_type),
      workspace_size_per_gpu_in_mb(workspace_size_per_gpu_in_mb),
      embedding_vec_size(embedding_vec_size),
      sparse_embedding_name(sparse_embedding_name),
      bottom_name(bottom_name),
      slot_size_array(slot_size_array),
      embedding_opt_params(embedding_opt_params),
      hybrid_embedding_param(hybrid_embedding_param) {
  if (combiner_str == "sum") {
    combiner = 0;
  } else if (combiner_str == "mean") {
    combiner = 1;
  } else {
    CK_THROW_(Error_t::WrongInput, "No such combiner type: " + combiner_str);
  }
  max_vocabulary_size_per_gpu =
      (workspace_size_per_gpu_in_mb * 1024 * 1024) / (sizeof(float) * embedding_vec_size);
}

DenseLayer::DenseLayer(Layer_t layer_type, std::vector<std::string>& bottom_names,
                       std::vector<std::string>& top_names, float factor, float eps,
                       Initializer_t gamma_init_type, Initializer_t beta_init_type,
                       float dropout_rate, float elu_alpha, size_t num_output,
                       Initializer_t weight_init_type, Initializer_t bias_init_type, int num_layers,
                       size_t leading_dim, size_t time_step, size_t batchsize, size_t SeqLength,
                       size_t vector_size, bool selected, std::vector<int> selected_slots,
                       std::vector<std::pair<int, int>> ranges, std::vector<int> indices,
                       std::vector<size_t> weight_dims, size_t out_dim, int axis,
                       std::vector<float> target_weight_vec, bool use_regularizer,
                       Regularizer_t regularizer_type, float lambda, FcPosition_t pos_type,
                       Activation_t act_type)
    : layer_type(layer_type),
      bottom_names(bottom_names),
      top_names(top_names),
      factor(factor),
      eps(eps),
      gamma_init_type(gamma_init_type),
      beta_init_type(beta_init_type),
      dropout_rate(dropout_rate),
      elu_alpha(elu_alpha),
      num_output(num_output),
      weight_init_type(weight_init_type),
      bias_init_type(bias_init_type),
      num_layers(num_layers),
      leading_dim(leading_dim),
      time_step(time_step),
      batchsize(batchsize),
      SeqLength(SeqLength),
      vector_size(vector_size),
      selected(selected),
      selected_slots(selected_slots),
      ranges(ranges),
      indices(indices),
      weight_dims(weight_dims),
      out_dim(out_dim),
      axis(axis),
      target_weight_vec(target_weight_vec),
      use_regularizer(use_regularizer),
      regularizer_type(regularizer_type),
      lambda(lambda),
      pos_type(pos_type),
      act_type(act_type) {}

GroupDenseLayer::GroupDenseLayer(GroupLayer_t group_layer_type,
                                 std::vector<std::string>& bottom_name_list,
                                 std::vector<std::string>& top_name_list,
                                 std::vector<size_t>& num_outputs, Activation_t last_act_type)
    : group_layer_type(group_layer_type),
      bottom_name_list(bottom_name_list),
      top_name_list(top_name_list),
      num_outputs(num_outputs),
      last_act_type(last_act_type) {
  size_t num_layers = num_outputs.size();
  if (num_layers < 2) {
    CK_THROW_(Error_t::WrongInput, "There should be at least two layers for GroupDenseLayer");
  }
  if (top_name_list.size() != num_layers) {
    CK_THROW_(Error_t::WrongInput,
              "There top_name_list.size() should be equal to num_outputs.size()");
  }
}

void init_optimizer(OptParams& opt_params, const Solver& solver,
                    const std::shared_ptr<OptParamsPy>& opt_params_py) {
  opt_params.optimizer = opt_params_py->optimizer;
  opt_params.lr = solver.lr;
  opt_params.update_type = opt_params_py->update_type;
  opt_params.scaler = solver.scaler;
  opt_params.hyperparams.adam.beta1 = opt_params_py->hyperparams.adam.beta1;
  opt_params.hyperparams.adam.beta2 = opt_params_py->hyperparams.adam.beta2;
  opt_params.hyperparams.adam.epsilon = opt_params_py->hyperparams.adam.epsilon;
  opt_params.hyperparams.adagrad.initial_accu_value =
      opt_params_py->hyperparams.adagrad.initial_accu_value;
  opt_params.hyperparams.adagrad.epsilon = opt_params_py->hyperparams.adagrad.epsilon;
  opt_params.hyperparams.momentum.factor = opt_params_py->hyperparams.momentum.factor;
  opt_params.hyperparams.nesterov.mu = opt_params_py->hyperparams.nesterov.mu;
  opt_params.hyperparams.sgd.atomic_update = opt_params_py->hyperparams.sgd.atomic_update;
}

void init_learning_rate_scheduler(std::shared_ptr<LearningRateScheduler>& lr_sch,
                                  const Solver& solver, GpuLearningRateSchedulers& gpu_lr_sches,
                                  const std::shared_ptr<ResourceManager>& resource_manager) {
  lr_sch.reset(new LearningRateScheduler(solver.lr, solver.warmup_steps, solver.decay_start,
                                         solver.decay_steps, solver.decay_power, solver.end_lr));
  for (size_t i = 0; i < resource_manager->get_local_gpu_count(); i++) {
    auto& gpu_resource = resource_manager->get_local_gpu(i);
    gpu_lr_sches.emplace_back(new GpuLearningRateScheduler(
        solver.lr, solver.warmup_steps, solver.decay_start, solver.decay_steps, solver.decay_power,
        solver.end_lr, gpu_resource));
  }
}

void init_exchange_wgrad(const std::shared_ptr<ResourceManager>& resource_manager,
                         std::shared_ptr<ExchangeWgrad>& exchange_wgrad, const Solver& solver) {
  HCTR_LOG(INFO, ROOT, "Using All-reduce algorithm: %s\n",
           ALLREDUCE_ALGO_TO_STRING[solver.all_reduce_algo].c_str());
  resource_manager->set_ar_comm(solver.all_reduce_algo, solver.use_mixed_precision);
  if (solver.grouped_all_reduce) {
    if (solver.use_mixed_precision) {
      exchange_wgrad = std::make_shared<GroupedExchangeWgrad<__half>>(resource_manager);
    } else {
      exchange_wgrad = std::make_shared<GroupedExchangeWgrad<float>>(resource_manager);
    }
  } else {
    if (solver.use_mixed_precision) {
      exchange_wgrad = std::make_shared<NetworkExchangeWgrad<__half>>(resource_manager);
    } else {
      exchange_wgrad = std::make_shared<NetworkExchangeWgrad<float>>(resource_manager);
    }
  }
}

Model::Model(const Solver& solver, const DataReaderParams& reader_params,
             std::shared_ptr<OptParamsPy>& opt_params_py,
             std::shared_ptr<EmbeddingTrainingCacheParams>& etc_params)
    : solver_(solver),
      reader_params_(reader_params),
      opt_params_py_(opt_params_py),
      etc_params_(etc_params),
      data_reader_train_status_(false),
      data_reader_eval_status_(false),
      buff_allocated_(false),
      etc_created_(false),
      is_dense_trainable_(true),
      current_eval_batchsize_(0),
      dlrm_bottom_mlp_(true),
      high_level_eval_(false) {
#ifdef ENABLE_MPI
  HCTR_MPI_THROW(MPI_Initialized(&mpi_preinitialized_));
  if (!mpi_preinitialized_) {
    HCTR_LOG(WARNING, ROOT, "Please add \"from mpi4py import MPI\" in your Python script\n");
    HCTR_MPI_THROW(MPI_Init(nullptr, nullptr));
    HCTR_LOG(INFO, ROOT, "MPI initialization is done\n");
  }
#endif
  if (solver_.is_dlrm) {
    timer_log.start();
    LOG(timer_log.elapsedMilliseconds(), "init_start");
  }
  HCTR_PRINT(INFO, "HugeCTR Version: %d.%d\n", HUGECTR_VERSION_MAJOR, HUGECTR_VERSION_MINOR);
  HCTR_PRINT(INFO,
             "====================================================Model "
             "Init=====================================================\n");
  if (!solver_.model_name.length()) {
    HCTR_LOG(WARNING, ROOT, "The model name is not specified when creating the solver.\n");
  } else {
    HCTR_LOG(INFO, ROOT, "Initialize model: %s\n", solver_.model_name.c_str());
  }
  resource_manager_ = ResourceManagerExt::create(solver.vvgpu, solver.seed, solver.device_layout);

  init_exchange_wgrad(resource_manager_, exchange_wgrad_, solver_);

  graph_scheduler_ = std::make_unique<GraphScheduler>(resource_manager_);
  for (auto dev : resource_manager_->get_local_gpu_device_id_list()) {
    if (solver_.use_mixed_precision) {
      check_device(dev, 7,
                   0);  // to support mixed precision training earliest supported device is CC=70
    } else {
      check_device(dev, 6, 0);  // earliest supported device is CC=60
    }
  }
  if (reader_params_.source.size() < 1 || reader_params_.eval_source.empty()) {
    CK_THROW_(Error_t::WrongInput,
              " The data source for training and evaluation should be specified");
  }
  if (etc_params_->use_embedding_training_cache && solver_.kafka_brokers.length()) {
    message_sink_.reset(new KafkaMessageSink<long long>(solver_.kafka_brokers));
  }
  if (etc_params_->use_embedding_training_cache && solver_.repeat_dataset) {
    CK_THROW_(Error_t::WrongInput,
              "The embedding training cache can only be used under epoch mode, "
              "i.e., repeat_dataset is set False");
  }
  if (etc_params_->use_embedding_training_cache &&
      reader_params_.keyset.size() != reader_params_.source.size()) {
    CK_THROW_(Error_t::WrongInput,
              "The number of keyset files must equal that of training data source when using "
              "embedding training cache");
  }
  int total_gpu_count = resource_manager_->get_global_gpu_count();
  if (0 != solver_.batchsize % total_gpu_count) {
    CK_THROW_(Error_t::WrongInput, "0 != batch_size\%total_gpu_count");
  }
  // reserve networks to be created
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    networks_.emplace_back(new Network(resource_manager_->get_local_cpu(),
                                       resource_manager_->get_local_gpu(i),
                                       solver_.use_mixed_precision, solver_.use_cuda_graph));
    blobs_buff_list_.emplace_back(GeneralBuffer2<CudaAllocator>::create());
  }

  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    train_weight_buff_list_.emplace_back(blobs_buff_list_[i]->create_block<float>());
    train_weight_buff_half_list_.emplace_back(blobs_buff_list_[i]->create_block<__half>());
    evaluate_weight_buff_list_.emplace_back(blobs_buff_list_[i]->create_block<float>());
    evaluate_weight_buff_half_list_.emplace_back(blobs_buff_list_[i]->create_block<__half>());
    wgrad_buff_placeholder_list_.emplace_back(blobs_buff_list_[i]->create_block<float>());
    wgrad_buff_half_placeholder_list_.emplace_back(blobs_buff_list_[i]->create_block<__half>());
    opt_buff_list_.emplace_back(blobs_buff_list_[i]->create_block<float>());
    opt_buff_half_list_.emplace_back(blobs_buff_list_[i]->create_block<__half>());
    auto id = resource_manager_->get_local_gpu(i)->get_local_id();
    if (solver_.use_mixed_precision) {
      wgrad_buff_half_list_.emplace_back(
          (solver_.grouped_all_reduce)
              ? std::dynamic_pointer_cast<GroupedExchangeWgrad<__half>>(exchange_wgrad_)
                    ->get_network_wgrad_buffs()[id]
              : std::dynamic_pointer_cast<NetworkExchangeWgrad<__half>>(exchange_wgrad_)
                    ->get_network_wgrad_buffs()[id]);
      wgrad_buff_list_.emplace_back(blobs_buff_list_[i]->create_block<float>());
    } else {
      wgrad_buff_list_.emplace_back(
          (solver_.grouped_all_reduce)
              ? std::dynamic_pointer_cast<GroupedExchangeWgrad<float>>(exchange_wgrad_)
                    ->get_network_wgrad_buffs()[id]
              : std::dynamic_pointer_cast<NetworkExchangeWgrad<float>>(exchange_wgrad_)
                    ->get_network_wgrad_buffs()[id]);
      wgrad_buff_half_list_.emplace_back(
          blobs_buff_list_[i]->create_block<__half>());  // placeholder
    }
  }

  // resize train_tensor_entries_list_ and evaluate_tensor_entries_list_
  train_tensor_entries_list_.resize(resource_manager_->get_local_gpu_count());
  evaluate_tensor_entries_list_.resize(resource_manager_->get_local_gpu_count());

  // initialize optimizer
  init_optimizer(opt_params_, solver_, opt_params_py);
  init_learning_rate_scheduler(lr_sch_, solver_, gpu_lr_sches_, resource_manager_);
}

Model::~Model() {
  for (auto device : resource_manager_->get_local_gpu_device_id_list()) {
    CudaDeviceContext context(device);
    cudaDeviceSynchronize();
  }
#ifdef ENABLE_MPI
  if (!mpi_preinitialized_) {
    HCTR_MPI_THROW(MPI_Finalize());
    HCTR_LOG(INFO, ROOT, "The MPI finalization is done\n");
  }
#endif
}

void Model::graph_to_json(std::string graph_config_file) {
  if (!graph_finalized_) {
    graph_analysis();
    graph_finalized_ = true;
  }
  nlohmann::json graph_config;
  std::ofstream file_stream(graph_config_file);
  nlohmann::json layer_config_array = nlohmann::json::array();
  save_graph_to_json(layer_config_array, dense_layer_params_, sparse_embedding_params_,
                     input_params_, embedding_opt_params_list_, solver_.use_mixed_precision);
  graph_config["layers"] = layer_config_array;
  // The model_name in dumped JSON will only be used for inference currently
  if (solver_.model_name.length()) {
    graph_config["model_name"] = solver_.model_name;
  }
  file_stream << std::setw(2) << graph_config;
  file_stream.close();
  HCTR_LOG(INFO, ROOT, "Save the model graph to %s successfully\n", graph_config_file.c_str());
}

void Model::construct_from_json(const std::string& graph_config_file, bool include_dense_network) {
  nlohmann::json graph_config = read_json_file(graph_config_file);
  auto j_layers_array = get_json(graph_config, "layers");
  const nlohmann::json& j_input = j_layers_array[0];
  Input input = get_input_from_json(j_input);
  add(input);

  unsigned int dense_layer_start_index = 1;
  for (unsigned int i = 1; i < j_layers_array.size(); i++) {
    const nlohmann::json& j = j_layers_array[i];
    Embedding_t embedding_type;
    auto embedding_type_name = get_value_from_json<std::string>(j, "type");
    if (!find_item_in_map(embedding_type, embedding_type_name, EMBEDDING_TYPE_MAP)) {
      dense_layer_start_index = i;
      break;
    }
    SparseEmbedding sparse_embedding = get_sparse_embedding_from_json(j);
    add(sparse_embedding);
  }

  if (include_dense_network) {
    for (unsigned int i = dense_layer_start_index; i < j_layers_array.size(); i++) {
      const nlohmann::json& j = j_layers_array[i];
      Layer_t layer_type;
      auto layer_type_name = get_value_from_json<std::string>(j, "type");
      if (!find_item_in_map(layer_type, layer_type_name, LAYER_TYPE_MAP) &&
          !find_item_in_map(layer_type, layer_type_name, LAYER_TYPE_MAP_MP)) {
        CK_THROW_(Error_t::WrongInput, "No such layer: " + layer_type_name);
      }
      DenseLayer dense_layer = get_dense_layer_from_json(j);
      add(dense_layer);
    }
  }

  HCTR_LOG(INFO, ROOT, "Load the model graph from %s successfully\n", graph_config_file.c_str());
}

void Model::add(Input& input) {
  if (!solver_.is_dlrm && reader_params_.data_reader_type == DataReaderType_t::RawAsync) {
    CK_THROW_(Error_t::WrongInput, "Raw async reader is restricted to DLRM use");
  }
  input_params_.push_back(input);
  activate_tensor(tensor_active_, input.label_name);
  activate_tensor(tensor_active_, input.dense_name);
  data_input_info_.push_back(input.label_name);
  data_input_info_.push_back(input.dense_name);
  tensor_shape_info_raw_.insert(
      std::make_pair(input.label_name, std::vector<int>{solver_.batchsize, input.label_dim}));
  tensor_shape_info_raw_.insert(
      std::make_pair(input.dense_name, std::vector<int>{solver_.batchsize, input.dense_dim}));
  std::vector<std::string> sparse_names;
  for (size_t i = 0; i < input.data_reader_sparse_param_array.size(); ++i) {
    sparse_names.push_back(input.data_reader_sparse_param_array[i].top_name);
    tensor_shape_info_raw_.insert(std::make_pair(
        input.data_reader_sparse_param_array[i].top_name,
        std::vector<int>{solver_.batchsize, input.data_reader_sparse_param_array[i].slot_num}));
  }
  data_input_info_.push_back(join(sparse_names, ","));
  for (unsigned int i = 0; i < input.data_reader_sparse_param_array.size(); i++) {
    activate_tensor(tensor_active_, input.data_reader_sparse_param_array[i].top_name);
  }
  if (solver_.i64_input_key) {
    add_input<long long>(input, reader_params_, sparse_input_map_64_, train_tensor_entries_list_,
                         evaluate_tensor_entries_list_, train_data_reader_, evaluate_data_reader_,
                         init_data_reader_, solver_.batchsize, solver_.batchsize_eval,
                         solver_.use_mixed_precision, solver_.repeat_dataset,
                         solver_.use_overlapped_pipeline, solver_.num_iterations_statistics,
                         resource_manager_);
  } else {
    add_input<unsigned int>(input, reader_params_, sparse_input_map_32_, train_tensor_entries_list_,
                            evaluate_tensor_entries_list_, train_data_reader_,
                            evaluate_data_reader_, init_data_reader_, solver_.batchsize,
                            solver_.batchsize_eval, solver_.use_mixed_precision,
                            solver_.repeat_dataset, solver_.use_overlapped_pipeline,
                            solver_.num_iterations_statistics, resource_manager_);
  }
}

void Model::add(SparseEmbedding& sparse_embedding) {
  if (!solver_.is_dlrm && sparse_embedding.embedding_type == Embedding_t::HybridSparseEmbedding) {
    CK_THROW_(Error_t::WrongInput, "Hybrid embedding is restricted to DLRM use");
  }
  if ((reader_params_.data_reader_type == DataReaderType_t::RawAsync &&
       sparse_embedding.embedding_type != Embedding_t::HybridSparseEmbedding) ||
      (reader_params_.data_reader_type != DataReaderType_t::RawAsync &&
       sparse_embedding.embedding_type == Embedding_t::HybridSparseEmbedding)) {
    CK_THROW_(Error_t::WrongInput, "Raw async reader and hybrid embedding must come together");
  }
  sparse_embedding.max_vocabulary_size_global =
      sparse_embedding.max_vocabulary_size_per_gpu * resource_manager_->get_global_gpu_count();
  sparse_embedding_params_.push_back(sparse_embedding);
  deactivate_tensor(tensor_active_, sparse_embedding.bottom_name);
  activate_tensor(tensor_active_, sparse_embedding.sparse_embedding_name);
  int slot_num = tensor_shape_info_raw_[sparse_embedding.bottom_name][1];
  tensor_shape_info_raw_.insert(
      std::make_pair(sparse_embedding.sparse_embedding_name,
                     std::vector<int>{solver_.batchsize, slot_num,
                                      static_cast<int>(sparse_embedding.embedding_vec_size)}));
  input_output_info_.push_back(
      std::make_pair(sparse_embedding.bottom_name, sparse_embedding.sparse_embedding_name));
  layer_info_.push_back(EMBEDDING_TYPE_TO_STRING[sparse_embedding.embedding_type]);
  OptParams embedding_opt_params;
  if (!(sparse_embedding.embedding_opt_params)->initialized) {
    sparse_embedding.embedding_opt_params = opt_params_py_;
  }
  embedding_opt_params_list_.push_back(sparse_embedding.embedding_opt_params);
  init_optimizer(embedding_opt_params, solver_, sparse_embedding.embedding_opt_params);
  if (solver_.i64_input_key && !solver_.use_mixed_precision) {
    add_sparse_embedding<long long, float>(
        sparse_embedding, sparse_input_map_64_, train_tensor_entries_list_,
        evaluate_tensor_entries_list_, embeddings_, resource_manager_, solver_.batchsize,
        solver_.batchsize_eval, embedding_opt_params, exchange_wgrad_, solver_.use_cuda_graph,
        solver_.grouped_all_reduce, solver_.use_holistic_cuda_graph,
        solver_.num_iterations_statistics, gpu_lr_sches_, solver_.overlap_ar_a2a);
  } else if (solver_.i64_input_key && solver_.use_mixed_precision) {
    add_sparse_embedding<long long, __half>(
        sparse_embedding, sparse_input_map_64_, train_tensor_entries_list_,
        evaluate_tensor_entries_list_, embeddings_, resource_manager_, solver_.batchsize,
        solver_.batchsize_eval, embedding_opt_params, exchange_wgrad_, solver_.use_cuda_graph,
        solver_.grouped_all_reduce, solver_.use_holistic_cuda_graph,
        solver_.num_iterations_statistics, gpu_lr_sches_, solver_.overlap_ar_a2a);
  } else if (!solver_.i64_input_key && !solver_.use_mixed_precision) {
    add_sparse_embedding<unsigned int, float>(
        sparse_embedding, sparse_input_map_32_, train_tensor_entries_list_,
        evaluate_tensor_entries_list_, embeddings_, resource_manager_, solver_.batchsize,
        solver_.batchsize_eval, embedding_opt_params, exchange_wgrad_, solver_.use_cuda_graph,
        solver_.grouped_all_reduce, solver_.use_holistic_cuda_graph,
        solver_.num_iterations_statistics, gpu_lr_sches_, solver_.overlap_ar_a2a);
  } else {
    add_sparse_embedding<unsigned int, __half>(
        sparse_embedding, sparse_input_map_32_, train_tensor_entries_list_,
        evaluate_tensor_entries_list_, embeddings_, resource_manager_, solver_.batchsize,
        solver_.batchsize_eval, embedding_opt_params, exchange_wgrad_, solver_.use_cuda_graph,
        solver_.grouped_all_reduce, solver_.use_holistic_cuda_graph,
        solver_.num_iterations_statistics, gpu_lr_sches_, solver_.overlap_ar_a2a);
  }
  embeddings_map_.insert(
      std::make_pair(sparse_embedding.sparse_embedding_name, embeddings_.back()));
}

void Model::add(DenseLayer& dense_layer) {
  for (auto& top_name : dense_layer.top_names) {
    if (tensor_shape_info_raw_.find(top_name) != tensor_shape_info_raw_.end()) {
      CK_THROW_(Error_t::WrongInput, top_name + ", top tensor name already exists");
    }
  }
  for (auto& bottom_name : dense_layer.bottom_names) {
    if (tensor_shape_info_raw_.find(bottom_name) == tensor_shape_info_raw_.end()) {
      CK_THROW_(Error_t::WrongInput, bottom_name + ", bottom tensor name does not exists");
    }
  }
  calculate_tensor_dimensions(tensor_shape_info_raw_, dense_layer);
  dense_layer_params_raw_.push_back(dense_layer);
}

void Model::add_internal(DenseLayer& dense_layer) {
  for (auto& bottom_name : dense_layer.bottom_names) {
    deactivate_tensor(tensor_active_, bottom_name);
  }
  for (auto& top_name : dense_layer.top_names) {
    activate_tensor(tensor_active_, top_name);
  }
  std::string input_names = join(dense_layer.bottom_names, ",");
  std::string output_names = join(dense_layer.top_names, ",");
  input_output_info_.push_back(std::make_pair(input_names, output_names));
  if (solver_.use_mixed_precision) {
    layer_info_.push_back(LAYER_TYPE_TO_STRING_MP[dense_layer.layer_type]);
  } else {
    layer_info_.push_back(LAYER_TYPE_TO_STRING[dense_layer.layer_type]);
  }
  if (dense_layer.layer_type == Layer_t::Interaction) {
    dlrm_bottom_mlp_ = false;
  }
  add_dense_layer(dense_layer);
}

void Model::add(GroupDenseLayer& group_dense_layer) {
  switch (group_dense_layer.group_layer_type) {
    case GroupLayer_t::GroupFusedInnerProduct: {
      size_t num_layers = group_dense_layer.num_outputs.size();
      std::vector<std::vector<std::string>> tensor_names_list(num_layers + 1,
                                                              std::vector<std::string>());
      for (auto& bottom_name : group_dense_layer.bottom_name_list) {
        tensor_names_list[0].push_back(bottom_name);
      }
      for (unsigned int i = 1; i < num_layers; i++) {
        std::string tensor_name = group_dense_layer.top_name_list[i - 1];
        tensor_names_list[i].push_back(tensor_name);
        tensor_names_list[i].push_back(tensor_name + "_mask");
        tensor_names_list[i].push_back(tensor_name + "_dRelu");
        tensor_names_list[i].push_back(tensor_name + "_db");
      }
      std::string top_name = group_dense_layer.top_name_list[num_layers - 1];
      tensor_names_list[num_layers].push_back(top_name);
      if (solver_.use_mixed_precision) {
        // leverage FusedReluBiasFullyConnectedLayer
        DenseLayer fused_fc_head_layer =
            DenseLayer(Layer_t::FusedInnerProduct, tensor_names_list[0], tensor_names_list[1]);
        fused_fc_head_layer.num_output = group_dense_layer.num_outputs[0];
        fused_fc_head_layer.pos_type = FcPosition_t::Head;
        fused_fc_head_layer.act_type = Activation_t::Relu;
        add(fused_fc_head_layer);
        for (unsigned int i = 1; i < num_layers - 1; i++) {
          DenseLayer fused_fc_body_layer = DenseLayer(
              Layer_t::FusedInnerProduct, tensor_names_list[i], tensor_names_list[i + 1]);
          fused_fc_body_layer.num_output = group_dense_layer.num_outputs[i];
          fused_fc_body_layer.pos_type = FcPosition_t::Body;
          fused_fc_body_layer.act_type = Activation_t::Relu;
          add(fused_fc_body_layer);
        }
        DenseLayer fused_fc_tail_layer =
            DenseLayer(Layer_t::FusedInnerProduct, tensor_names_list[num_layers - 1],
                       tensor_names_list[num_layers]);
        fused_fc_tail_layer.num_output = group_dense_layer.num_outputs[num_layers - 1];
        fused_fc_tail_layer.pos_type = FcPosition_t::Tail;
        fused_fc_tail_layer.act_type = group_dense_layer.last_act_type;
        add(fused_fc_tail_layer);
        // TODO: support fp32 for FusedReluBiasFullyConnectedLayer
      } else {
        CK_THROW_(Error_t::WrongInput,
                  "GroupFusedInnerProduct can only be used when use_mixed_precision is set true");
      }
      break;
    }
    default: {
      assert(!"Error: no such group layer && should never get here!");
    }
  }
}

void Model::graph_analysis() {
  HCTR_LOG(INFO, ROOT, "Graph analysis to resolve tensor dependency\n");
  std::map<std::string, unsigned int> tensor_usage;
  std::map<std::string, DenseLayer> tensor_slice_layer;
  std::map<std::string, unsigned int> tensor_slice_index;
  for (auto& dense_layer : dense_layer_params_raw_) {
    for (auto& bottom_name : dense_layer.bottom_names) {
      analyze_tensor(tensor_usage, bottom_name);
    }
  }
  for (auto iter = tensor_usage.begin(); iter != tensor_usage.end(); iter++) {
    if (iter->second > 5) {
      CK_THROW_(Error_t::WrongInput, "The graph should not include more than 5-way branches");
    }
    if (iter->second > 1) {
      std::vector<std::string> bottom_names{iter->first};
      std::vector<std::string> top_names;
      std::vector<std::pair<int, int>> ranges;
      for (unsigned int i = 0; i < iter->second; i++) {
        top_names.push_back(iter->first + "_slice" + std::to_string(i));
        ranges.emplace_back(std::make_pair(0, tensor_shape_info_raw_[iter->first][1]));
      }
      DenseLayer slice_layer(Layer_t::Slice, bottom_names, top_names);
      slice_layer.ranges = ranges;
      tensor_slice_layer.insert(std::pair<std::string, DenseLayer>(iter->first, slice_layer));
      tensor_slice_index.insert(std::pair<std::string, unsigned int>(iter->first, 0));
      HCTR_LOG(INFO, ROOT, "Add Slice layer for tensor: %s, creating %d copies\n",
               iter->first.c_str(), iter->second);
    }
  }
  for (auto& dense_layer : dense_layer_params_raw_) {
    bool flag = true;
    for (auto& bottom_name : dense_layer.bottom_names) {
      if (tensor_usage[bottom_name] > 1) {
        flag = false;
        break;
      }
    }
    if (flag) {
      dense_layer_params_.push_back(dense_layer);
    } else {
      DenseLayer new_dense_layer = dense_layer;
      for (unsigned int i = 0; i < new_dense_layer.bottom_names.size(); i++) {
        std::string old_bottom_name = new_dense_layer.bottom_names[i];
        if (tensor_slice_index.find(old_bottom_name) != tensor_slice_index.end()) {
          auto iter = tensor_slice_layer.find(old_bottom_name);
          if (tensor_slice_index[old_bottom_name] == 0) {
            dense_layer_params_.push_back(iter->second);
          }
          std::string new_bottom_name = iter->second.top_names[tensor_slice_index[old_bottom_name]];
          tensor_slice_index[old_bottom_name] += 1;
          new_dense_layer.bottom_names[i] = new_bottom_name;
        }
      }
      dense_layer_params_.push_back(new_dense_layer);
    }
  }
  for (auto& dense_layer : dense_layer_params_) {
    add_internal(dense_layer);
  }
}

void Model::compile() {
  if (!graph_finalized_) {
    graph_analysis();
    graph_finalized_ = true;
  }
  if (data_input_info_.size() < 3 || layer_info_.size() < 2) {
    CK_THROW_(Error_t::IllegalCall, "The model should include input and at least two layers");
  }
  HCTR_PRINT(INFO,
             "===================================================Model "
             "Compile===================================================\n");
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    if (solver_.use_mixed_precision) {
      networks_[i]->optimizer_ = std::move(Optimizer::Create(
          opt_params_, train_weight_buff_list_[i]->as_tensor(),
          train_weight_buff_half_list_[i]->as_tensor(), wgrad_buff_half_list_[i]->as_tensor(),
          solver_.scaler, opt_buff_half_list_[i], resource_manager_->get_local_gpu(i),
          solver_.use_mixed_precision));
    } else {
      networks_[i]->optimizer_ = std::move(
          Optimizer::Create(opt_params_, train_weight_buff_list_[i]->as_tensor(),
                            train_weight_buff_half_list_[i]->as_tensor(),
                            wgrad_buff_list_[i]->as_tensor(), solver_.scaler, opt_buff_list_[i],
                            resource_manager_->get_local_gpu(i), solver_.use_mixed_precision));
    }
    if (solver_.overlap_lr) networks_[i]->optimizer_->set_skip_lr_update();

    networks_[i]->train_weight_tensor_ = train_weight_buff_list_[i]->as_tensor();
    networks_[i]->train_weight_tensor_half_ = train_weight_buff_half_list_[i]->as_tensor();
    networks_[i]->wgrad_tensor_ = wgrad_buff_list_[i]->as_tensor();
    networks_[i]->wgrad_tensor_half_ = wgrad_buff_half_list_[i]->as_tensor();
    networks_[i]->evaluate_weight_tensor_ = evaluate_weight_buff_list_[i]->as_tensor();
    networks_[i]->evaluate_weight_tensor_half_ = evaluate_weight_buff_half_list_[i]->as_tensor();
    networks_[i]->opt_tensor_ = opt_buff_list_[i]->as_tensor();
    networks_[i]->opt_tensor_half_ = opt_buff_half_list_[i]->as_tensor();
    CudaDeviceContext context(resource_manager_->get_local_gpu(i)->get_device_id());
    blobs_buff_list_[i]->allocate();
  }
  exchange_wgrad_->allocate();
  buff_allocated_ = true;
#ifndef DATA_READING_TEST
#pragma omp parallel num_threads(networks_.size())
  {
    size_t id = omp_get_thread_num();
    networks_[id]->initialize();
    if (solver_.use_algorithm_search) {
      networks_[id]->search_algorithm();
    }
    CK_CUDA_THROW_(cudaStreamSynchronize(resource_manager_->get_local_gpu(id)->get_stream()));
  }
#endif
  init_params_for_dense_();
  init_params_for_sparse_();
  if (etc_params_->use_embedding_training_cache) {
    init_embedding_training_cache_(etc_params_->ps_types, etc_params_->sparse_models,
                                   etc_params_->local_paths, etc_params_->hmem_cache_configs);
  }
  int num_total_gpus = resource_manager_->get_global_gpu_count();
  for (const auto& metric : solver_.metrics_spec) {
    metrics_.emplace_back(std::move(metrics::Metric::Create(
        metric.first, solver_.use_mixed_precision, solver_.batchsize_eval / num_total_gpus,
        solver_.max_eval_batches, resource_manager_)));
  }

  if (solver_.use_holistic_cuda_graph) {
    train_graphs_.resize(networks_.size());
    for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
      auto& gpu_resource = resource_manager_->get_local_gpu(i);
      CudaCPUDeviceContext context(gpu_resource->get_device_id());
      // CudaDeviceContext context(gpu_resource->get_device_id());
      cudaEvent_t event;
      CK_CUDA_THROW_(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      fork_events_.push_back(event);
    }
  }

  // TODO: currently it is only for HE
  if (embeddings_.size() == 1) {
    auto lr_scheds = embeddings_[0]->get_learning_rate_schedulers();
    for (size_t i = 0; i < lr_scheds.size(); i++) {
      if (solver_.overlap_lr) {
        lr_scheds[i]->set_overlapped();
      }
      networks_[i]->set_learning_rate_scheduler(lr_scheds[i]);
    }
  }

  auto train_data_reader_ar_i64 = dynamic_cast<AsyncReader<long long>*>(train_data_reader_.get());
  auto eval_data_reader_ar_i64 = dynamic_cast<AsyncReader<long long>*>(evaluate_data_reader_.get());
  auto init_data_reader_ar_i64 = dynamic_cast<AsyncReader<long long>*>(init_data_reader_.get());

  auto train_data_reader_ar_i32 =
      dynamic_cast<AsyncReader<unsigned int>*>(train_data_reader_.get());
  auto eval_data_reader_ar_i32 =
      dynamic_cast<AsyncReader<unsigned int>*>(evaluate_data_reader_.get());
  auto init_data_reader_ar_i32 = dynamic_cast<AsyncReader<unsigned int>*>(init_data_reader_.get());

  // If doing async indices, init them before touching the data
  for (size_t i = 0; i < sparse_embedding_params_.size(); i++) {
    if (sparse_embedding_params_[i].embedding_type == Embedding_t::HybridSparseEmbedding) {
      if (solver_.use_mixed_precision && solver_.i64_input_key) {
        auto hybrid_embedding =
            dynamic_cast<HybridSparseEmbedding<long long, __half>*>(embeddings_[i].get());
        hybrid_embedding->setup_async_mode(train_data_reader_ar_i64, eval_data_reader_ar_i64);
      } else if (solver_.use_mixed_precision && !solver_.i64_input_key) {
        auto hybrid_embedding =
            dynamic_cast<HybridSparseEmbedding<unsigned int, __half>*>(embeddings_[i].get());
        hybrid_embedding->setup_async_mode(train_data_reader_ar_i32, eval_data_reader_ar_i32);
      } else if (!solver_.use_mixed_precision && solver_.i64_input_key) {
        auto hybrid_embedding =
            dynamic_cast<HybridSparseEmbedding<long long, float>*>(embeddings_[i].get());
        hybrid_embedding->setup_async_mode(train_data_reader_ar_i64, eval_data_reader_ar_i64);
      } else {
        auto hybrid_embedding =
            dynamic_cast<HybridSparseEmbedding<unsigned int, float>*>(embeddings_[i].get());
        hybrid_embedding->setup_async_mode(train_data_reader_ar_i32, eval_data_reader_ar_i32);
      }
    }
  }

  // start to touch dataset, so we can record run_start
  if (solver_.is_dlrm) {
    LOG(timer_log.elapsedMilliseconds(), "init_end");
    LOG(timer_log.elapsedMilliseconds(), "run_start");
  }

  if (init_data_reader_ar_i32) {
    init_data_reader_ar_i32->start();
    init_data_reader_ar_i32->read_a_batch_to_device();
  }
  if (init_data_reader_ar_i64) {
    init_data_reader_ar_i64->start();
    init_data_reader_ar_i64->read_a_batch_to_device();
  }

  size_t embed_wgrad_size = 0;
  for (size_t i = 0; i < sparse_embedding_params_.size(); i++) {
    if (sparse_embedding_params_[i].embedding_type == Embedding_t::HybridSparseEmbedding) {
      if (solver_.use_mixed_precision && solver_.i64_input_key) {
        auto hybrid_embedding =
            dynamic_cast<HybridSparseEmbedding<long long, __half>*>(embeddings_[i].get());
        hybrid_embedding->init_model(init_data_reader_ar_i64->get_value_tensors(),
                                     embed_wgrad_size);
      } else if (solver_.use_mixed_precision && !solver_.i64_input_key) {
        auto hybrid_embedding =
            dynamic_cast<HybridSparseEmbedding<unsigned int, __half>*>(embeddings_[i].get());
        hybrid_embedding->init_model(init_data_reader_ar_i32->get_value_tensors(),
                                     embed_wgrad_size);
      } else if (!solver_.use_mixed_precision && solver_.i64_input_key) {
        auto hybrid_embedding =
            dynamic_cast<HybridSparseEmbedding<long long, float>*>(embeddings_[i].get());
        hybrid_embedding->init_model(init_data_reader_ar_i64->get_value_tensors(),
                                     embed_wgrad_size);
      } else {
        auto hybrid_embedding =
            dynamic_cast<HybridSparseEmbedding<unsigned int, float>*>(embeddings_[i].get());
        hybrid_embedding->init_model(init_data_reader_ar_i32->get_value_tensors(),
                                     embed_wgrad_size);
      }
    }
  }

  if (solver_.grouped_all_reduce) {
    exchange_wgrad_->update_embed_wgrad_size(embed_wgrad_size);
  }

#ifdef ENABLE_MPI
  if (resource_manager_->get_num_process() > 1) {
    resource_manager_->set_ready_to_transfer();
  }
#endif
}

void Model::load_dense_optimizer_states(const std::string& dense_opt_states_file) {
  if (!buff_allocated_) {
    CK_THROW_(Error_t::IllegalCall,
              "Cannot load the dense optimizer states before calling Model.compile()");
  }
  load_opt_states_for_dense_(dense_opt_states_file);
}

void Model::load_sparse_optimizer_states(const std::vector<std::string>& sparse_opt_states_files) {
  if (!buff_allocated_) {
    CK_THROW_(Error_t::IllegalCall,
              "Cannot load the sparse optimizer states before calling Model.compile()");
  }
  if (etc_params_->use_embedding_training_cache) {
    CK_THROW_(Error_t::IllegalCall,
              "Cannot load the sparse optimizer states after embedding training cache is created");
  }
  if (sparse_opt_states_files.size() != embeddings_.size()) {
    CK_THROW_(Error_t::WrongInput,
              "The size of sparse opt state files should be equal to the number of embeddings");
  }
  load_opt_states_for_sparse_(sparse_opt_states_files);
}

void Model::load_sparse_optimizer_states(
    const std::map<std::string, std::string>& sparse_opt_states_files_map) {
  if (!buff_allocated_) {
    CK_THROW_(Error_t::IllegalCall,
              "Cannot load the sparse optimizer states before calling Model.compile()");
  }
  if (etc_params_->use_embedding_training_cache) {
    CK_THROW_(Error_t::IllegalCall,
              "Cannot load the sparse optimizer states after model oversubscriber is created");
  }
  for (auto iter = sparse_opt_states_files_map.begin(); iter != sparse_opt_states_files_map.end();
       iter++) {
    if (embeddings_map_.find(iter->first) == embeddings_map_.end()) {
      CK_THROW_(Error_t::WrongInput, "No such embedding name: " + iter->first);
    }
    auto embedding_target = embeddings_map_.find(iter->first)->second;
    std::ifstream sparse_opt_stream(iter->first, std::ifstream::binary);
    if (!sparse_opt_stream.is_open()) {
      CK_THROW_(Error_t::WrongInput, "Cannot open sparse optimizer states file");
    }
    MESSAGE_("Loading sparse optimizer states: " + iter->first);
    embedding_target->load_opt_states(sparse_opt_stream);
    sparse_opt_stream.close();
  }
}

void Model::load_dense_weights(const std::string& dense_model_file) {
  if (!buff_allocated_) {
    CK_THROW_(Error_t::IllegalCall, "Cannot load the dense weights before calling Model.compile()");
  }
  load_params_for_dense_(dense_model_file);
}

void Model::load_sparse_weights(const std::vector<std::string>& sparse_embedding_files) {
  if (!buff_allocated_) {
    CK_THROW_(Error_t::IllegalCall,
              "Cannot load the sparse weights before calling Model.compile()");
  }
  if (etc_params_->use_embedding_training_cache) {
    CK_THROW_(Error_t::IllegalCall,
              "Cannot load the sparse weights after embedding training cache is created");
  }
  if (sparse_embedding_files.size() != embeddings_.size()) {
    CK_THROW_(Error_t::WrongInput,
              "The size of sparse embedding files should be equal to the number of embeddings");
  }
  load_params_for_sparse_(sparse_embedding_files);
}

void Model::load_sparse_weights(
    const std::map<std::string, std::string>& sparse_embedding_files_map) {
  if (!buff_allocated_) {
    CK_THROW_(Error_t::IllegalCall,
              "Cannot load the sparse weights before calling Model.compile()");
  }
  if (etc_params_->use_embedding_training_cache) {
    CK_THROW_(Error_t::IllegalCall,
              "Cannot load the sparse weights after model oversubscriber is created");
  }
  for (auto iter = sparse_embedding_files_map.begin(); iter != sparse_embedding_files_map.end();
       iter++) {
    if (embeddings_map_.find(iter->first) == embeddings_map_.end()) {
      CK_THROW_(Error_t::WrongInput, "No such embedding name: " + iter->first);
    }
    auto embedding_target = embeddings_map_.find(iter->first)->second;
    MESSAGE_("Loading sparse model: " + iter->second);
    embedding_target->load_parameters(iter->second);
  }
}

void Model::summary() {
  if (!graph_finalized_) {
    graph_analysis();
    graph_finalized_ = true;
  }
  if (data_input_info_.size() < 3 || layer_info_.size() < 2) {
    CK_THROW_(Error_t::IllegalCall, "The model should include input and at least two layers");
  }
  for (auto tensor_entry : train_tensor_entries_list_[0]) {
    tensor_shape_info_.insert(std::make_pair(tensor_entry.name, tensor_entry.bag.get_dimensions()));
  }

  std::stringstream buf;
  buf << std::left << std::setw(40) << std::setfill(' ') << "label" << std::left << std::setw(30)
      << std::setfill(' ') << "Dense" << std::left << std::setw(30) << std::setfill(' ') << "Sparse"
      << std::endl;
  buf << std::left << std::setw(40) << std::setfill(' ') << data_input_info_[0] << std::left
      << std::setw(30) << std::setfill(' ') << data_input_info_[1] << " " << std::left
      << std::setw(30) << std::setfill(' ') << data_input_info_[2] << std::endl;
  buf << std::left << std::setw(40) << std::setfill(' ')
      << get_tensor_shape(data_input_info_[0], tensor_shape_info_) << std::left << std::setw(40)
      << std::setfill(' ') << get_tensor_shape(data_input_info_[1], tensor_shape_info_)
      << std::endl;
  buf << "—————————————————————————————————————————————————————————————————————————————————"
         "—————————————————————————————————"
      << std::endl;
  buf << std::left << std::setw(40) << std::setfill(' ') << "Layer Type" << std::left
      << std::setw(30) << std::setfill(' ') << "Input Name" << std::left << std::setw(30)
      << std::setfill(' ') << "Output Name" << std::left << std::setw(30) << std::setfill(' ')
      << "Output Shape" << std::endl;
  buf << "—————————————————————————————————————————————————————————————————————————————————"
         "—————————————————————————————————"
      << std::endl;
  for (size_t i = 0; i < layer_info_.size(); ++i) {
    std::vector<std::string> layer_type{layer_info_[i]};
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    split(input_output_info_[i].first, ',', input_names);
    split(input_output_info_[i].second, ',', output_names);
    size_t lines =
        input_names.size() > output_names.size() ? input_names.size() : output_names.size();
    layer_type.insert(layer_type.end(), lines - 1, "");
    if (lines > input_names.size()) {
      input_names.insert(input_names.end(), lines - input_names.size(), "");
    }
    if (lines > output_names.size()) {
      output_names.insert(output_names.end(), lines - output_names.size(), "");
    }
    for (size_t j = 0; j < lines; j++) {
      buf << std::left << std::setw(40) << std::setfill(' ') << layer_type[j] << std::left
          << std::setw(30) << std::setfill(' ') << input_names[j] << std::left << std::setw(30)
          << std::setfill(' ') << output_names[j] << std::left << std::setw(30) << std::setfill(' ')
          << get_tensor_shape(output_names[j], tensor_shape_info_) << std::endl;
    }
    buf << "---------------------------------------------------------------------------------"
           "---------------------------------"
        << std::endl;
  }
  HCTR_PRINT(INFO,
             "===================================================Model "
             "Summary===================================================\n"
             "%s",
             buf.str().c_str());
}

void Model::set_source(std::vector<std::string> source, std::vector<std::string> keyset,
                       std::string eval_source) {
  if (solver_.repeat_dataset || !etc_params_->use_embedding_training_cache) {
    CK_THROW_(Error_t::IllegalCall,
              "The set source method can only be used under the embedding training cache mode");
  }
  if (source.size() != keyset.size()) {
    CK_THROW_(Error_t::WrongInput,
              "The number of training data sources should equal that of the keyset files");
  }
  if (set_source_flag_) {
    etc_params_->incremental_keyset_files.insert(etc_params_->incremental_keyset_files.end(),
                                                 reader_params_.keyset.begin(),
                                                 reader_params_.keyset.end());
    set_source_flag_ = false;
  }
  reader_params_.source.assign(source.begin(), source.end());
  reader_params_.keyset.assign(keyset.begin(), keyset.end());
  reader_params_.eval_source.assign(eval_source);

  auto it{etc_params_->incremental_keyset_files.end()};
  etc_params_->incremental_keyset_files.insert(it, keyset.begin(), keyset.end());
}

void Model::set_source(std::string source, std::string eval_source) {
  if (solver_.repeat_dataset || etc_params_->use_embedding_training_cache) {
    CK_THROW_(Error_t::IllegalCall, "The set source method can only be used under the epoch mode");
  }
  std::vector<std::string>().swap(reader_params_.source);
  reader_params_.source.push_back(source);
  reader_params_.eval_source.assign(eval_source);
}

void Model::fit(int num_epochs, int max_iter, int display, int eval_interval, int snapshot,
                std::string snapshot_prefix) {
  if (!buff_allocated_) {
    CK_THROW_(Error_t::IllegalCall,
              "Cannot start the training process before calling Model.compile()");
  }
  if (solver_.repeat_dataset && max_iter <= 0) {
    CK_THROW_(Error_t::WrongInput, "Require max_iter>0 under non-epoch mode");
  }
  if (!solver_.repeat_dataset && !etc_params_->use_embedding_training_cache && num_epochs <= 0) {
    CK_THROW_(Error_t::WrongInput, "Require num_epochs>0 under epoch mode");
  }
  if (etc_params_->use_embedding_training_cache && !etc_created_) {
    CK_THROW_(Error_t::IllegalCall, "The embedding training cache should be created first");
  }
  high_level_eval_ = true;

  HugeCTR::Timer timer;
  HugeCTR::Timer timer_train;
  HugeCTR::Timer timer_eval;

  bool epoch_mode = !solver_.repeat_dataset;
  bool etc_mode = etc_params_->use_embedding_training_cache;
  int etc_epochs = num_epochs < 1 ? 1 : num_epochs;
  HCTR_PRINT(INFO,
             "=====================================================Model "
             "Fit=====================================================\n");
  if (epoch_mode && !etc_mode) {
    HCTR_LOG(INFO, ROOT, "Use epoch mode with number of epochs: %d\n", num_epochs);
  } else if (epoch_mode && etc_mode) {
    HCTR_LOG(INFO, ROOT,
             "Use embedding training cache mode with number of training sources: %zu, number of "
             "epochs: %d\n",
             reader_params_.source.size(), etc_epochs);
  } else {
    HCTR_LOG(INFO, ROOT, "Use non-epoch mode with number of iterations: %d\n", max_iter);
  }
  HCTR_LOG(INFO, ROOT, "Training batchsize: %d, evaluation batchsize: %d\n", solver_.batchsize,
           solver_.batchsize_eval);
  HCTR_LOG(INFO, ROOT, "Evaluation interval: %d, snapshot interval: %d\n", eval_interval, snapshot);
  // FYI, A string literal is statically allocated so we can assume it is safe to return it.
  auto b2s = [](const char val) { return val ? "True" : "False"; };

  HCTR_LOG(INFO, ROOT, "Dense network trainable: %s\n", b2s(is_dense_trainable_));
  for (auto iter = embeddings_map_.begin(); iter != embeddings_map_.end(); iter++) {
    HCTR_LOG(INFO, ROOT, "Sparse embedding %s trainable: %s\n", iter->first.c_str(),
             b2s(iter->second->is_trainable()));
  }
  HCTR_LOG(INFO, ROOT, "Use mixed precision: %s, scaler: %f, use cuda graph: %s\n",
           b2s(solver_.use_mixed_precision), solver_.scaler, b2s(solver_.use_cuda_graph));
  HCTR_LOG(INFO, ROOT, "lr: %f, warmup_steps: %zu, end_lr: %f\n", solver_.lr, solver_.warmup_steps,
           solver_.end_lr);
  HCTR_LOG(INFO, ROOT, "decay_start: %zu, decay_steps: %zu, decay_power: %f\n", solver_.decay_start,
           solver_.decay_steps, solver_.decay_power);
  timer.start();
  timer_train.start();

#ifdef ENABLE_PROFILING
  HugeCTR::global_profiler.initialize(solver_.use_cuda_graph);
#endif

  bool is_first_train_batch_after_eval = true;
  if (epoch_mode && !etc_mode) {
    int iter = 0;
    int batches;
    auto data_reader_train = this->get_train_data_reader();
    auto data_reader_eval = this->get_evaluate_data_reader();
    if (!data_reader_eval_status_) {
      data_reader_eval->set_source(reader_params_.eval_source);
      data_reader_eval_status_ = true;
    }
    MESSAGE_("Training source file: " + reader_params_.source[0]);
    MESSAGE_("Evaluation source file: " + reader_params_.eval_source);
    for (int e = 0; e < num_epochs; e++) {
      MESSAGE_("-----------------------------------Epoch " + std::to_string(e) +
               "-----------------------------------");
      data_reader_train->set_source(reader_params_.source[0]);
      data_reader_train_status_ = true;
      do {
        float lr = 0;
        if (!this->use_gpu_learning_rate_scheduling()) {
#ifdef ENABLE_PROFILING
          // profiler may run very long, so prevent lr < 0
          lr = std::numeric_limits<float>::min();
          this->set_learning_rate(lr);
#else
          lr = lr_sch_->get_next();
          this->set_learning_rate(lr);
#endif
        }
        data_reader_train_status_ = this->train(is_first_train_batch_after_eval);
        is_first_train_batch_after_eval = false;
        if (display > 0 && iter % display == 0 && iter != 0) {
          timer_train.stop();
          float loss = 0;
          this->get_current_loss(&loss);
          if (isnan(loss)) {
            throw std::runtime_error(std::string("Train Runtime error: Loss cannot converge") +
                                     " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n");
          }
          if (!solver_.use_holistic_cuda_graph) {
            MESSAGE_("Iter: " + std::to_string(iter) + " Time(" + std::to_string(display) +
                     " iters): " + std::to_string(timer_train.elapsedSeconds()) +
                     "s Loss: " + std::to_string(loss) + " lr:" + std::to_string(lr));
          } else {
            MESSAGE_("Iter: " + std::to_string(iter) + " Time(" + std::to_string(display) +
                     " iters): " + std::to_string(timer_train.elapsedSeconds()) +
                     "s Loss: " + std::to_string(loss));
          }
          timer_train.start();
        }
        if (eval_interval > 0 && iter % eval_interval == 0 && iter != 0) {
          this->check_overflow();
          this->copy_weights_for_evaluation();
          batches = 0;
          is_first_train_batch_after_eval = true;
          timer_eval.start();
          while (data_reader_eval_status_) {
            if (solver_.max_eval_batches == 0 || batches >= solver_.max_eval_batches) {
              break;
            }
            data_reader_eval_status_ = this->eval(batches == 0);
            batches++;
          }
          if (!data_reader_eval_status_) {
            data_reader_eval->set_source(reader_params_.eval_source);
            data_reader_eval_status_ = true;
          }
          timer_eval.stop();
          auto eval_metrics = this->get_eval_metrics();
          for (auto& eval_metric : eval_metrics) {
            MESSAGE_("Evaluation, " + eval_metric.first + ": " +
                     std::to_string(eval_metric.second));
            if (!eval_metric.first.compare("AUC")) {
              const auto auc_threshold = solver_.metrics_spec[HugeCTR::metrics::Type::AUC];
              if (eval_metric.second > auc_threshold) {
                timer.stop();
                HCTR_LOG(INFO, ROOT,
                         "Hit target accuracy AUC %f at %d / %d epochs"
                         " %d global iterations with batchsize %d in %.2fs."
                         " Average speed %f records/s.\n",
                         auc_threshold, e, num_epochs, iter, solver_.batchsize,
                         timer.elapsedSeconds(),
                         float(iter) * solver_.batchsize / timer.elapsedSeconds());
                return;
              }
            }
          }
          MESSAGE_("Eval Time for " + std::to_string(solver_.max_eval_batches) +
                   " iters: " + std::to_string(timer_eval.elapsedSeconds()) + "s");
        }
        if (snapshot > 0 && iter % snapshot == 0 && iter != 0) {
          this->download_params_to_files(snapshot_prefix, iter);
        }
        iter++;
      } while (data_reader_train_status_);
      timer.stop();
    }  // end for epoch
    HCTR_LOG(INFO, ROOT, "Finish %d epochs %d global iterations with batchsize %d in %.2fs.\n",
             num_epochs, iter, solver_.batchsize, timer.elapsedSeconds());
  } else if (epoch_mode && etc_mode) {
    int iter = 0;
    int batches;
    auto data_reader_train = this->get_train_data_reader();
    auto data_reader_eval = this->get_evaluate_data_reader();
    auto embedding_training_cache = this->get_embedding_training_cache();
    if (!data_reader_eval_status_) {
      data_reader_eval->set_source(reader_params_.eval_source);
      data_reader_eval_status_ = true;
    }
    MESSAGE_("Evaluation source file: " + reader_params_.eval_source);
    bool is_first_train_batch_after_eval = true;
    for (int e = 0; e < etc_epochs; e++) {
      for (unsigned int f = 0; f < reader_params_.source.size(); f++) {
        MESSAGE_("--------------------Epoch " + std::to_string(e) +
                 ", source file: " + reader_params_.source[f] + "--------------------");
        data_reader_train->set_source(reader_params_.source[f]);
        data_reader_train_status_ = true;
        embedding_training_cache->update(reader_params_.keyset[f]);
        do {
          float lr = 0;
          if (!this->use_gpu_learning_rate_scheduling()) {
#ifdef ENABLE_PROFILING
            // profiler may run very long, so prevent lr < 0
            lr = std::numeric_limits<float>::min();
            this->set_learning_rate(lr);
#else
            lr = lr_sch_->get_next();
            this->set_learning_rate(lr);
#endif
          }
          data_reader_train_status_ = this->train(is_first_train_batch_after_eval);
          is_first_train_batch_after_eval = false;
          if (display > 0 && iter % display == 0 && iter != 0) {
            timer_train.stop();
            float loss = 0;
            this->get_current_loss(&loss);
            if (isnan(loss)) {
              throw std::runtime_error(std::string("Train Runtime error: Loss cannot converge") +
                                       " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n");
            }
            if (!solver_.use_holistic_cuda_graph) {
              MESSAGE_("Iter: " + std::to_string(iter) + " Time(" + std::to_string(display) +
                       " iters): " + std::to_string(timer_train.elapsedSeconds()) +
                       "s Loss: " + std::to_string(loss) + " lr:" + std::to_string(lr));
            } else {
              MESSAGE_("Iter: " + std::to_string(iter) + " Time(" + std::to_string(display) +
                       " iters): " + std::to_string(timer_train.elapsedSeconds()) +
                       "s Loss: " + std::to_string(loss));
            }
            timer_train.start();
          }
          if (eval_interval > 0 && iter % eval_interval == 0 && iter != 0) {
            this->check_overflow();
            this->copy_weights_for_evaluation();
            batches = 0;
            is_first_train_batch_after_eval = true;
            timer_eval.start();
            while (data_reader_eval_status_) {
              if (solver_.max_eval_batches == 0 || batches >= solver_.max_eval_batches) {
                break;
              }
              data_reader_eval_status_ = this->eval(batches == 0);
              batches++;
            }
            if (!data_reader_eval_status_) {
              data_reader_eval->set_source(reader_params_.eval_source);
              data_reader_eval_status_ = true;
            }
            timer_eval.stop();
            auto eval_metrics = this->get_eval_metrics();
            for (auto& eval_metric : eval_metrics) {
              MESSAGE_("Evaluation, " + eval_metric.first + ": " +
                       std::to_string(eval_metric.second));
            }
            MESSAGE_("Eval Time for " + std::to_string(solver_.max_eval_batches) +
                     " iters: " + std::to_string(timer_eval.elapsedSeconds()) + "s");
          }
          iter++;
        } while (data_reader_train_status_);
      }  // end for file list
    }    // end for epoch
  } else {
    MESSAGE_("Training source file: " + reader_params_.source[0]);
    MESSAGE_("Evaluation source file: " + reader_params_.eval_source);
    this->start_data_reading();
    bool is_first_train_batch_after_eval = true;
    for (int iter = 0; iter < max_iter; iter++) {
      float lr = 0;
      if (!this->use_gpu_learning_rate_scheduling()) {
#ifdef ENABLE_PROFILING
        // profiler may run very long, so prevent lr < 0
        lr = std::numeric_limits<float>::min();
        this->set_learning_rate(lr);
#else
        lr = lr_sch_->get_next();
        this->set_learning_rate(lr);
#endif
      }
      this->train(is_first_train_batch_after_eval);
      is_first_train_batch_after_eval = false;
#ifdef ENABLE_PROFILING
      iter = 0;
      continue;
#endif
      if (display > 0 && iter % display == 0 && iter != 0) {
        timer_train.stop();
        float loss = 0;
        this->get_current_loss(&loss);
        if (isnan(loss)) {
          throw std::runtime_error(std::string("Train Runtime error: Loss cannot converge") + " " +
                                   __FILE__ + ":" + std::to_string(__LINE__) + " \n");
        }
        if (!solver_.use_holistic_cuda_graph) {
          MESSAGE_("Iter: " + std::to_string(iter) + " Time(" + std::to_string(display) +
                   " iters): " + std::to_string(timer_train.elapsedSeconds()) +
                   "s Loss: " + std::to_string(loss) + " lr:" + std::to_string(lr));
        } else {
          MESSAGE_("Iter: " + std::to_string(iter) + " Time(" + std::to_string(display) +
                   " iters): " + std::to_string(timer_train.elapsedSeconds()) +
                   "s Loss: " + std::to_string(loss));
        }
        timer_train.start();
      }
      if (eval_interval > 0 && iter % eval_interval == 0 && iter != 0) {
        this->check_overflow();
        this->copy_weights_for_evaluation();
        is_first_train_batch_after_eval = true;
        timer_eval.start();
        for (int batches = 0; batches < solver_.max_eval_batches; batches++) {
          this->eval(batches == 0);
        }
        timer_eval.stop();
        auto eval_metrics = this->get_eval_metrics();
        for (auto& eval_metric : eval_metrics) {
          MESSAGE_("Evaluation, " + eval_metric.first + ": " + std::to_string(eval_metric.second));
          if (!eval_metric.first.compare("AUC")) {
            const auto auc_threshold = solver_.metrics_spec[HugeCTR::metrics::Type::AUC];
            if (eval_metric.second > auc_threshold) {
              timer.stop();
              HCTR_LOG(INFO, ROOT,
                       "Hit target accuracy AUC %f at %d / %d iterations with batchsize %d"
                       " in %.2fs. Average speed %f records/s.\n",
                       auc_threshold, iter, max_iter, solver_.batchsize, timer.elapsedSeconds(),
                       float(iter) * solver_.batchsize / timer.elapsedSeconds());
              return;
            }
          }
        }
        MESSAGE_("Eval Time for " + std::to_string(solver_.max_eval_batches) +
                 " iters: " + std::to_string(timer_eval.elapsedSeconds()) + "s");
      }
      if (snapshot > 0 && iter % snapshot == 0 && iter != 0) {
        this->download_params_to_files(snapshot_prefix, iter);
      }
    }  // end for iter

    timer.stop();
    HCTR_LOG(INFO, ROOT, "Finish %d iterations with batchsize: %d in %.2fs.\n", max_iter,
             solver_.batchsize, timer.elapsedSeconds());
  }  // end if else
  high_level_eval_ = false;
}

void Model::exchange_wgrad(size_t device_id) {
  auto& gpu_resource = resource_manager_->get_local_gpu(device_id);
  CudaCPUDeviceContext context(gpu_resource->get_device_id());
  // CudaDeviceContext context(gpu_resource->get_device_id());
  PROFILE_RECORD("exchange_wgrad.start", gpu_resource->get_stream(), false);
  if (solver_.async_mlp_wgrad) gpu_resource->wait_on_wgrad_event(gpu_resource->get_stream());
  exchange_wgrad_->allreduce(device_id, gpu_resource->get_stream());
  PROFILE_RECORD("exchange_wgrad.stop", gpu_resource->get_stream(), false);
}

void Model::train_overlapped() {
  auto change_state = [](TrainState* state) -> bool {
    switch (state->state) {
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
        CK_THROW_(Error_t::InvalidEnv, "model state reached invalid status");
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
    // CudaDeviceContext context(device_id);
    long long current_batchsize_per_device =
        train_data_reader_->get_current_batchsize_per_device(id);

    TrainState state;
    auto sync = [&state, &stream, id]() {
      if (state.event) {
        CK_CUDA_THROW_(cudaStreamWaitEvent(stream, *state.event));
      }
      state.event = nullptr;
    };

    auto schedule_reader = [&, this](TrainState_t expected) {
      if (scheduled_reader && state.state == expected) {
        if (solver_.use_holistic_cuda_graph) {
          scheduled_reader->schedule_here_graph(stream, id);
        } else {
          scheduled_reader->schedule_here(stream, id);
        }
        graph_scheduler_->record_execution(id, stream);
      }
    };

    auto schedule_split3way = [&, this](TrainState_t state_to_schedule) {
      if (state.state == state_to_schedule) {
        scheduled_reader->schedule_precompute_here(stream, id, solver_.use_holistic_cuda_graph);
      }
    };

    auto schedule_d2d = [&, this](TrainState_t state_to_schedule) {
      if (state.state == state_to_schedule) {
        scheduled_reader->schedule_d2d_here(stream, id, solver_.use_holistic_cuda_graph);
      }
    };

    auto do_it = [&, this](cudaStream_t submit_stream) {
      if (solver_.use_holistic_cuda_graph) {
        CK_CUDA_THROW_(cudaEventRecord(fork_events_[id], submit_stream));
        state.event = &fork_events_[id];
      }

      // Network just runs unconditionally
      // Embedding manages events from the networks and waits if necessary
      // Session inserts a wait if it gets a non-null event from the embedding

      do {
        state = embeddings_[0]->train(true, id, state);
        sync();
        if (resource_manager_->get_num_process() == 1) {
          schedule_reader(TrainState_t::TopMLPFprop);
          schedule_split3way(TrainState_t::MLPExchangeWgrad);
          schedule_d2d(TrainState_t::MLPUpdate);
        }
        state = networks_[id]->train(
            current_batchsize_per_device, [this, id]() { this->exchange_wgrad(id); }, state);
        if (resource_manager_->get_num_process() > 1) {
          schedule_reader(TrainState_t::TopMLPFprop);
          schedule_split3way(TrainState_t::BottomMLPBprop);
          schedule_d2d(TrainState_t::MLPExchangeWgrad);
        }
      } while (change_state(&state));
      sync();
    };

    if (solver_.use_holistic_cuda_graph) {
      if (!train_graphs_[id].initialized) {
        train_graphs_[id].capture(do_it, stream);
#ifdef ENABLE_MPI
#pragma omp master
        MPI_Barrier(MPI_COMM_WORLD);
#endif
#pragma omp barrier
      }
      train_graphs_[id].exec(stream);
      if (scheduled_reader) {
        scheduled_reader->update_schedule_graph(id);
      }
    } else {
      do_it(stream);
    }
  }
}

bool Model::train(bool is_first_batch) {
  try {
    if (train_data_reader_->is_started() == false) {
      CK_THROW_(Error_t::IllegalCall, "Start the data reader first before calling Model::train()");
    }
#ifndef DATA_READING_TEST
    // TODO: assuming the there are enough training iterations, incomplete batches
    // are discarded, so that we can bypass the runtime error in the epoch mode,
    // whilst maintaining the dense network training logic.
    // To minimize the wasted batches, consider to adjust # of data reader workers.
    // For instance, with a file list source, set "num_workers" to a dvisior of
    // the number of data files in the file list.
    // We will look into some alternatives in the long term.
    if (solver_.is_dlrm) {
      graph_scheduler_->trickling();
    }
    long long current_batchsize = 0;
    while ((current_batchsize = train_data_reader_->read_a_batch_to_device_delay_release()) &&
           (current_batchsize < train_data_reader_->get_full_batchsize()) && !solver_.is_dlrm) {
      MESSAGE_("train drop incomplete batch. batchsize:" + std::to_string(current_batchsize));
      train_data_reader_->ready_to_collect();
    }
    if (!current_batchsize) {
      return false;
    }
#pragma omp parallel num_threads(networks_.size())
    {
      size_t id = omp_get_thread_num();
      CudaCPUDeviceContext ctx(resource_manager_->get_local_gpu(id)->get_device_id());
      // CudaDeviceContext context(resource_manager_->get_local_gpu(id)->get_device_id());
      cudaStreamSynchronize(resource_manager_->get_local_gpu(id)->get_stream());
    }
    train_data_reader_->ready_to_collect();
#ifdef ENABLE_PROFILING
    global_profiler.iter_check();
#endif

    if (solver_.use_overlapped_pipeline) {
      train_overlapped();
    } else {
      for (auto& one_embedding : embeddings_) {
        one_embedding->forward(true, is_first_batch);
      }
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
      for (auto& one_embedding : embeddings_) {
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
      } else {
        const auto& local_gpu = resource_manager_->get_local_gpu(0);
        local_gpu->set_compute2_event_sync(local_gpu->get_comp_overlap_stream());
        local_gpu->wait_on_compute2_event(local_gpu->get_stream());
      }
    }
    return true;
#else
    train_data_reader_->read_a_batch_to_device();
#endif
  } catch (const std::exception& err) {
    Logger::print_exception(err, 0);
    throw err;
  }
}

bool Model::eval(bool is_first_batch) {
  try {
    if (evaluate_data_reader_ == nullptr) return true;
    if (evaluate_data_reader_->is_started() == false) {
      CK_THROW_(Error_t::IllegalCall, "Start the data reader first before calling Model::eval()");
    }
    if (!high_level_eval_) {
      this->check_overflow();
      this->copy_weights_for_evaluation();
    }
    long long current_batchsize = 0;
    while ((current_batchsize = evaluate_data_reader_->read_a_batch_to_device_delay_release()) &&
           (current_batchsize < evaluate_data_reader_->get_full_batchsize()) && !solver_.is_dlrm) {
      MESSAGE_("eval drop incomplete batch. batchsize:" + std::to_string(current_batchsize));
      evaluate_data_reader_->ready_to_collect();
    }
    for (auto& metric : metrics_) {
      metric->set_current_batch_size(current_batchsize);
    }
    current_eval_batchsize_ = current_batchsize;
    if (!current_batchsize) {
      return false;
    }
    evaluate_data_reader_->ready_to_collect();
#ifndef DATA_READING_TEST
    for (auto& one_embedding : embeddings_) {
      one_embedding->forward(false, is_first_batch);
    }

    if (networks_.size() > 1) {
#pragma omp parallel num_threads(networks_.size())
      {
        size_t id = omp_get_thread_num();
        long long current_batchsize_per_device =
            evaluate_data_reader_->get_current_batchsize_per_device(id);
        networks_[id]->eval(current_batchsize_per_device);
        for (auto& metric : metrics_) {
          metric->local_reduce(id, networks_[id]->get_raw_metrics());
        }
      }
    } else if (networks_.size() == 1) {
      long long current_batchsize_per_device =
          evaluate_data_reader_->get_current_batchsize_per_device(0);
      networks_[0]->eval(current_batchsize_per_device);
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
  } catch (const std::exception& err) {
    Logger::print_exception(err, 0);
    throw err;
  }
}

Error_t Model::export_predictions(const std::string& output_prediction_file_name,
                                  const std::string& output_label_file_name) {
  try {
    if (current_eval_batchsize_ == 0) {
      MESSAGE_("Reach end of eval dataset. Skip export prediction");
      return Error_t::Success;
    }
    CudaDeviceContext context;
    const std::vector<int>& local_gpu_device_id_list =
        resource_manager_->get_local_gpu_device_id_list();
    const size_t global_gpu_count = resource_manager_->get_global_gpu_count();
    const size_t local_gpu_count = resource_manager_->get_local_gpu_count();
    size_t batchsize_eval_per_gpu = solver_.batchsize_eval / global_gpu_count;
    size_t total_prediction_count = batchsize_eval_per_gpu * local_gpu_count;
    std::unique_ptr<float[]> local_prediction_result(
        new float[total_prediction_count * input_params_[0].label_dim]);
    std::unique_ptr<float[]> local_label_result(
        new float[total_prediction_count * input_params_[0].label_dim]);

    for (unsigned int i = 0; i < networks_.size(); ++i) {
      int gpu_id = local_gpu_device_id_list[i];
      context.set_device(gpu_id);

      get_raw_metric_as_host_float_tensor(
          networks_[i]->get_raw_metrics(), metrics::RawType::Pred, solver_.use_mixed_precision,
          local_prediction_result.get() + batchsize_eval_per_gpu * input_params_[0].label_dim * i,
          batchsize_eval_per_gpu * input_params_[0].label_dim);
      get_raw_metric_as_host_float_tensor(
          networks_[i]->get_raw_metrics(), metrics::RawType::Label, false,
          local_label_result.get() + batchsize_eval_per_gpu * input_params_[0].label_dim * i,
          batchsize_eval_per_gpu * input_params_[0].label_dim);
    }

    std::unique_ptr<float[]> global_prediction_result;
    std::unique_ptr<float[]> global_label_result;
    int numprocs = 1;
    int pid = 0;
#ifdef ENABLE_MPI
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
#endif
    if (numprocs > 1) {
#ifdef ENABLE_MPI
      if (pid == 0) {
        global_prediction_result.reset(
            new float[solver_.batchsize_eval * input_params_[0].label_dim]);
        global_label_result.reset(new float[solver_.batchsize_eval * input_params_[0].label_dim]);
      }
      CK_MPI_THROW_(MPI_Gather(
          local_prediction_result.get(), total_prediction_count * input_params_[0].label_dim,
          MPI_FLOAT, global_prediction_result.get(),
          total_prediction_count * input_params_[0].label_dim, MPI_FLOAT, 0, MPI_COMM_WORLD));
      CK_MPI_THROW_(MPI_Gather(
          local_label_result.get(), total_prediction_count * input_params_[0].label_dim, MPI_FLOAT,
          global_label_result.get(), total_prediction_count * input_params_[0].label_dim, MPI_FLOAT,
          0, MPI_COMM_WORLD));
#endif
    } else {
      global_prediction_result = std::move(local_prediction_result);
      global_label_result = std::move(local_label_result);
    }
    if (pid == 0) {
      // write
      auto write_func = [](const std::string& output_file_name, float* res, size_t num) {
        std::ofstream output_stream(output_file_name, std::ios::out | std::ios::app);
        if (!output_stream.is_open()) {
          CK_THROW_(Error_t::WrongInput, "Cannot open output file " + output_file_name);
        }
        for (unsigned int i = 0; i < num; ++i) {
          output_stream << res[i] << " ";
        }
        output_stream.close();
      };
      write_func(output_prediction_file_name, global_prediction_result.get(),
                 current_eval_batchsize_ * input_params_[0].label_dim);
      write_func(output_label_file_name, global_label_result.get(),
                 current_eval_batchsize_ * input_params_[0].label_dim);
    }
  } catch (const internal_runtime_error& rt_err) {
    Logger::print_exception(rt_err, 0);
    return rt_err.get_error();
  } catch (const std::exception& err) {
    Logger::print_exception(err, 0);
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
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
    Logger::print_exception(rt_err, 0);
    return rt_err.get_error();
  } catch (const std::exception& err) {
    Logger::print_exception(err, 0);
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

Error_t Model::download_params_to_files(std::string prefix, int iter) {
  std::string snapshot_dense_name = prefix + "_dense_" + std::to_string(iter) + ".model";
  std::string snapshot_dense_opt_name = prefix + "_opt_dense_" + std::to_string(iter) + ".model";
  std::vector<std::string> snapshot_sparse_names;
  std::vector<std::string> snapshot_sparse_opt_names;
  for (unsigned int i = 0; i < embeddings_.size(); i++) {
    snapshot_sparse_names.push_back(prefix + std::to_string(i) + "_sparse_" + std::to_string(iter) +
                                    ".model");
    snapshot_sparse_opt_names.push_back(prefix + std::to_string(i) + "_opt_sparse_" +
                                        std::to_string(iter) + ".model");
  }
  if (etc_params_->use_embedding_training_cache) {
    embedding_training_cache_->dump();
    embedding_training_cache_->update_sparse_model_file();
  } else {
    download_sparse_params_to_files_(snapshot_sparse_names, snapshot_sparse_opt_names);
  }
  return download_dense_params_to_files_(snapshot_dense_name, snapshot_dense_opt_name);
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

Error_t Model::download_dense_params_to_files_(std::string weights_file,
                                               std::string dense_opt_states_file) {
  try {
    if (resource_manager_->is_master_process()) {
      std::ofstream out_stream_weight(weights_file, std::ofstream::binary);
      networks_[0]->download_params_to_host(out_stream_weight);
      MESSAGE_("Dumping dense weights to file, successful");
      std::ofstream out_dense_opt_state_weight(dense_opt_states_file, std::ofstream::binary);
      networks_[0]->download_opt_states_to_host(out_dense_opt_state_weight);
      MESSAGE_("Dumping dense optimizer states to file, successful");
      std::string no_trained_params = networks_[0]->get_no_trained_params_in_string();
      if (no_trained_params.length() != 0) {
        std::string ntp_file = weights_file + ".ntp.json";
        std::ofstream out_stream_ntp(ntp_file, std::ofstream::out);
        out_stream_ntp.write(no_trained_params.c_str(), no_trained_params.length());
        out_stream_ntp.close();
      }
      MESSAGE_("Dumping untrainable weights to file, successful");
      out_stream_weight.close();
      out_dense_opt_state_weight.close();
    }
  } catch (const internal_runtime_error& rt_err) {
    Logger::print_exception(rt_err, 0);
    return rt_err.get_error();
  } catch (const std::exception& err) {
    Logger::print_exception(err, 0);
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

Error_t Model::download_sparse_params_to_files_(
    const std::vector<std::string>& embedding_files,
    const std::vector<std::string>& sparse_opt_state_files) {
  try {
    {
      int i = 0;
      for (auto& embedding_file : embedding_files) {
        embeddings_[i]->dump_parameters(embedding_file);
        i++;
      }
    }
    MESSAGE_("Dumping sparse weights to files, successful");
    {
      int i = 0;
      for (auto& sparse_opt_state_file : sparse_opt_state_files) {
        std::ofstream out_stream_opt(sparse_opt_state_file, std::ofstream::binary);
        embeddings_[i]->dump_opt_states(out_stream_opt);
        out_stream_opt.close();
        i++;
      }
    }
    MESSAGE_("Dumping sparse optimzer states to files, successful");
  } catch (const internal_runtime_error& rt_err) {
    Logger::print_exception(rt_err, 0);
    return rt_err.get_error();
  } catch (const std::exception& err) {
    Logger::print_exception(err, 0);
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

template <typename TypeEmbeddingComp>
std::shared_ptr<EmbeddingTrainingCache> Model::create_embedding_training_cache_(
    const std::vector<TrainPSType_t>& ps_types,
    const std::vector<std::string>& sparse_embedding_files,
    const std::vector<std::string>& local_paths,
    const std::vector<HMemCacheConfig>& hmem_cache_configs) {
  try {
    return std::shared_ptr<EmbeddingTrainingCache>(new EmbeddingTrainingCache(
        ps_types, embeddings_, sparse_embedding_files, resource_manager_,
        solver_.use_mixed_precision, solver_.i64_input_key, local_paths, hmem_cache_configs));
  } catch (const internal_runtime_error& rt_err) {
    Logger::print_exception(rt_err, 0);
    throw rt_err;
  } catch (const std::exception& err) {
    Logger::print_exception(err, 0);
    throw err;
  }
}

Error_t Model::load_opt_states_for_dense_(const std::string& dense_opt_states_file) {
  try {
    std::ifstream opt_states_stream(dense_opt_states_file, std::ifstream::binary);
    if (!opt_states_stream.is_open()) {
      CK_THROW_(Error_t::WrongInput, "Cannot open dense opt states file");
    }
    size_t opt_states_size_in_byte = networks_[0]->get_opt_states_size_in_byte();
    std::unique_ptr<char[]> opt_states(new char[opt_states_size_in_byte]());
    opt_states_stream.read(opt_states.get(), opt_states_size_in_byte);

    MESSAGE_("Loading dense opt states: " + dense_opt_states_file);
    for (auto& network : networks_) {
      network->upload_opt_states_to_device(opt_states.get());
    }
    opt_states_stream.close();
  } catch (const internal_runtime_error& rt_err) {
    Logger::print_exception(rt_err, 0);
    return rt_err.get_error();
  } catch (const std::exception& err) {
    Logger::print_exception(err, 0);
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

Error_t Model::load_opt_states_for_sparse_(
    const std::vector<std::string>& sparse_opt_states_files) {
  try {
    for (size_t i = 0; i < embeddings_.size(); i++) {
      if (i < sparse_opt_states_files.size()) {
        std::ifstream sparse_opt_stream(sparse_opt_states_files[i], std::ifstream::binary);
        if (!sparse_opt_stream.is_open()) {
          CK_THROW_(Error_t::WrongInput, "Cannot open sparse optimizer states file");
        }
        MESSAGE_("Loading sparse optimizer states: " + sparse_opt_states_files[i]);
        embeddings_[i]->load_opt_states(sparse_opt_stream);
        sparse_opt_stream.close();
      }
    }
  } catch (const internal_runtime_error& rt_err) {
    Logger::print_exception(rt_err, 0);
    return rt_err.get_error();
  } catch (const std::exception& err) {
    Logger::print_exception(err, 0);
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

Error_t Model::load_params_for_dense_(const std::string& model_file) {
  try {
    std::ifstream model_stream(model_file, std::ifstream::binary);
    if (!model_stream.is_open()) {
      CK_THROW_(Error_t::WrongInput, "Cannot open dense model file");
    }
    std::unique_ptr<float[]> weight(new float[networks_[0]->get_params_num()]());
    model_stream.read(reinterpret_cast<char*>(weight.get()),
                      networks_[0]->get_params_num() * sizeof(float));
    MESSAGE_("Loading dense model: " + model_file);
    for (auto& network : networks_) {
      network->upload_params_to_device(weight.get());
    }
    model_stream.close();
  } catch (const internal_runtime_error& rt_err) {
    Logger::print_exception(rt_err, 0);
    return rt_err.get_error();
  } catch (const std::exception& err) {
    Logger::print_exception(err, 0);
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

Error_t Model::load_params_for_sparse_(const std::vector<std::string>& embedding_model_files) {
  try {
    for (size_t i = 0; i < embeddings_.size(); i++) {
      if (i < embedding_model_files.size()) {
        MESSAGE_("Loading sparse model: " + embedding_model_files[i]);
        embeddings_[i]->load_parameters(embedding_model_files[i]);
      }
    }
  } catch (const internal_runtime_error& rt_err) {
    Logger::print_exception(rt_err, 0);
    return rt_err.get_error();
  } catch (const std::exception& err) {
    Logger::print_exception(err, 0);
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

void Model::init_params_for_dense_() {
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    networks_[i]->init_params(i);
  }
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    CK_CUDA_THROW_(cudaStreamSynchronize(resource_manager_->get_local_gpu(i)->get_stream()));
  }
}

void Model::init_params_for_sparse_() {
  for (size_t i = 0; i < embeddings_.size(); i++) {
    embeddings_[i]->init_params();
  }
}

void Model::init_embedding_training_cache_(const std::vector<TrainPSType_t>& ps_types,
                                           const std::vector<std::string>& sparse_embedding_files,
                                           const std::vector<std::string>& local_paths,
                                           const std::vector<HMemCacheConfig>& hmem_cache_configs) {
  if (solver_.use_mixed_precision) {
    embedding_training_cache_ = create_embedding_training_cache_<__half>(
        ps_types, sparse_embedding_files, local_paths, hmem_cache_configs);
  } else {
    embedding_training_cache_ = create_embedding_training_cache_<float>(
        ps_types, sparse_embedding_files, local_paths, hmem_cache_configs);
  }
  etc_created_ = true;
}

std::vector<std::pair<std::vector<long long>, std::vector<float>>>& Model::get_incremental_model() {
  if (!etc_params_->use_embedding_training_cache) {
    CK_THROW_(Error_t::IllegalCall, "Get incremental is only supported in ETC");
  }
  if (set_source_flag_) {
    etc_params_->incremental_keyset_files.insert(etc_params_->incremental_keyset_files.end(),
                                                 reader_params_.keyset.begin(),
                                                 reader_params_.keyset.end());
    set_source_flag_ = false;
  }
  // dump model from GPU to PS
  embedding_training_cache_->dump();
  // load keyset to vector (processed keys_vec should be long long format)
  auto& inc_keyset_file{etc_params_->incremental_keyset_files};
  std::vector<long long> keys_vec;
  if (solver_.i64_input_key) {
    keys_vec = load_key_files<long long>(inc_keyset_file);
  } else {
    auto keys_i32_vec = load_key_files<unsigned>(inc_keyset_file);
    keys_vec.resize(keys_i32_vec.size());
    std::transform(keys_i32_vec.begin(), keys_i32_vec.end(), keys_vec.begin(),
                   [](unsigned key) { return static_cast<long long>(key); });
  }
  inc_keyset_file.clear();
  // get the incremental sparse model
  inc_sparse_model_ = embedding_training_cache_->get_incremental_model(keys_vec);
  return inc_sparse_model_;
}

void Model::dump_incremental_model_2kafka() {
  if (!etc_params_->use_embedding_training_cache) {
    CK_THROW_(Error_t::IllegalCall, "Get incremental is only supported in ETC");
  }
  if (set_source_flag_) {
    etc_params_->incremental_keyset_files.insert(etc_params_->incremental_keyset_files.end(),
                                                 reader_params_.keyset.begin(),
                                                 reader_params_.keyset.end());
    set_source_flag_ = false;
  }
  // dump model from GPU to PS
  embedding_training_cache_->dump();
  // load keyset to vector (processed keys_vec should be long long format)
  auto& inc_keyset_file{etc_params_->incremental_keyset_files};
  std::vector<long long> keys_vec;
  if (solver_.i64_input_key) {
    keys_vec = load_key_files<long long>(inc_keyset_file);
  } else {
    auto keys_i32_vec = load_key_files<unsigned>(inc_keyset_file);
    keys_vec.resize(keys_i32_vec.size());
    std::transform(keys_i32_vec.begin(), keys_i32_vec.end(), keys_vec.begin(),
                   [](unsigned key) { return static_cast<long long>(key); });
  }
  inc_keyset_file.clear();
  // get the incremental sparse model
  inc_sparse_model_ = embedding_training_cache_->get_incremental_model(keys_vec);

  for (unsigned int i = 0; i < sparse_embedding_params_.size(); i++) {
    size_t embedding_size = sparse_embedding_params_[i].embedding_vec_size;
    std::string table_name = sparse_embedding_params_[i].sparse_embedding_name;
    std::string tag = parameter_server_base::make_tag_name(solver_.model_name, table_name);
    const char* vectors_ = reinterpret_cast<const char*>(inc_sparse_model_[i].second.data());
    size_t num_pairs = inc_sparse_model_[i].first.size();
    HCTR_LOG(
        INFO, WORLD,
        "Dump incremental parameters of %s into kafka. Embedding size is %zd, num_pairs is %zd \n",
        tag.c_str(), embedding_size, num_pairs);
    message_sink_->post(tag, num_pairs, inc_sparse_model_[i].first.data(), vectors_,
                        embedding_size * sizeof(float));
  }
}

}  // namespace HugeCTR
