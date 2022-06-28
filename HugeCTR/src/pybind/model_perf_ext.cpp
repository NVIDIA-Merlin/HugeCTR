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

#include <data_readers/async_reader/data_reader_scheduling.hpp>
#include <pybind/model_perf_ext.hpp>

namespace HugeCTR {

namespace {

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

}  // Anonymous namespace

ModelPerfExt::ModelPerfExt(const Solver& solver, const DataReaderParams& reader_params,
                           std::shared_ptr<OptParamsPy>& opt_params_py,
                           std::shared_ptr<EmbeddingTrainingCacheParams>& etc_params)
    : Model(solver, reader_params, opt_params_py, etc_params) {
  graph_scheduler_ = std::make_unique<GraphScheduler>(resource_manager_);
}

bool ModelPerfExt::train(bool is_first_batch) {
  try {
    if (train_data_reader_->is_started() == false) {
      HCTR_OWN_THROW(Error_t::IllegalCall,
                     "Start the data reader first before calling Model::train()");
    }
    graph_scheduler_->trickling();
    // When async indices is enabled, we prefetch next batch on internal stream, that stream will
    // block until the schedule event in the graph is executed
    train_data_reader_->read_a_batch_to_device_delay_release();
    train_data_reader_->ready_to_collect();

#ifdef ENABLE_PROFILING
    global_profiler.iter_check();
#endif
    if (solver_.all_reduce_algo == AllReduceAlgo::NCCL and
        train_data_reader_->current_batch_incomplete()) {
#pragma omp parallel num_threads(networks_.size())
      {
        size_t id = omp_get_thread_num();
        CudaCPUDeviceContext ctx(resource_manager_->get_local_gpu(id)->get_device_id());
        cudaStreamSynchronize(resource_manager_->get_local_gpu(id)->get_stream());
      }
    }

    if (solver_.use_overlapped_pipeline) {
      train_overlapped();
    } else {
      embeddings_[0]->forward(true, is_first_batch);

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

      embeddings_[0]->backward();

      if (networks_.size() > 1) {
#pragma omp parallel num_threads(networks_.size())
        {
          size_t id = omp_get_thread_num();
          exchange_wgrad(id);
          networks_[id]->update_params();
        }
      }

      embeddings_[0]->update_params();

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
  } catch (const internal_runtime_error& err) {
    HCTR_LOG_S(ERROR, WORLD) << err.what() << std::endl;
    throw err;
  } catch (const std::exception& err) {
    HCTR_LOG_S(ERROR, WORLD) << err.what() << std::endl;
    throw err;
  }
}

bool ModelPerfExt::eval(bool is_first_batch) {
  try {
    if (evaluate_data_reader_ == nullptr) return true;
    if (evaluate_data_reader_->is_started() == false) {
      HCTR_OWN_THROW(Error_t::IllegalCall,
                     "Start the data reader first before calling Model::eval()");
    }
    if (!high_level_eval_) {
      this->check_overflow();
      this->copy_weights_for_evaluation();
    }
    if (is_first_batch && solver_.eval_overlap) {
      auto scheduled_reader = dynamic_cast<IDataReaderWithScheduling*>(evaluate_data_reader_.get());
#pragma omp parallel num_threads(networks_.size())
      {
        size_t id = omp_get_thread_num();
        scheduled_reader->schedule_precompute_here(
            resource_manager_->get_local_gpu(id)->get_stream(), id, false);
      }
    }
    long long current_batchsize = evaluate_data_reader_->read_a_batch_to_device_delay_release();
    for (auto& metric : metrics_) {
      metric->set_current_batch_size(current_batchsize);
    }
    current_eval_batchsize_ = current_batchsize;
    evaluate_data_reader_->ready_to_collect();
    embeddings_[0]->forward(false, is_first_batch);

#pragma omp parallel num_threads(networks_.size())
    {
      size_t id = omp_get_thread_num();
      long long current_batchsize_per_device =
          evaluate_data_reader_->get_current_batchsize_per_device(id);

      // doesn't do anything if eval_overlap disabled
      auto gpu = resource_manager_->get_local_gpu(id);
      HCTR_LIB_THROW(cudaStreamWaitEvent(gpu->get_stream(), gpu->get_event("eval_comp_wait")));

      networks_[id]->eval(current_batchsize_per_device);

      // doesn't do anything if eval_overlap disabled
      HCTR_LIB_THROW(cudaEventRecord(gpu->get_event("eval_comm_wait"), gpu->get_stream()));

      for (auto& metric : metrics_) {
        metric->local_reduce(id, networks_[id]->get_raw_metrics_all().begin()->second);
      }
    }

    for (auto& metric : metrics_) {
      metric->global_reduce(networks_.size());
    }

    return true;
  } catch (const internal_runtime_error& err) {
    HCTR_LOG_S(ERROR, WORLD) << err.what() << std::endl;
    throw err;
  } catch (const std::exception& err) {
    HCTR_LOG_S(ERROR, WORLD) << err.what() << std::endl;
    throw err;
  }
}

void ModelPerfExt::fit(int num_epochs, int max_iter, int display, int eval_interval, int snapshot,
                       std::string snapshot_prefix) {
  if (!buff_allocated_) {
    HCTR_OWN_THROW(Error_t::IllegalCall,
                   "Cannot start the training process before calling Model.compile()");
  }

  if (solver_.repeat_dataset && max_iter <= 0) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Require max_iter>0 under non-epoch mode");
  }
  if (!solver_.repeat_dataset) {
    HCTR_OWN_THROW(Error_t::WrongInput, "Epoch mode cannot be used with ModelPerfExt");
  }
  if (etc_params_->use_embedding_training_cache) {
    HCTR_OWN_THROW(Error_t::WrongInput, "ETC cannot be used with ModelPerfExt");
  }
  // if (!solver_.is_dlrm) {
  //   HCTR_OWN_THROW(Error_t::WrongInput, "ModelPerfExt can be used only with is_dlrm flag");
  // }

  high_level_eval_ = true;
  int __PID(0);
#ifdef ENABLE_MPI
  HCTR_MPI_THROW(MPI_Comm_rank(MPI_COMM_WORLD, &__PID));
#endif

  HugeCTR::Timer timer;
  HugeCTR::Timer timer_train;
  HugeCTR::Timer timer_eval;

  if (__PID == 0) {
    HCTR_LOG_S(INFO, ROOT) << "=====================================================Model "
                              "Fit====================================================="
                           << std::endl;
  }

  HCTR_LOG(INFO, ROOT, "Use non-epoch mode with number of iterations: %d\n", max_iter);

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

  HCTR_LOG_S(INFO, ROOT) << "Training source file: " << reader_params_.source[0] << std::endl;
  HCTR_LOG_S(INFO, ROOT) << "Evaluation source file: " << reader_params_.eval_source << std::endl;

  if (solver_.is_dlrm) {
    HCTR_LOG_ARGS(timer_log.elapsedMilliseconds(), "epoch_start", 0);
  }

  this->start_data_reading();
  this->init_data_reader_.reset();

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
      float loss = 0.0f;
      if (solver_.gen_loss_summary) {
        this->get_current_loss(&loss);
        if (isnan(loss)) {
          throw std::runtime_error(std::string("Train Runtime error: Loss cannot converge") + " " +
                                   __FILE__ + ":" + std::to_string(__LINE__) + " \n");
        }
      }
      if (!solver_.use_holistic_cuda_graph) {
        HCTR_LOG_S(INFO, ROOT) << "Iter: " << iter << " Time(" << display
                               << " iters): " << timer_train.elapsedSeconds() << "s Loss: " << loss
                               << " lr:" << lr << std::endl;
      } else {
        HCTR_LOG_S(INFO, ROOT) << "Iter: " << iter << " Time(" << display
                               << " iters): " << timer_train.elapsedSeconds() << "s Loss: " << loss
                               << std::endl;
      }
      timer_train.start();
    }
    if (eval_interval > 0 && iter % eval_interval == 0 && iter != 0) {
      if (solver_.all_reduce_algo == AllReduceAlgo::NCCL) {
#pragma omp parallel num_threads(networks_.size())
        {
          size_t id = omp_get_thread_num();
          CudaCPUDeviceContext ctx(resource_manager_->get_local_gpu(id)->get_device_id());
          cudaStreamSynchronize(resource_manager_->get_local_gpu(id)->get_stream());
        }
      }
      this->check_overflow();
      this->copy_weights_for_evaluation();
      is_first_train_batch_after_eval = true;
      timer_eval.start();
      if (solver_.is_dlrm) {
        HCTR_LOG_ARGS(timer_log.elapsedMilliseconds(), "eval_start", float(iter) / max_iter);
      }
      for (int batches = 0; batches < solver_.max_eval_batches; batches++) {
        this->eval(batches == 0);
      }
      auto eval_metrics = this->get_eval_metrics();
      for (auto& eval_metric : eval_metrics) {
        HCTR_LOG_S(INFO, ROOT) << "Evaluation, " << eval_metric.first << ": " << eval_metric.second
                               << std::endl;
        if (solver_.is_dlrm) {
          HCTR_LOG_ARGS(timer_log.elapsedMilliseconds(), "eval_accuracy", eval_metric.second,
                        float(iter) / max_iter, iter);
        }
        if (!eval_metric.first.compare("AUC")) {
          const auto auc_threshold = solver_.metrics_spec[HugeCTR::metrics::Type::AUC];
          if (eval_metric.second > auc_threshold) {
            timer.stop();

            if (solver_.is_dlrm) {
              size_t train_samples =
                  static_cast<size_t>(iter + 1) * static_cast<size_t>(solver_.batchsize);

              HCTR_LOG_S(INFO, WORLD)
                  << "Hit target accuracy AUC " << auc_threshold << " at " << iter << "/"
                  << max_iter << " iterations with batchsize " << solver_.batchsize << " in "
                  << std::setiosflags(std::ios::fixed) << std::setprecision(2)
                  << timer.elapsedSeconds() << " s. Average speed "
                  << (float(iter) * solver_.batchsize / timer.elapsedSeconds()) << " records/s."
                  << std::endl;

              HCTR_LOG_ARGS(timer_log.elapsedMilliseconds(), "eval_stop", float(iter) / max_iter);

              HCTR_LOG_ARGS(timer_log.elapsedMilliseconds(), "epoch_stop", 1);

              if (__PID == 0) {
                HCTR_LOG_ARGS(timer_log.elapsedMilliseconds(), "run_stop");
                HCTR_LOG_ARGS(timer_log.elapsedMilliseconds(), "train_samples", train_samples);
              }
              timer_log.stop();
            }

            if (__PID == 0) {
              HCTR_LOG(INFO, ROOT,
                       "Hit target accuracy AUC %f at %d / %d iterations with batchsize %d"
                       " in %.2fs. Average speed %f records/s.\n",
                       auc_threshold, iter, max_iter, solver_.batchsize, timer.elapsedSeconds(),
                       float(iter) * solver_.batchsize / timer.elapsedSeconds());
            }
            return;
          }
        }
      }
      timer_eval.stop();
      HCTR_LOG_S(INFO, ROOT) << "Eval Time for " << solver_.max_eval_batches
                             << " iters: " << timer_eval.elapsedSeconds() << "s" << std::endl;
      if (solver_.is_dlrm) {
        HCTR_LOG_ARGS(timer_log.elapsedMilliseconds(), "eval_stop",
                      float(iter) / max_iter);  // use iteration to calculate it's in which epoch
      }
    }
    if (snapshot > 0 && iter % snapshot == 0 && iter != 0) {
      this->download_params_to_files(snapshot_prefix, iter);
    }
  }  // end for iter
  if (solver_.is_dlrm) {
    HCTR_LOG_ARGS(timer_log.elapsedMilliseconds(), "epoch_stop", 1);

    if (__PID == 0) {
      HCTR_LOG_ARGS(timer_log.elapsedMilliseconds(), "run_stop");
      size_t train_samples = static_cast<size_t>(max_iter) * static_cast<size_t>(solver_.batchsize);
      HCTR_LOG_ARGS(timer_log.elapsedMilliseconds(), "train_samples", train_samples);
    }
    timer_log.stop();
  }

  timer.stop();
  if (__PID == 0) {
    HCTR_LOG_S(INFO, ROOT) << "Finish "
                           << max_iter + " iterations with batchsize: " << solver_.batchsize
                           << " in " << std::setiosflags(std::ios::fixed) << std::setprecision(2)
                           << timer.elapsedSeconds() << "s" << std::endl;
  }

  high_level_eval_ = false;
}

void ModelPerfExt::train_overlapped() {
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
        HCTR_OWN_THROW(Error_t::InvalidEnv, "model state reached invalid status");
    }
    return true;
  };

  auto scheduled_reader = dynamic_cast<IDataReaderWithScheduling*>(train_data_reader_.get());
  const bool use_graph =
      solver_.use_holistic_cuda_graph && !scheduled_reader->current_batch_incomplete();

#pragma omp parallel num_threads(resource_manager_->get_local_gpu_count())
  {
    size_t id = omp_get_thread_num();
    auto device_id = resource_manager_->get_local_gpu(id)->get_device_id();
    auto stream = resource_manager_->get_local_gpu(id)->get_stream();
    CudaCPUDeviceContext context(device_id);
    long long current_batchsize_per_device =
        train_data_reader_->get_current_batchsize_per_device(id);

    TrainState state;
    auto sync = [&state, &stream, id]() {
      if (state.event) {
        HCTR_LIB_THROW(cudaStreamWaitEvent(stream, *state.event));
      }
      state.event = nullptr;
    };

    auto schedule_reader = [&, this](TrainState_t expected) {
      if (scheduled_reader && state.state == expected) {
        if (use_graph) {
          scheduled_reader->schedule_here_graph(stream, id);
        } else {
          scheduled_reader->schedule_here(stream, id);
        }
        graph_scheduler_->record_execution(id, stream);
      }
    };

    auto schedule_split3way = [&, this](TrainState_t state_to_schedule) {
      if (state.state == state_to_schedule) {
        scheduled_reader->schedule_precompute_here(stream, id, use_graph);
      }
    };

    auto schedule_d2d = [&, this](TrainState_t state_to_schedule) {
      if (state.state == state_to_schedule) {
        scheduled_reader->schedule_d2d_here(stream, id, use_graph);
      }
    };

    auto do_it = [&, this](cudaStream_t submit_stream) {
      if (use_graph || scheduled_reader->precompute_enabled()) {
        HCTR_LIB_THROW(cudaEventRecord(fork_events_[id], submit_stream));
        state.event = &fork_events_[id];
      }

      // Network just runs unconditionally
      // Embedding manages events from the networks and waits if necessary
      // Session inserts a wait if it gets a non-null event from the embedding

      PROFILE_RECORD("iteration.start", submit_stream, true);
      do {
        state = embeddings_[0]->train(true, id, state);
        sync();
        schedule_reader(TrainState_t::TopMLPFprop);
        schedule_split3way(TrainState_t::MLPExchangeWgrad);
        schedule_d2d(TrainState_t::MLPUpdate);
        state = networks_[id]->train(
            current_batchsize_per_device, [this, id]() { this->exchange_wgrad(id); }, state);
      } while (change_state(&state));
      PROFILE_RECORD("iteration.stop", submit_stream, true);
      sync();
    };

    if (use_graph) {
#ifdef ENABLE_PROFILING
      if (!train_graphs_[id].initialized || profiler_init_cuda_graph_this_iter()) {
#else
      if (!train_graphs_[id].initialized) {
#endif
        train_graphs_[id].capture(do_it, stream);
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

void ModelPerfExt::exchange_wgrad(size_t device_id) {
  auto& gpu_resource = resource_manager_->get_local_gpu(device_id);
  CudaCPUDeviceContext context(gpu_resource->get_device_id());
  PROFILE_RECORD("exchange_wgrad.start", resource_manager_->get_local_gpu(device_id)->get_stream(),
                 true, device_id);
  Model::exchange_wgrad(device_id);
  PROFILE_RECORD("exchange_wgrad.stop", resource_manager_->get_local_gpu(device_id)->get_stream(),
                 true, device_id);
}

void ModelPerfExt::add(DenseLayer& dense_layer) {
  for (auto& top_name : dense_layer.top_names) {
    if (tensor_shape_info_raw_.find(top_name) != tensor_shape_info_raw_.end()) {
      HCTR_OWN_THROW(Error_t::WrongInput, top_name + ", top tensor name already exists");
    }
  }
  for (auto& bottom_name : dense_layer.bottom_names) {
    if (tensor_shape_info_raw_.find(bottom_name) == tensor_shape_info_raw_.end()) {
      HCTR_OWN_THROW(Error_t::WrongInput, bottom_name + ", bottom tensor name does not exists");
    }
  }
  calculate_tensor_dimensions(tensor_shape_info_raw_, dense_layer);
  dense_layer_params_raw_.push_back(dense_layer);
}

void ModelPerfExt::add_internal(DenseLayer& dense_layer) {
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

}  // namespace HugeCTR
