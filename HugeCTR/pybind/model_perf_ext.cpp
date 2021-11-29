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

#include <HugeCTR/pybind/model_perf_ext.hpp>
#include <data_readers/async_reader/data_reader_scheduling.hpp>


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

} // Anonymous namespace


ModelPerfExt::ModelPerfExt(const Solver& solver, const DataReaderParams& reader_params,
                           std::shared_ptr<OptParamsPy>& opt_params_py,
                           std::shared_ptr<EmbeddingTrainingCacheParams>& etc_params)
    : Model(solver, reader_params, opt_params_py, etc_params) {
      graph_scheduler_ = std::make_unique<GraphScheduler>(resource_manager_);
    }

bool ModelPerfExt::train(bool is_first_batch) {
  try {
    if (train_data_reader_->is_started() == false) {
      CK_THROW_(Error_t::IllegalCall, "Start the data reader first before calling Model::train()");
    }
    graph_scheduler_->trickling();
    // When async indices is enabled, we prefetch next batch on internal stream, that stream will block until
    // the schedule event in the graph is executed
    train_data_reader_->read_a_batch_to_device_delay_release();
    train_data_reader_->ready_to_collect();

#ifdef ENABLE_PROFILING
    global_profiler.iter_check();
#endif

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
    std::cerr << err.what() << std::endl;
    throw err;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw err;
  }
}

bool ModelPerfExt::eval(bool is_first_batch) {
  try {
    if (evaluate_data_reader_ == nullptr) return true;
    if (evaluate_data_reader_->is_started() == false) {
      CK_THROW_(Error_t::IllegalCall, "Start the data reader first before calling Model::eval()");
    }
    if (!high_level_eval_) {
      this->check_overflow();
      this->copy_weights_for_evaluation();
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
      networks_[id]->eval(current_batchsize_per_device);
      for (auto& metric : metrics_) {
        metric->local_reduce(id, networks_[id]->get_raw_metrics());
      }
    }

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

void ModelPerfExt::fit(int num_epochs, int max_iter, int display, int eval_interval, int snapshot,
                       std::string snapshot_prefix) {
  if (!buff_allocated_) {
    CK_THROW_(Error_t::IllegalCall,
              "Cannot start the training process before calling Model.compile()");
  }

  if (solver_.repeat_dataset && max_iter <= 0) {
    CK_THROW_(Error_t::WrongInput, "Require max_iter>0 under non-epoch mode");
  }
  if (!solver_.repeat_dataset) {
    CK_THROW_(Error_t::WrongInput, "Epoch mode cannot be used with ModelPerfExt");
  }
  if (etc_params_->use_embedding_training_cache) {
    CK_THROW_(Error_t::WrongInput, "ETC cannot be used with ModelPerfExt");
  }
  // if (!solver_.is_dlrm) {
  //   CK_THROW_(Error_t::WrongInput, "ModelPerfExt can be used only with is_dlrm flag");
  // }

  high_level_eval_ = true;
  int __PID(0);
#ifdef ENABLE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &__PID);
#endif

  HugeCTR::Timer timer;
  HugeCTR::Timer timer_train;
  HugeCTR::Timer timer_eval;

  if (__PID == 0) {
    std::cout << "=====================================================Model "
                 "Fit====================================================="
              << std::endl;
  }

  MESSAGE_("Use non-epoch mode with number of iterations: " + std::to_string(max_iter));

  MESSAGE_("Training batchsize: " + std::to_string(solver_.batchsize) +
           ", evaluation batchsize: " + std::to_string(solver_.batchsize_eval));
  MESSAGE_("Evaluation interval: " + std::to_string(eval_interval) +
           ", snapshot interval: " + std::to_string(snapshot));
  MESSAGE_("Sparse embedding trainable: " + std::to_string(is_embedding_trainable_) +
           ", dense network trainable: " + std::to_string(is_dense_trainable_));
  MESSAGE_("Use mixed precision: " + std::to_string(solver_.use_mixed_precision) +
           ", scaler: " + std::to_string(solver_.scaler) +
           ", use cuda graph: " + std::to_string(solver_.use_cuda_graph));
  MESSAGE_("lr: " + std::to_string(solver_.lr) +
           ", warmup_steps: " + std::to_string(solver_.warmup_steps) +
           ", decay_start: " + std::to_string(solver_.decay_start) +
           ", decay_steps: " + std::to_string(solver_.decay_steps) + ", decay_power: " +
           std::to_string(solver_.decay_power) + ", end_lr: " + std::to_string(solver_.end_lr));

  timer.start();
  timer_train.start();

#ifdef ENABLE_PROFILING
  HugeCTR::global_profiler.initialize(solver_.use_cuda_graph);
#endif

  MESSAGE_("Training source file: " + reader_params_.source[0]);
  MESSAGE_("Evaluation source file: " + reader_params_.eval_source);

  if (solver_.is_dlrm) {
    LOG(timer_log.elapsedMilliseconds(), "train_epoch_start", 0);  // just 1 epoch. dlrm logger
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
      // #pragma omp parallel num_threads(networks_.size())
      // {
      //   size_t id = omp_get_thread_num();
      //   CudaCPUDeviceContext ctx(resource_manager_->get_local_gpu(id)->get_device_id());
      //   cudaStreamSynchronize(resource_manager_->get_local_gpu(id)->get_stream());
      // }
      this->check_overflow();
      this->copy_weights_for_evaluation();
      is_first_train_batch_after_eval = true;
      timer_eval.start();
      if (solver_.is_dlrm) {
        LOG(timer_log.elapsedMilliseconds(), "eval_start", float(iter) / max_iter);
      }
      for (int batches = 0; batches < solver_.max_eval_batches; batches++) {
        this->eval(batches == 0);
      }
      auto eval_metrics = this->get_eval_metrics();
      for (auto& eval_metric : eval_metrics) {
        MESSAGE_("Evaluation, " + eval_metric.first + ": " + std::to_string(eval_metric.second));
        if (solver_.is_dlrm) {
          LOG(timer_log.elapsedMilliseconds(), "eval_accuracy", eval_metric.second,
              float(iter) / max_iter, iter);
        }
        if (!eval_metric.first.compare("AUC")) {
          const auto auc_threshold = solver_.metrics_spec[HugeCTR::metrics::Type::AUC];
          if (eval_metric.second >= auc_threshold) {
            timer.stop();

            if (solver_.is_dlrm) {
              size_t train_samples =
                  static_cast<size_t>(iter + 1) * static_cast<size_t>(solver_.batchsize);

              std::string epoch_num_str = std::to_string(float(iter) / max_iter);

              std::cout << "Hit target accuracy AUC " + std::to_string(auc_threshold) + " at " +
                               std::to_string(iter) + "/" + std::to_string(max_iter) +
                               " iterations with batchsize "
                        << solver_.batchsize << " in " << std::setiosflags(std::ios::fixed)
                        << std::setprecision(2) << timer.elapsedSeconds() << " s. Average speed "
                        << float(iter) * solver_.batchsize / timer.elapsedSeconds() << " records/s."
                        << std::endl;

              LOG(timer_log.elapsedMilliseconds(), "eval_stop" + epoch_num_str);

              LOG(timer_log.elapsedMilliseconds(), "train_epoch_end", 1);

              if (__PID == 0) {
                LOG(timer_log.elapsedMilliseconds(), "run_stop");
                LOG(timer_log.elapsedMilliseconds(), "train_samples", train_samples);
              }
              timer_log.stop();
            }

            if (__PID == 0) {
              HCTR_LOG(INFO, ROOT, "Hit target accuracy AUC %f at %d / %d iterations with batchsize %d"
                                   " in %.2fs. Average speed %f records/s.\n",
                                   auc_threshold, iter, max_iter,
                                   solver_.batchsize, timer.elapsedSeconds(),
                                   float(iter) * solver_.batchsize / timer.elapsedSeconds());
            }
            return;
          }
        }
      }
      timer_eval.stop();
      MESSAGE_("Eval Time for " + std::to_string(solver_.max_eval_batches) +
               " iters: " + std::to_string(timer_eval.elapsedSeconds()) + "s");
      if (solver_.is_dlrm) {
        LOG(timer_log.elapsedMilliseconds(), "eval_stop",
            float(iter) / max_iter);  // use iteration to calculate it's in which epoch
      }
    }
    if (snapshot > 0 && iter % snapshot == 0 && iter != 0) {
      this->download_params_to_files(snapshot_prefix, iter);
    }
  }  // end for iter
  if (solver_.is_dlrm) {
    LOG(timer_log.elapsedMilliseconds(), "train_epoch_end", 1);

    if (__PID == 0) {
      LOG(timer_log.elapsedMilliseconds(), "run_stop");
      size_t train_samples = static_cast<size_t>(max_iter) * static_cast<size_t>(solver_.batchsize);
      LOG(timer_log.elapsedMilliseconds(), "train_samples", train_samples);
    }
    timer_log.stop();
  }

  timer.stop();
  if (__PID == 0) {
    std::cout << "Finish "
              << std::to_string(max_iter) + " iterations with batchsize: " << solver_.batchsize
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

      PROFILE_RECORD("iteration.start", submit_stream, true);
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
      PROFILE_RECORD("iteration.stop", submit_stream, true);
      sync();
    };

    if (solver_.use_holistic_cuda_graph) {
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
  PROFILE_RECORD("exchange_wgrad.start", resource_manager_->get_local_gpu(device_id)->get_stream(), true, device_id);
  if (solver_.async_mlp_wgrad)
    gpu_resource->wait_on_wgrad_event(gpu_resource->get_stream());
  Model::exchange_wgrad(device_id);
  PROFILE_RECORD("exchange_wgrad.stop", resource_manager_->get_local_gpu(device_id)->get_stream(), true, device_id);

}

void ModelPerfExt::add(DenseLayer& dense_layer) {
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
