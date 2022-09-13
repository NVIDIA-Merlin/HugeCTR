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

#include "HugeCTR/core/hctr_impl/hctr_backend.hpp"
#include "HugeCTR/embedding/embedding_collection.hpp"
#include "HugeCTR/embedding/embedding_planner.hpp"

namespace HugeCTR {

void Model::create_train_network_pipeline() {
  graph_.train_pipeline_.resize(resource_manager_->get_local_gpu_count());

  for (size_t local_id = 0; local_id < resource_manager_->get_local_gpu_count(); local_id++) {
    auto gpu_resource = resource_manager_->get_local_gpu(local_id);
    CudaCPUDeviceContext context(gpu_resource->get_device_id());

    auto network_forward_and_backward = std::make_shared<StreamContextScheduleable>([=] {
      long long current_batchsize_per_device =
          train_data_reader_->get_current_batchsize_per_device(local_id);
      networks_[local_id]->train(current_batchsize_per_device);
    });

    auto async_mlp_syncback = std::make_shared<StreamContextScheduleable>([=] {
      if (solver_.async_mlp_wgrad) gpu_resource->wait_on_wgrad_event(gpu_resource->get_stream());
    });

    graph_.train_pipeline_[local_id] =
        Pipeline{"default", gpu_resource, {network_forward_and_backward, async_mlp_syncback}};
  }
}

void Model::create_eval_network_pipeline() {
  graph_.evaluate_pipeline_.resize(resource_manager_->get_local_gpu_count());

  for (size_t local_id = 0; local_id < resource_manager_->get_local_gpu_count(); local_id++) {
    auto gpu_resource = resource_manager_->get_local_gpu(local_id);
    CudaCPUDeviceContext context(gpu_resource->get_device_id());

    auto network_eval = std::make_shared<StreamContextScheduleable>([=] {
      long long current_batchsize_per_device =
          evaluate_data_reader_->get_current_batchsize_per_device(local_id);
      networks_[local_id]->eval(current_batchsize_per_device);
    });

    auto cal_metrics = std::make_shared<StreamContextScheduleable>([=] {
      for (auto& metric : metrics_) {
        for (auto& loss_metrics : networks_[local_id]->get_raw_metrics_all()) {
          metric->local_reduce(local_id, loss_metrics.second);
        }
      }
    });

    graph_.evaluate_pipeline_[local_id] =
        Pipeline{"default", gpu_resource, {network_eval, cal_metrics}};
  }
}

void Model::create_train_pipeline() {
  auto scheduled_reader = dynamic_cast<SchedulableDataReader*>(train_data_reader_.get());
  auto scheduled_embedding = dynamic_cast<SchedulableEmbeding*>(embeddings_[0].get());
  bool is_train = true;
  bool use_graph = solver_.use_cuda_graph;

  if (solver_.train_inter_iteration_overlap) {
    graph_.train_pipeline_.resize(2 * resource_manager_->get_local_gpu_count());
  } else {
    graph_.train_pipeline_.resize(resource_manager_->get_local_gpu_count());
  }

#pragma omp parallel for num_threads(resource_manager_->get_local_gpu_count())
  for (size_t local_id = 0; local_id < resource_manager_->get_local_gpu_count(); local_id++) {
    auto gpu_resource = resource_manager_->get_local_gpu(local_id);
    CudaCPUDeviceContext context(gpu_resource->get_device_id());

    // create scheduleable
    auto iteration_start = std::make_shared<StreamContextScheduleable>([=] {});

    auto schedule_reader = std::make_shared<StreamContextScheduleable>([=] {
      auto stream = gpu_resource->get_stream();
      if (use_graph && !scheduled_reader->current_batch_incomplete()) {
        scheduled_reader->schedule_here_graph(stream, local_id);
      } else {
        scheduled_reader->schedule_here(stream, local_id);
      }
      graph_scheduler_->record_execution(local_id, stream);
    });

    auto EMB_input_ready_wait = std::make_shared<StreamContextScheduleable>([=] {
      auto stream = gpu_resource->get_stream();
      scheduled_reader->stream_wait_sparse_tensors(
          stream, local_id, use_graph && !scheduled_reader->current_batch_incomplete());
    });

    auto BNET_input_ready_wait = std::make_shared<StreamContextScheduleable>([=] {
      auto stream = gpu_resource->get_stream();
      scheduled_reader->stream_wait_dense_tensors(
          stream, local_id, use_graph && !scheduled_reader->current_batch_incomplete());
    });

    auto schedule_split_3way = std::make_shared<StreamContextScheduleable>([=] {
      auto stream = gpu_resource->get_stream();
      scheduled_reader->schedule_split_3_way_here(
          stream, local_id, use_graph && !scheduled_reader->current_batch_incomplete());
    });

    auto schedule_d2d = std::make_shared<StreamContextScheduleable>([=] {
      auto stream = gpu_resource->get_stream();
      scheduled_reader->schedule_d2d_here(
          stream, local_id, use_graph && !scheduled_reader->current_batch_incomplete());
    });

    auto embedding_index_calculation = std::make_shared<StreamContextScheduleable>(
        [=] { scheduled_embedding->index_calculation(is_train, local_id); });

    auto cross_iteration_sync = std::make_shared<StreamContextScheduleable>([] {});

    auto embedding_freq_forward = std::make_shared<StreamContextScheduleable>(
        [=] { scheduled_embedding->freq_forward(is_train, local_id); });

    auto embedding_freq_backward = std::make_shared<StreamContextScheduleable>(
        [=] { scheduled_embedding->freq_backward(local_id); });

    auto embedding_freq_update_params = std::make_shared<StreamContextScheduleable>(
        [=] { scheduled_embedding->freq_update_params(local_id); });

    auto embedding_infreq_model_forward = std::make_shared<StreamContextScheduleable>(
        [=] { scheduled_embedding->infreq_model_forward(local_id); });

    auto embedding_infreq_network_forward = std::make_shared<StreamContextScheduleable>(
        [=] { scheduled_embedding->infreq_network_forward(is_train, local_id); });

    auto embedding_infreq_network_backward = std::make_shared<StreamContextScheduleable>(
        [=] { scheduled_embedding->infreq_network_backward(local_id); });

    auto embedding_infreq_model_backward = std::make_shared<StreamContextScheduleable>(
        [=] { scheduled_embedding->infreq_model_backward(local_id); });

    auto network_init = std::make_shared<StreamContextScheduleable>([=] {
      if (networks_[local_id]->use_mixed_precision_ &&
          networks_[local_id]->optimizer_->get_optimizer_type() != Optimizer_t::SGD) {
        networks_[local_id]->conv_weight_(networks_[local_id]->train_weight_tensor_half_,
                                          networks_[local_id]->train_weight_tensor_);
      }
    });

    auto bottom_network_fprop = std::make_shared<StreamContextScheduleable>([=] {
      networks_[local_id]->prop_layers(networks_[local_id]->bottom_layers_, true, is_train);
    });

    auto top_network_fprop = std::make_shared<StreamContextScheduleable>([=] {
      networks_[local_id]->prop_layers(networks_[local_id]->top_layers_, true, is_train);
    });

    auto init_wgrad = std::make_shared<StreamContextScheduleable>([=] {
      networks_[local_id]->train_losses_.begin()->second->regularizer_initialize_wgrad(is_train);
    });

    auto lr_sched_update = std::make_shared<StreamContextScheduleable>(
        [=]() { networks_[local_id]->lr_sched_->update(); });

    auto cal_loss = std::make_shared<StreamContextScheduleable>([=] {
      float rterm = networks_[local_id]->train_losses_.begin()->second->regularizer_compute_rterm();
      long long current_batchsize_per_device =
          scheduled_reader->get_current_batchsize_per_device(local_id);

      networks_[local_id]->train_losses_.begin()->second->compute(
          is_train, current_batchsize_per_device, rterm);
    });

    auto top_network_bprop = std::make_shared<StreamContextScheduleable>([=] {
      networks_[local_id]->prop_layers(networks_[local_id]->top_layers_, false, is_train);
    });

    auto bottom_network_bprop = std::make_shared<StreamContextScheduleable>([=] {
      networks_[local_id]->prop_layers(networks_[local_id]->bottom_layers_, false, is_train);
    });

    auto network_exchange_wgrad =
        std::make_shared<StreamContextScheduleable>([=] { this->exchange_wgrad(local_id); });

    auto update_params =
        std::make_shared<StreamContextScheduleable>([=] { networks_[local_id]->update_params(); });

    auto iteration_end = std::make_shared<StreamContextScheduleable>([] {});

    std::vector<std::shared_ptr<Scheduleable>> scheduleable_list = {
        iteration_start,
        EMB_input_ready_wait,
        embedding_index_calculation,
        BNET_input_ready_wait,
        cross_iteration_sync,
        embedding_infreq_model_forward,
        embedding_infreq_network_forward,
        embedding_freq_forward,
        network_init,
        bottom_network_fprop,
        init_wgrad,
        schedule_reader,
        top_network_fprop,
        lr_sched_update,
        cal_loss,
        top_network_bprop,
        embedding_freq_backward,
        bottom_network_bprop,
        embedding_infreq_network_backward,
        embedding_infreq_model_backward,
        schedule_split_3way,
        network_exchange_wgrad,
        schedule_d2d,
        embedding_freq_update_params,
        update_params,
        iteration_end,
    };

    if (solver_.train_intra_iteration_overlap) {
      std::string infreq_stream = "side_stream";
      std::string freq_stream = "freq_stream";
      std::string network_side_stream = "network_side_stream";

      auto done_iteration_start = iteration_start->record_done();
      auto done_cross_iteration_sync = cross_iteration_sync->record_done();
      auto done_embedding_infreq_model_forward = embedding_infreq_model_forward->record_done();
      auto done_embedding_infreq_network_forward = embedding_infreq_network_forward->record_done();
      auto done_embedding_freq_forward = embedding_freq_forward->record_done();
      auto done_bottom_network_fprop = bottom_network_fprop->record_done();
      auto done_top_network_fprop = top_network_fprop->record_done();
      auto done_init_wgrad = init_wgrad->record_done();
      auto done_lr_sched_update = lr_sched_update->record_done();
      auto done_top_network_bprop = top_network_bprop->record_done();
      auto done_embedding_freq_backward = embedding_freq_backward->record_done();
      auto done_bottom_network_bprop = bottom_network_bprop->record_done();
      auto done_network_exchange_wgrad = network_exchange_wgrad->record_done();
      auto done_embedding_infreq_network_backward =
          embedding_infreq_network_backward->record_done();
      auto done_freq_update_params = embedding_freq_update_params->record_done();

      EMB_input_ready_wait->set_stream(infreq_stream);
      EMB_input_ready_wait->wait_event({done_iteration_start});
      embedding_index_calculation->set_stream(infreq_stream);
      cross_iteration_sync->set_stream(infreq_stream);

      embedding_infreq_model_forward->set_stream(infreq_stream);
      embedding_infreq_network_forward->set_stream(infreq_stream);

      const bool overlap_infreq_freq =
          (sparse_embedding_params_[0].hybrid_embedding_param.communication_type !=
           CommunicationType::NVLink_SingleNode);

      if (overlap_infreq_freq) {
        embedding_freq_forward->set_stream(freq_stream);
        embedding_freq_forward->wait_event(
            {done_cross_iteration_sync, done_embedding_infreq_model_forward});
      } else {
        embedding_freq_forward->set_stream(infreq_stream);
      }

      bottom_network_fprop->wait_event({done_embedding_infreq_model_forward});
      schedule_reader->wait_event({
          done_embedding_infreq_network_forward,
          done_embedding_freq_forward,
      });

      init_wgrad->set_stream(network_side_stream);
      init_wgrad->wait_event({done_bottom_network_fprop});

      lr_sched_update->set_stream(network_side_stream);
      lr_sched_update->wait_event({done_top_network_fprop});
      top_network_bprop->wait_event({
          done_init_wgrad,
          done_lr_sched_update,
      });

      embedding_freq_backward->set_stream(infreq_stream);
      embedding_freq_backward->wait_event({done_top_network_bprop});

      network_exchange_wgrad->wait_event({
          done_embedding_freq_backward,
          done_bottom_network_bprop,
      });

      embedding_infreq_network_backward->set_stream(infreq_stream);
      embedding_infreq_network_backward->wait_event({done_top_network_bprop});
      embedding_infreq_model_backward->set_stream(infreq_stream);

      embedding_freq_update_params->set_stream(infreq_stream);
      embedding_freq_update_params->wait_event({done_network_exchange_wgrad});
      iteration_end->wait_event({
          done_embedding_infreq_network_backward,
          done_freq_update_params,
      });
    }

    graph_.train_pipeline_[local_id] = Pipeline{"train", gpu_resource, scheduleable_list};
    if (solver_.train_inter_iteration_overlap) {
      cudaStream_t s3w_stream = gpu_resource->get_stream("s3w");
      cudaStream_t d2d_stream = gpu_resource->get_stream("s3w");
      scheduled_reader->set_schedule_streams(s3w_stream, d2d_stream, local_id);

      auto done_iteration_end = iteration_end->record_done(use_graph);
      cross_iteration_sync->wait_event({done_iteration_end}, use_graph);

      graph_.train_pipeline_[local_id + resource_manager_->get_local_gpu_count()] =
          Pipeline{"train2", gpu_resource, scheduleable_list};
    } else {
      cudaStream_t s3w_stream = gpu_resource->get_stream("train");
      cudaStream_t d2d_stream = gpu_resource->get_stream("train");
      scheduled_reader->set_schedule_streams(s3w_stream, d2d_stream, local_id);
    }
  }
}

void Model::train_pipeline(size_t current_batch_size) {
  auto scheduled_reader = dynamic_cast<SchedulableDataReader*>(train_data_reader_.get());
  auto scheduled_embedding = dynamic_cast<SchedulableEmbeding*>(embeddings_[0].get());

  const auto inflight_id = scheduled_reader->get_current_inflight_id();
  const bool cached = scheduled_reader->is_batch_cached();

  const bool use_graph = solver_.use_cuda_graph && !scheduled_reader->current_batch_incomplete();

  scheduled_embedding->assign_input_tensors(true, current_batch_size, inflight_id, cached);

#pragma omp parallel num_threads(resource_manager_->get_local_gpu_count())
  {
    size_t id = omp_get_thread_num();
    auto device_id = resource_manager_->get_local_gpu(id)->get_device_id();
    CudaCPUDeviceContext context(device_id);

    const auto graph_id = solver_.train_inter_iteration_overlap
                              ? (inflight_id * resource_manager_->get_local_gpu_count() + id)
                              : id;
    HCTR_CHECK_HINT(graph_id < graph_.train_pipeline_.size(), "graph_id out of range");

    if (use_graph && !scheduled_reader->current_batch_incomplete()) {
      graph_.train_pipeline_[graph_id].run_graph();
      if (scheduled_reader) {
        scheduled_reader->update_schedule_graph(id);
      }
    } else {
      graph_.train_pipeline_[graph_id].run();
    }
    cudaStream_t graph_stream = resource_manager_->get_local_gpu(id)->get_stream(
        graph_.train_pipeline_[graph_id].get_stream_name());

    auto train_sync_back_event =
        resource_manager_->get_local_gpu(id)->get_event("train_sync_back_event");
    HCTR_LIB_THROW(cudaEventRecord(train_sync_back_event, graph_stream));
    HCTR_LIB_THROW(cudaStreamWaitEvent(resource_manager_->get_local_gpu(id)->get_stream(),
                                       train_sync_back_event));
  }
}

void Model::create_evaluate_pipeline() {
  auto scheduled_reader = dynamic_cast<SchedulableDataReader*>(evaluate_data_reader_.get());
  auto scheduled_embedding = dynamic_cast<SchedulableEmbeding*>(embeddings_[0].get());
  bool is_train = false;

  graph_.evaluate_pipeline_.resize(resource_manager_->get_local_gpu_count());

  for (size_t local_id = 0; local_id < resource_manager_->get_local_gpu_count(); local_id++) {
    auto gpu_resource = resource_manager_->get_local_gpu(local_id);
    CudaCPUDeviceContext ctx(gpu_resource->get_device_id());

    // create scheduleable
    auto iteration_strat = std::make_shared<StreamContextScheduleable>([] {});

    auto EMB_input_ready_wait = std::make_shared<StreamContextScheduleable>([=] {
      auto stream = gpu_resource->get_stream();
      scheduled_reader->stream_wait_sparse_tensors(stream, local_id, false);
    });

    auto BNET_input_ready_wait = std::make_shared<StreamContextScheduleable>([=] {
      auto stream = gpu_resource->get_stream();
      scheduled_reader->stream_wait_dense_tensors(stream, local_id, false);
    });

    auto embedding_index_calculation = std::make_shared<StreamContextScheduleable>(
        [=] { scheduled_embedding->index_calculation(is_train, local_id); });

    auto embedding_freq_forward = std::make_shared<StreamContextScheduleable>([=] {
      scheduled_embedding->freq_forward(is_train, local_id, this->graph_.is_first_eval_batch_);
    });

    auto embedding_infreq_model_forward = std::make_shared<StreamContextScheduleable>(
        [=] { scheduled_embedding->infreq_model_forward(local_id); });

    auto embedding_infreq_network_forward = std::make_shared<StreamContextScheduleable>(
        [=] { scheduled_embedding->infreq_network_forward(is_train, local_id); });

    auto embedding_global_barrier = std::make_shared<StreamContextScheduleable>(
        [=] { scheduled_embedding->global_barrier(is_train, local_id); });

    auto network_init = std::make_shared<StreamContextScheduleable>([=] {
      if (networks_[local_id]->use_mixed_precision_ &&
          networks_[local_id]->optimizer_->get_optimizer_type() != Optimizer_t::SGD) {
        networks_[local_id]->conv_weight_(networks_[local_id]->train_weight_tensor_half_,
                                          networks_[local_id]->train_weight_tensor_);
      }
    });

    auto network_eval = std::make_shared<StreamContextScheduleable>([=] {
      long long current_batchsize_per_device =
          scheduled_reader->get_current_batchsize_per_device(local_id);

      networks_[local_id]->eval(current_batchsize_per_device);
    });

    auto cal_metrics = std::make_shared<StreamContextScheduleable>([=] {
      for (auto& metric : metrics_) {
        for (auto& loss_metrics : networks_[local_id]->get_raw_metrics_all()) {
          metric->local_reduce(local_id, loss_metrics.second);
        }
      }
    });

    std::vector<std::shared_ptr<Scheduleable>> scheduleable_list = {
        BNET_input_ready_wait,
        EMB_input_ready_wait,
        embedding_index_calculation,
        embedding_infreq_model_forward,
        embedding_infreq_network_forward,
        embedding_freq_forward,
        embedding_global_barrier,
        network_init,
        network_eval,
        cal_metrics,
    };

    const bool overlap_infreq_freq =
        (sparse_embedding_params_[0].hybrid_embedding_param.communication_type !=
         CommunicationType::NVLink_SingleNode) &&
        solver_.eval_intra_iteration_overlap;
    std::string eval_embedding = "eval_embedding";
    std::string eval_freq = "eval_freq";

    if (solver_.eval_inter_iteration_overlap) {
      // s3w_stream should be the same with embedding stream
      cudaStream_t s3w_stream = gpu_resource->get_stream(eval_embedding);
      cudaStream_t d2d_stream = gpu_resource->get_stream("default");
      scheduled_reader->set_schedule_streams(s3w_stream, d2d_stream, local_id);

      auto done_embedding_infreq_model_forward = embedding_infreq_model_forward->record_done();
      auto done_embedding_infreq_network_forward = embedding_infreq_network_forward->record_done();
      auto done_embedding_freq_forward = embedding_freq_forward->record_done();
      auto done_network_eval = network_eval->record_done();

      EMB_input_ready_wait->set_absolute_stream(eval_embedding);
      embedding_index_calculation->set_absolute_stream(eval_embedding);
      embedding_infreq_model_forward->set_absolute_stream(eval_embedding);
      embedding_infreq_network_forward->set_absolute_stream(eval_embedding);
      embedding_infreq_network_forward->wait_event({done_network_eval});

      if (overlap_infreq_freq) {
        embedding_freq_forward->set_stream(eval_freq);
        embedding_freq_forward->wait_event(
            {done_embedding_infreq_model_forward, done_network_eval});
      } else {
        embedding_freq_forward->set_absolute_stream(eval_embedding);
      }
      embedding_global_barrier->set_absolute_stream(eval_embedding);

      network_init->wait_event(
          {done_embedding_infreq_network_forward, done_embedding_freq_forward});
    } else if (overlap_infreq_freq) {
      auto done_embedding_infreq_model_forward = embedding_infreq_model_forward->record_done();
      auto done_embedding_freq_forward = embedding_freq_forward->record_done();

      embedding_freq_forward->set_stream(eval_freq);
      embedding_freq_forward->wait_event({done_embedding_infreq_model_forward});
      network_init->wait_event({done_embedding_freq_forward});
    }

    graph_.evaluate_pipeline_[local_id] = Pipeline{"default", gpu_resource, scheduleable_list};
  }
}

void Model::evaluate_pipeline(size_t current_batch_size) {
  auto scheduled_reader = dynamic_cast<SchedulableDataReader*>(evaluate_data_reader_.get());
  auto scheduled_embedding = dynamic_cast<SchedulableEmbeding*>(embeddings_[0].get());

  const auto inflight_id = scheduled_reader->get_current_inflight_id();
  const bool cached = scheduled_reader->is_batch_cached();

  scheduled_embedding->assign_input_tensors(false, current_batch_size, inflight_id, cached);

#pragma omp parallel num_threads(networks_.size())
  {
    size_t id = omp_get_thread_num();
    auto gpu = resource_manager_->get_local_gpu(id);
    CudaCPUDeviceContext ctx(gpu->get_device_id());

    if (graph_.is_first_eval_batch_) {
      auto eval_start_event = gpu->get_event("eval_start_event");
      HCTR_LIB_THROW(cudaEventRecord(eval_start_event, gpu->get_stream()));

      cudaStream_t evaluate_stream =
          gpu->get_stream(graph_.evaluate_pipeline_[id].get_stream_name());
      HCTR_LIB_THROW(cudaStreamWaitEvent(evaluate_stream, eval_start_event));
      cudaStream_t eval_embedding_stream = gpu->get_stream("eval_embedding");
      HCTR_LIB_THROW(cudaStreamWaitEvent(eval_embedding_stream, eval_start_event));
    }

    graph_.evaluate_pipeline_[id].run();
  }

  for (auto& metric : metrics_) {
    metric->global_reduce(networks_.size());
  }
}
}  // namespace HugeCTR
