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

#include <HugeCTR/include/data_readers/multi_hot/async_data_reader.hpp>
#include <algorithm>
#include <core23/logger.hpp>
#include <core23_network.hpp>
#include <data_readers/async_reader/async_reader_adapter.hpp>
#include <embeddings/hybrid_sparse_embedding.hpp>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <pybind/model.hpp>
#include <resource_managers/resource_manager_ext.hpp>
#include <sstream>

namespace HugeCTR {

template <typename NetworkType>
void Model::create_train_network_pipeline(std::vector<std::shared_ptr<NetworkType>>& networks) {
  graph_.train_pipeline_.resize(resource_manager_->get_local_gpu_count());

  auto scheduled_reader = dynamic_cast<SchedulableDataReader*>(train_data_reader_.get());

  for (int local_id = 0; local_id < resource_manager_->get_local_gpu_count(); local_id++) {
    auto gpu_resource = resource_manager_->get_local_gpu(local_id);
    CudaCPUDeviceContext context(gpu_resource->get_device_id());

    auto BNET_input_ready_wait = std::make_shared<StreamContextScheduleable>([=] {
      auto stream = gpu_resource->get_stream();
      scheduled_reader->stream_wait_dense_tensors(
          stream, local_id,
          solver_.use_cuda_graph && !scheduled_reader->current_batch_incomplete());
    });

    auto network_forward_and_backward = std::make_shared<StreamContextScheduleable>([=] {
      long long current_batchsize_per_device =
          train_data_reader_->get_current_batchsize_per_device(local_id);
      networks[local_id]->train(current_batchsize_per_device);
    });

    std::vector<std::shared_ptr<Scheduleable>> scheduleable_list;
    if (scheduled_reader) {
      scheduleable_list = {BNET_input_ready_wait, network_forward_and_backward};
    } else {
      scheduleable_list = {network_forward_and_backward};
    }

    auto graph = std::make_shared<GraphScheduleable>(scheduleable_list);
    graph_.train_pipeline_[local_id] = Pipeline{"default", gpu_resource, {graph}};
  }
}

template <typename NetworkType>
void Model::create_eval_network_pipeline(std::vector<std::shared_ptr<NetworkType>>& networks) {
  graph_.evaluate_pipeline_.resize(resource_manager_->get_local_gpu_count());

  for (int local_id = 0; local_id < static_cast<int>(resource_manager_->get_local_gpu_count());
       local_id++) {
    auto gpu_resource = resource_manager_->get_local_gpu(local_id);
    CudaCPUDeviceContext context(gpu_resource->get_device_id());

    auto network_eval = std::make_shared<StreamContextScheduleable>([=] {
      long long current_batchsize_per_device =
          evaluate_data_reader_->get_current_batchsize_per_device(local_id);
      networks[local_id]->eval(current_batchsize_per_device);
    });

    auto cal_metrics = std::make_shared<StreamContextScheduleable>([=] {
      for (auto& metric : metrics_) {
        auto metric_map = networks[local_id]->get_raw_metrics_all().begin()->second;
        metric->local_reduce(local_id, metric_map);
      }
    });

    std::vector<std::shared_ptr<Scheduleable>> scheduleable_list = {network_eval, cal_metrics};

    auto graph = std::make_shared<GraphScheduleable>(scheduleable_list);
    graph_.evaluate_pipeline_[local_id] = Pipeline{"default", gpu_resource, {graph}};
  }
}

template <typename NetworkType>
void Model::create_train_pipeline(std::vector<std::shared_ptr<NetworkType>>& networks) {
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
  for (int local_id = 0; local_id < static_cast<int>(resource_manager_->get_local_gpu_count());
       local_id++) {
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
      if (networks[local_id]->use_mixed_precision_ &&
          networks[local_id]->optimizer_->get_optimizer_type() != Optimizer_t::SGD) {
        networks[local_id]->conv_weight_(networks[local_id]->train_weight_tensor_half_,
                                         networks[local_id]->train_weight_tensor_);
      }
    });

    auto bottom_network_fprop = std::make_shared<StreamContextScheduleable>([=] {
      networks[local_id]->prop_layers(networks[local_id]->bottom_layers_, true, is_train);
    });

    auto top_network_fprop = std::make_shared<StreamContextScheduleable>(
        [=] { networks[local_id]->prop_layers(networks[local_id]->top_layers_, true, is_train); });

    auto init_wgrad = std::make_shared<StreamContextScheduleable>([=] {
      networks[local_id]->train_losses_.begin()->second->regularizer_initialize_wgrad(is_train);
    });

    auto lr_sched_update = std::make_shared<StreamContextScheduleable>(
        [=]() { networks[local_id]->lr_sched_->update(); });

    auto cal_loss = std::make_shared<StreamContextScheduleable>([=] {
      float rterm = networks[local_id]->train_losses_.begin()->second->regularizer_compute_rterm();
      long long current_batchsize_per_device =
          scheduled_reader->get_current_batchsize_per_device(local_id);

      networks[local_id]->train_losses_.begin()->second->compute(
          is_train, current_batchsize_per_device, rterm);
    });

    auto top_network_bprop = std::make_shared<StreamContextScheduleable>(
        [=] { networks[local_id]->prop_layers(networks[local_id]->top_layers_, false, is_train); });

    auto bottom_network_bprop = std::make_shared<StreamContextScheduleable>([=] {
      networks[local_id]->prop_layers(networks[local_id]->bottom_layers_, false, is_train);
    });

    auto network_exchange_wgrad =
        std::make_shared<StreamContextScheduleable>([=] { this->exchange_wgrad(local_id); });

    auto update_params =
        std::make_shared<StreamContextScheduleable>([=] { networks[local_id]->update_params(); });

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
           hybrid_embedding::CommunicationType::NVLink_SingleNode);

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

    auto graph = std::make_shared<GraphScheduleable>(scheduleable_list);
    graph_.train_pipeline_[local_id] = Pipeline{"train", gpu_resource, {graph}};
    if (solver_.train_inter_iteration_overlap) {
      cudaStream_t s3w_stream = gpu_resource->get_stream("s3w");
      cudaStream_t d2d_stream = gpu_resource->get_stream("s3w");
      scheduled_reader->set_schedule_streams(s3w_stream, d2d_stream, local_id);

      auto done_iteration_end = iteration_end->record_done(use_graph);
      cross_iteration_sync->wait_event({done_iteration_end}, use_graph);

      auto graph2 = std::make_shared<GraphScheduleable>(scheduleable_list);
      graph_.train_pipeline_[local_id + resource_manager_->get_local_gpu_count()] =
          Pipeline{"train2", gpu_resource, {graph2}};
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
    int id = omp_get_thread_num();
    auto device_id = resource_manager_->get_local_gpu(id)->get_device_id();
    CudaCPUDeviceContext context(device_id);

    const auto graph_id = solver_.train_inter_iteration_overlap
                              ? (inflight_id * resource_manager_->get_local_gpu_count() + id)
                              : id;
    HCTR_CHECK_HINT(graph_id < graph_.train_pipeline_.size(), "graph_id out of range");

    if (use_graph) {
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

template <typename NetworkType>
void Model::create_evaluate_pipeline(std::vector<std::shared_ptr<NetworkType>>& networks) {
  auto scheduled_reader = dynamic_cast<SchedulableDataReader*>(evaluate_data_reader_.get());
  auto scheduled_embedding = dynamic_cast<SchedulableEmbeding*>(embeddings_[0].get());
  bool is_train = false;

  graph_.evaluate_pipeline_.resize(resource_manager_->get_local_gpu_count());

  for (int local_id = 0; local_id < resource_manager_->get_local_gpu_count(); local_id++) {
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
      if (networks[local_id]->use_mixed_precision_ &&
          networks[local_id]->optimizer_->get_optimizer_type() != Optimizer_t::SGD) {
        networks[local_id]->conv_weight_(networks[local_id]->train_weight_tensor_half_,
                                         networks[local_id]->train_weight_tensor_);
      }
    });

    auto network_eval = std::make_shared<StreamContextScheduleable>([=] {
      long long current_batchsize_per_device =
          scheduled_reader->get_current_batchsize_per_device(local_id);

      networks[local_id]->eval(current_batchsize_per_device);
    });

    auto cal_metrics = std::make_shared<StreamContextScheduleable>([=] {
      for (auto& metric : metrics_) {
        auto metric_map = networks[local_id]->get_raw_metrics_all().begin()->second;
        metric->local_reduce(local_id, metric_map);
      }
    });

    std::vector<std::shared_ptr<Scheduleable>> scheduleable_list = {
        iteration_strat,
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
         hybrid_embedding::CommunicationType::NVLink_SingleNode) &&
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

    auto graph = std::make_shared<GraphScheduleable>(scheduleable_list);
    graph_.evaluate_pipeline_[local_id] = Pipeline{"default", gpu_resource, {graph}};
  }
}

void Model::evaluate_pipeline(size_t current_batch_size) {
  auto scheduled_reader = dynamic_cast<SchedulableDataReader*>(evaluate_data_reader_.get());
  auto scheduled_embedding = dynamic_cast<SchedulableEmbeding*>(embeddings_[0].get());

  const auto inflight_id = scheduled_reader->get_current_inflight_id();
  const bool cached = scheduled_reader->is_batch_cached();

  scheduled_embedding->assign_input_tensors(false, current_batch_size, inflight_id, cached);

#pragma omp parallel num_threads(number_of_networks())
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
    metric->global_reduce(number_of_networks());
  }
}
bool is_first_data_distributor = true;

template <typename NetworkType>
void Model::create_train_pipeline_with_ebc(std::vector<std::shared_ptr<NetworkType>>& networks) {
  bool is_train = true;
  bool use_graph = solver_.use_cuda_graph;

  graph_.train_pipeline_.resize(resource_manager_->get_local_gpu_count());

#pragma omp parallel for num_threads(resource_manager_->get_local_gpu_count())
  for (int local_id = 0; local_id < static_cast<int>(resource_manager_->get_local_gpu_count());
       local_id++) {
    auto gpu_resource = resource_manager_->get_local_gpu(local_id);
    CudaCPUDeviceContext context(gpu_resource->get_device_id());

    // create scheduleable
    auto distribute_data = std::make_shared<StreamContextScheduleable>([=] {
      if (skip_prefetch_in_last_batch(is_train)) return;

      const char* const skip_data_distributor_env = std::getenv("SKIP_DATA_DISTRIBUTOR");
      bool skip_data_distributor =
          (skip_data_distributor_env != nullptr && 1 == std::atoi(skip_data_distributor_env));

      if (is_scheduled_datareader()) {
        if (skip_data_distributor && !is_first_data_distributor) {
          auto stream = gpu_resource->get_stream();
          HCTR_LIB_THROW(cudaStreamSynchronize(stream));
          return;
        }
        if (auto reader =
                dynamic_cast<MultiHot::AsyncDataReader<uint32_t>*>(train_data_reader_.get())) {
          train_data_distributor_->distribute(
              local_id, reader->get_current_sparse_values()[local_id], {},
              train_ddl_output_[local_id], train_data_reader_->get_current_batchsize());
        } else if (auto reader = dynamic_cast<MultiHot::AsyncDataReader<long long>*>(
                       train_data_reader_.get())) {
          train_data_distributor_->distribute(
              local_id, reader->get_current_sparse_values()[local_id], {},
              train_ddl_output_[local_id], train_data_reader_->get_current_batchsize());
        }
        is_first_data_distributor = false;
      } else {
        HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput,
                       "embedding collection can only be used with AsyncMultiHot DataReader.");
      }
    });

    const char* const skip_embedding_env = std::getenv("SKIP_EMBEDDING");
    bool skip_embedding = (skip_embedding_env != nullptr && 1 == std::atoi(skip_embedding_env));

    DataDistributor::Result& ddl_output = solver_.train_inter_iteration_overlap
                                              ? cache_train_ddl_output_[local_id]
                                              : train_ddl_output_[local_id];

    auto ebc_forward = [=, &ddl_output](embedding::Stage stage) {
      if (skip_embedding) return;

      for (auto& ebc : ebc_list_) {
        ebc->forward_per_gpu(stage, is_train, local_id, ddl_output, train_ebc_outptut_[local_id],
                             train_data_reader_->get_full_batchsize());
      }
    };

    auto ebc_backward = [=, &ddl_output](embedding::Stage stage) {
      if (skip_embedding) return;

      for (auto& ebc : ebc_list_) {
        ebc->backward_per_gpu(stage, local_id, ddl_output, train_ebc_outptut_[local_id],
                              train_data_reader_->get_full_batchsize());
      }
    };

    auto ebc_mp_model_forward = std::make_shared<StreamContextScheduleable>([=] {
      ebc_forward(embedding::Stage::MPModelForward);
      ebc_forward(embedding::Stage::HierMPModelForward);
      ebc_forward(embedding::Stage::DenseMPModelForward);
    });

    auto ebc_mp_network_forward = std::make_shared<StreamContextScheduleable>([=] {
      ebc_forward(embedding::Stage::MPNetworkdForward);
      ebc_forward(embedding::Stage::HierMPNetworkForward);
      ebc_forward(embedding::Stage::DenseMPNetworkForward);
    });

    auto ebc_mp_network_backward = std::make_shared<StreamContextScheduleable>([=]() {
      ebc_backward(embedding::Stage::MPNetworkBackward);
      ebc_backward(embedding::Stage::HierMPNetworkBackward);
      ebc_backward(embedding::Stage::DenseMPNetworkBackward);
    });

    auto ebc_mp_backward_index_calculation = std::make_shared<StreamContextScheduleable>([=] {
      ebc_backward(embedding::Stage::MPBackwardIndexCalculation);
      ebc_backward(embedding::Stage::HierMPBackwardIndexCalculation);
      ebc_backward(embedding::Stage::DenseMPBackwardIndexCalculation);
    });

    auto ebc_mp_local_reduce = std::make_shared<StreamContextScheduleable>([=]() {
      ebc_backward(embedding::Stage::HierMPLocalReduce);
      ebc_backward(embedding::Stage::MPLocalReduce);
      ebc_backward(embedding::Stage::DenseMPLocalReduce);
    });

    auto ebc_mp_update = std::make_shared<StreamContextScheduleable>([=]() {
      if (skip_embedding) return;

      for (auto& ebc : ebc_list_) {
        ebc->update_per_gpu(local_id, embedding::EmbeddingGroupType::SparseModelParallel);
        ebc->update_per_gpu(local_id, embedding::EmbeddingGroupType::DenseModelParallel);
      }
    });

    auto ebc_dp_forward = std::make_shared<StreamContextScheduleable>(
        [=] { ebc_forward(embedding::Stage::DPForward); });

    auto ebc_dp_backward_index_calculation = std::make_shared<StreamContextScheduleable>(
        [=] { ebc_backward(embedding::Stage::DPBackwardIndexCalculation); });

    auto ebc_dp_local_reduce = std::make_shared<StreamContextScheduleable>(
        [=]() { ebc_backward(embedding::Stage::DPLocalReduce); });

    auto ebc_dp_allreduce = std::make_shared<StreamContextScheduleable>(
        [=]() { ebc_backward(embedding::Stage::DPAllreduce); });

    auto ebc_dp_update = std::make_shared<StreamContextScheduleable>([=]() {
      if (skip_embedding) return;

      for (auto& ebc : ebc_list_) {
        ebc->update_per_gpu(local_id, embedding::EmbeddingGroupType::DataParallel);
      }
    });

    const char* const skip_bottom_mlp_env = std::getenv("SKIP_BOTTOM_MLP");
    bool skip_bottom_mlp = (skip_bottom_mlp_env != nullptr && 1 == std::atoi(skip_bottom_mlp_env));

    const char* const skip_top_mlp_env = std::getenv("SKIP_TOP_MLP");
    bool skip_top_mlp = (skip_top_mlp_env != nullptr && 1 == std::atoi(skip_top_mlp_env));

    auto network_init = std::make_shared<StreamContextScheduleable>([=] {
      if (skip_bottom_mlp) return;

      if (networks[local_id]->use_mixed_precision_ &&
          networks[local_id]->optimizer_->get_optimizer_type() != Optimizer_t::SGD) {
        networks[local_id]->conv_weight_(networks[local_id]->train_weight_tensor_half_,
                                         networks[local_id]->train_weight_tensor_);
      }
    });

    auto bottom_network_fprop = std::make_shared<StreamContextScheduleable>([=] {
      if (skip_bottom_mlp) return;
      networks[local_id]->prop_layers(networks[local_id]->bottom_layers_, true, is_train);
    });

    auto top_network_fprop = std::make_shared<StreamContextScheduleable>([=] {
      if (skip_top_mlp) return;
      networks[local_id]->prop_layers(networks[local_id]->top_layers_, true, is_train);
    });

    auto init_wgrad = std::make_shared<StreamContextScheduleable>([=] {
      networks[local_id]->train_losses_.begin()->second->regularizer_initialize_wgrad(is_train);
    });

    auto cal_loss = std::make_shared<StreamContextScheduleable>([=] {
      float rterm = networks[local_id]->train_losses_.begin()->second->regularizer_compute_rterm();
      long long current_batchsize_per_device =
          graph_.is_last_train_batch_
              ? train_data_reader_->get_current_batchsize_per_device(local_id)
              : train_data_reader_->get_full_batchsize() /
                    resource_manager_->get_global_gpu_count();

      networks[local_id]->train_losses_.begin()->second->compute(
          is_train, current_batchsize_per_device, rterm);
    });

    auto top_network_bprop = std::make_shared<StreamContextScheduleable>([=] {
      if (skip_top_mlp) return;
      networks[local_id]->prop_layers(networks[local_id]->top_layers_, false, is_train);
    });

    auto bottom_network_bprop = std::make_shared<StreamContextScheduleable>([=] {
      if (skip_bottom_mlp) return;
      networks[local_id]->prop_layers(networks[local_id]->bottom_layers_, false, is_train);
    });

    auto network_graph = std::make_shared<GraphScheduleable>(
        network_init, bottom_network_fprop, top_network_fprop, init_wgrad, cal_loss,
        top_network_bprop, bottom_network_bprop);

    const char* const skip_allreduce_env = std::getenv("SKIP_ALLREDUCE");
    bool skip_allreduce = (skip_allreduce_env != nullptr && 1 == std::atoi(skip_allreduce_env));

    auto network_exchange_wgrad = std::make_shared<StreamContextScheduleable>([=] {
      if (skip_allreduce) return;
      this->exchange_wgrad(local_id);
    });

    auto update_params =
        std::make_shared<StreamContextScheduleable>([=] { networks[local_id]->update_params(); });

    auto sync_back = std::make_shared<StreamContextScheduleable>([] {});

    if (solver_.train_intra_iteration_overlap) {
      std::string dp_stream = "dp";
      ebc_dp_forward->set_stream(dp_stream);
      ebc_dp_backward_index_calculation->set_stream(dp_stream);
      ebc_mp_backward_index_calculation->set_stream(dp_stream);
      ebc_dp_local_reduce->set_stream(dp_stream);
      ebc_dp_update->set_stream(dp_stream);

      std::string mp_stream = "mp";
      ebc_mp_model_forward->set_stream(mp_stream);
      ebc_mp_network_forward->set_stream(mp_stream);
      ebc_mp_network_backward->set_stream(mp_stream);
      ebc_mp_local_reduce->set_stream(mp_stream);
      ebc_mp_update->set_stream(mp_stream);

      // dp_emb_forward, bmlp_fprop wait for mp_emb_model_forward
      auto done_mp_model_forward = ebc_mp_model_forward->record_done();
      ebc_dp_forward->wait_event({done_mp_model_forward});
      bottom_network_fprop->wait_event({done_mp_model_forward}, use_graph);

      // tmlp_fprop wait for embedding
      auto done_mp_network_forward = ebc_mp_network_forward->record_done();
      auto done_dp_forward = ebc_dp_forward->record_done();
      top_network_fprop->wait_event({done_dp_forward, done_mp_network_forward}, use_graph);

      // mp_emb_bck, dp_emb_bck wait for tmlp bprop
      auto done_top_network_bprop = top_network_bprop->record_done(use_graph);
      ebc_mp_network_backward->wait_event({done_top_network_bprop});
      ebc_dp_local_reduce->wait_event({done_top_network_bprop});

      // mp_local_reduce wait mp_backward_index_calculation
      auto done_ebc_mp_backward_index_calculation =
          ebc_mp_backward_index_calculation->record_done();
      ebc_mp_local_reduce->wait_event({done_ebc_mp_backward_index_calculation});

      // allreduce wait dp local reduce
      auto done_ebc_dp_local_reduce = ebc_dp_local_reduce->record_done();
      network_exchange_wgrad->wait_event({done_ebc_dp_local_reduce});

      // dp update wait allreduce
      auto done_ebc_dp_allreduce = ebc_dp_allreduce->record_done();
      ebc_dp_update->wait_event({done_ebc_dp_allreduce});

      // sync back
      auto done_ebc_dp_update = ebc_dp_update->record_done();
      auto done_ebc_mp_update = ebc_mp_update->record_done();
      sync_back->wait_event({done_ebc_dp_update, done_ebc_mp_update});
    }

    if (!solver_.train_inter_iteration_overlap) {
      std::vector<std::shared_ptr<Scheduleable>> scheduleable_list = {
          distribute_data,
          ebc_mp_model_forward,
          ebc_mp_network_forward,
          ebc_dp_forward,
          network_graph,
          ebc_mp_backward_index_calculation,
          ebc_dp_backward_index_calculation,
          ebc_mp_network_backward,
          ebc_dp_local_reduce,
          network_exchange_wgrad,
          ebc_dp_allreduce,
          update_params,
          ebc_mp_local_reduce,
          ebc_mp_update,
          ebc_dp_update,
          sync_back,
      };
      auto done_distribute_data = distribute_data->record_done();
      ebc_mp_model_forward->wait_event({done_distribute_data});

      graph_.train_pipeline_[local_id] = Pipeline{"default", gpu_resource, scheduleable_list};
    } else {
      auto ebc_cache_train_ddl_output =
          std::make_shared<StreamContextScheduleable>([=, &ddl_output] {
            for (auto& ebc : ebc_list_) {
              ebc->cache_ddl_output(local_id, train_ddl_output_[local_id], ddl_output,
                                    train_data_reader_->get_full_batchsize());
            }
          });

      auto copy_next_iter_network_input = std::make_shared<StreamContextScheduleable>([=]() {
        if (skip_prefetch_in_last_batch(is_train)) return;

        graph_.train_copy_ops_[local_id]->run();
        graph_.train_copy_ops_[local_id + resource_manager_->get_local_gpu_count()]->run();
      });

      std::vector<std::shared_ptr<Scheduleable>> scheduleable_list = {
          ebc_cache_train_ddl_output,
          ebc_mp_model_forward,
          ebc_mp_network_forward,
          ebc_dp_forward,
          network_graph,
          ebc_mp_backward_index_calculation,
          ebc_dp_backward_index_calculation,
          distribute_data,
          ebc_mp_network_backward,
          ebc_dp_local_reduce,
          network_exchange_wgrad,
          ebc_dp_allreduce,
          update_params,
          ebc_mp_local_reduce,
          ebc_mp_update,
          ebc_dp_update,
          sync_back,
          copy_next_iter_network_input,
      };
      std::string prefetch_stream = "prefetch";

      auto done_ebc_cache_train_ddl_output = ebc_cache_train_ddl_output->record_done();
      ebc_mp_model_forward->wait_event({done_ebc_cache_train_ddl_output});

      distribute_data->set_absolute_stream(prefetch_stream);
      distribute_data->wait_event({done_ebc_cache_train_ddl_output});

      auto done_distribute_data = distribute_data->record_done();
      ebc_cache_train_ddl_output->wait_event({done_distribute_data});
      graph_.train_pipeline_[local_id] = Pipeline{"default", gpu_resource, scheduleable_list};
    }
  }
}

void Model::train_pipeline_with_ebc() {
  if (graph_.is_first_train_batch_ && solver_.train_inter_iteration_overlap) {
#pragma omp parallel num_threads(number_of_networks())
    {
      size_t id = omp_get_thread_num();
      CudaCPUDeviceContext ctx(resource_manager_->get_local_gpu(id)->get_device_id());
      HCTR_CHECK(solver_.use_embedding_collection);
      if (is_scheduled_datareader()) {
        if (auto reader =
                dynamic_cast<MultiHot::AsyncDataReader<uint32_t>*>(train_data_reader_.get())) {
          train_data_distributor_->distribute(id, reader->get_current_sparse_values()[id], {},
                                              train_ddl_output_[id],
                                              train_data_reader_->get_full_batchsize());
        } else if (auto reader = dynamic_cast<MultiHot::AsyncDataReader<long long>*>(
                       train_data_reader_.get())) {
          train_data_distributor_->distribute(id, reader->get_current_sparse_values()[id], {},
                                              train_ddl_output_[id],
                                              train_data_reader_->get_full_batchsize());
        }
      } else {
        HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput,
                       "embedding collection can only be used with AsyncMultiHot DataReader.");
      }
      graph_.train_copy_ops_[id]->run();
      graph_.train_copy_ops_[id + resource_manager_->get_local_gpu_count()]->run();

      HCTR_LIB_THROW(cudaStreamSynchronize(resource_manager_->get_local_gpu(id)->get_stream()));
    }

    if (!graph_.is_last_eval_batch_) {
      bool is_train = true;

      long long current_batchsize = read_a_batch(is_train);
      if (!current_batchsize) return;
    }
  }

  const bool use_graph = solver_.use_cuda_graph && !train_data_reader_->current_batch_incomplete();

#pragma omp parallel num_threads(resource_manager_->get_local_gpu_count())
  {
    int id = omp_get_thread_num();
    auto device_id = resource_manager_->get_local_gpu(id)->get_device_id();
    CudaCPUDeviceContext context(device_id);

    if (use_graph) {
      graph_.train_pipeline_[id].run_graph();
    } else {
      graph_.train_pipeline_[id].run();
    }
  }
}

template <typename NetworkType>
void Model::create_evaluate_pipeline_with_ebc(std::vector<std::shared_ptr<NetworkType>>& networks) {
  bool is_train = false;
  //  bool use_graph = solver_.use_cuda_graph;
  graph_.evaluate_pipeline_.resize(resource_manager_->get_local_gpu_count());

#pragma omp parallel for num_threads(resource_manager_->get_local_gpu_count())
  for (int local_id = 0; local_id < static_cast<int>(resource_manager_->get_local_gpu_count());
       local_id++) {
    auto gpu_resource = resource_manager_->get_local_gpu(local_id);
    CudaCPUDeviceContext context(gpu_resource->get_device_id());

    auto eval_data_distribute = std::make_shared<StreamContextScheduleable>([=] {
      if (skip_prefetch_in_last_batch(is_train)) return;
      if (is_scheduled_datareader()) {
        if (auto reader =
                dynamic_cast<MultiHot::AsyncDataReader<uint32_t>*>(evaluate_data_reader_.get())) {
          eval_data_distributor_->distribute(
              local_id, reader->get_current_sparse_values()[local_id], {},
              evaluate_ddl_output_[local_id], evaluate_data_reader_->get_current_batchsize());
        } else if (auto reader = dynamic_cast<MultiHot::AsyncDataReader<long long>*>(
                       evaluate_data_reader_.get())) {
          eval_data_distributor_->distribute(
              local_id, reader->get_current_sparse_values()[local_id], {},
              evaluate_ddl_output_[local_id], evaluate_data_reader_->get_current_batchsize());
        }
      } else {
        HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput,
                       "embedding collection can only be used with AsyncMultiHot DataReader.");
      }
    });

    DataDistributor::Result& ddl_output = solver_.eval_inter_iteration_overlap
                                              ? cache_evaluate_ddl_output_[local_id]
                                              : evaluate_ddl_output_[local_id];

    auto ebc_forward = [=, &ddl_output](embedding::Stage stage) {
      for (auto& ebc : ebc_list_) {
        ebc->forward_per_gpu(stage, is_train, local_id, ddl_output, evaluate_ebc_outptut_[local_id],
                             evaluate_data_reader_->get_full_batchsize());
      }
    };

    auto ebc_mp_model_forward = std::make_shared<StreamContextScheduleable>([=] {
      ebc_forward(embedding::Stage::MPModelForward);
      ebc_forward(embedding::Stage::HierMPModelForward);
      ebc_forward(embedding::Stage::DenseMPModelForward);
    });

    auto ebc_mp_network_forward = std::make_shared<StreamContextScheduleable>([=] {
      ebc_forward(embedding::Stage::MPNetworkdForward);
      ebc_forward(embedding::Stage::HierMPNetworkForward);
      ebc_forward(embedding::Stage::DenseMPNetworkForward);
    });

    auto ebc_dp_forward = std::make_shared<StreamContextScheduleable>(
        [=] { ebc_forward(embedding::Stage::DPForward); });

    auto network_eval = std::make_shared<StreamContextScheduleable>([=] {
      long long current_batchsize_per_device =
          graph_.is_last_eval_batch_
              ? evaluate_data_reader_->get_current_batchsize_per_device(local_id)
              : evaluate_data_reader_->get_full_batchsize() /
                    resource_manager_->get_global_gpu_count();
      networks[local_id]->eval(current_batchsize_per_device);
    });
    auto network_graph = std::make_shared<GraphScheduleable>(network_eval);

    auto cal_metrics = std::make_shared<StreamContextScheduleable>([=] {
      for (auto& metric : metrics_) {
        auto metric_map = networks[local_id]->get_raw_metrics_all().begin()->second;
        metric->local_reduce(local_id, metric_map);
      }
    });

    if (!solver_.eval_inter_iteration_overlap) {
      graph_.evaluate_pipeline_[local_id] =
          Pipeline{"default",
                   gpu_resource,
                   {eval_data_distribute, ebc_mp_model_forward, ebc_mp_network_forward,
                    ebc_dp_forward, network_graph, cal_metrics}};
    } else {
      auto ebc_cache_eval_ddl_output =
          std::make_shared<StreamContextScheduleable>([=, &ddl_output] {
            for (auto& ebc : ebc_list_) {
              ebc->cache_ddl_output(local_id, evaluate_ddl_output_[local_id], ddl_output,
                                    evaluate_data_reader_->get_full_batchsize());
            }
          });

      auto copy_next_iter_network_input = std::make_shared<StreamContextScheduleable>([=]() {
        if (skip_prefetch_in_last_batch(is_train)) return;

        graph_.evaluate_copy_ops_[local_id]->run();
        graph_.evaluate_copy_ops_[local_id + resource_manager_->get_local_gpu_count()]->run();
      });

      std::vector<std::shared_ptr<Scheduleable>> scheduleable_list = {
          ebc_cache_eval_ddl_output,
          ebc_mp_model_forward,
          eval_data_distribute,
          ebc_mp_network_forward,
          ebc_dp_forward,
          network_graph,
          cal_metrics,
          copy_next_iter_network_input,
      };

      //      ebc_mp_network_forward->set_stream("side_stream");
      //      ebc_dp_forward->set_stream("side_stream");
      //      network_eval->set_stream("side_stream");
      //      cal_metrics->set_stream("side_stream");

      eval_data_distribute->set_absolute_stream("prefetch");

      auto done_ebc_cache_eval_ddl_output = ebc_cache_eval_ddl_output->record_done();
      eval_data_distribute->wait_event({done_ebc_cache_eval_ddl_output});

      auto done_eval_data_distribute = eval_data_distribute->record_done();
      ebc_cache_eval_ddl_output->wait_event({done_eval_data_distribute});

      //      auto done_ebc_mp_model_forward = ebc_mp_model_forward->record_done();
      //      ebc_mp_network_forward->wait_event({done_ebc_mp_model_forward});
      //
      //      auto done_cal_metrics = cal_metrics->record_done();
      //      copy_next_iter_network_input->wait_event({done_cal_metrics});
      graph_.evaluate_pipeline_[local_id] = Pipeline{"default", gpu_resource, scheduleable_list};
    }
  }
}

void Model::evaluate_pipeline_with_ebc() {
  if (graph_.is_first_eval_batch_ && solver_.eval_inter_iteration_overlap) {
#pragma omp parallel num_threads(number_of_networks())
    {
      size_t id = omp_get_thread_num();
      CudaCPUDeviceContext ctx(resource_manager_->get_local_gpu(id)->get_device_id());
      if (is_scheduled_datareader()) {
        if (auto reader =
                dynamic_cast<MultiHot::AsyncDataReader<uint32_t>*>(evaluate_data_reader_.get())) {
          eval_data_distributor_->distribute(id, reader->get_current_sparse_values()[id], {},
                                             evaluate_ddl_output_[id],
                                             evaluate_data_reader_->get_current_batchsize());
        } else if (auto reader = dynamic_cast<MultiHot::AsyncDataReader<long long>*>(
                       evaluate_data_reader_.get())) {
          eval_data_distributor_->distribute(id, reader->get_current_sparse_values()[id], {},
                                             evaluate_ddl_output_[id],
                                             evaluate_data_reader_->get_current_batchsize());
        }

      } else {
        HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput,
                       "embedding collection can only be used with AsyncMultiHot DataReader.");
      }
      graph_.evaluate_copy_ops_[id]->run();
      graph_.evaluate_copy_ops_[id + resource_manager_->get_local_gpu_count()]->run();
      HCTR_LIB_THROW(cudaStreamSynchronize(resource_manager_->get_local_gpu(id)->get_stream()));
    }

    // in case we only have 1 batch in eval
    if (!graph_.is_last_eval_batch_) {
      bool is_train = false;
      long long current_batchsize = read_a_batch(is_train);
      if (!current_batchsize) return;
    }
  }

  const bool use_graph =
      solver_.use_cuda_graph && !evaluate_data_reader_->current_batch_incomplete();

#pragma omp parallel num_threads(number_of_networks())
  {
    size_t id = omp_get_thread_num();
    auto device_id = resource_manager_->get_local_gpu(id)->get_device_id();
    CudaCPUDeviceContext context(device_id);

    if (use_graph) {
      graph_.evaluate_pipeline_[id].run_graph();
    } else {
      graph_.evaluate_pipeline_[id].run();
    }
  }

  for (auto& metric : metrics_) {
    metric->global_reduce(number_of_networks());
  }
}

template void Model::create_train_pipeline(std::vector<std::shared_ptr<Network>>&);
template void Model::create_evaluate_pipeline(std::vector<std::shared_ptr<Network>>&);
template void Model::create_train_network_pipeline(std::vector<std::shared_ptr<Network>>&);
template void Model::create_eval_network_pipeline(std::vector<std::shared_ptr<Network>>&);
template void Model::create_train_pipeline_with_ebc(
    std::vector<std::shared_ptr<Network>>& networks);
template void Model::create_evaluate_pipeline_with_ebc(std::vector<std::shared_ptr<Network>>&);

}  // namespace HugeCTR
