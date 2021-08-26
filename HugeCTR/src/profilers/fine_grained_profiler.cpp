#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <unistd.h>

#include <chrono>
#include <common.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <profiler.hpp>
#include <sstream>
#include <string>
#include <vector>

using nlohmann::json;

namespace HugeCTR {
namespace Profiler {

void FineGrainedProfiler::initialize(bool use_cuda_graph, bool exit_when_finished) {
  BaseProfiler::initialize(use_cuda_graph, exit_when_finished);
  MESSAGE_("Profiler using PROFILING_MODE: fine_grained");

  char* repeat_times_str = std::getenv("PROFILING_REPEAT_TIMES_PER_EVENT");
  if (repeat_times_str == NULL) {
    repeat_times_ = 50;
  } else {
    repeat_times_ = std::atoi(repeat_times_str);
  }
  MESSAGE_(std::string("Profiler using REPEAT_TIMES_PER_EVENT: ") + std::to_string(repeat_times_));

  if (use_cuda_graph) {
    repeat_times_ += warmup_after_cudagraph_reinit_;
  }

  repeat_times_ += 1;
  current_reapted_times_ = 0;
  current_event_idx_ = 0;
  init_cuda_graph_this_iter = false;
  map_stream_to_gpu_timer_.clear();
}

bool FineGrainedProfiler::iter_check() {
  auto end = std::chrono::steady_clock::now();
  for (auto& x : map_internal_) {
    for (auto it = x.second->begin(); it != x.second->end(); it++) {
      it->second = 0;
    }
  }
  if (current_iteration_ <= warmup_iterations_ - 1) {
    current_iteration_ += 1;
    return false;
  } else {
    if (events_.size() == 0) {
      MESSAGE_(
          "No profiling labels found int code or they are not in prof.events. Please have a check. "
          "Program exit.");
      std::exit(0);
    }
  }
  if (current_iteration_ == warmup_iterations_) {
    // first iteration
    current_iteration_ += 1;
    prepare_iter_start();
  } else {
    // not first iteration
    // whether to record result logic block
    if (!init_cuda_graph_this_iter) {
      if (!use_cuda_graph_ ||
          (use_cuda_graph_ && current_reapted_times_ > warmup_after_cudagraph_reinit_)) {
        iter_time_ms_.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - iter_check_).count() /
            1000000.0);

        auto gpu_timer =
            map_stream_to_gpu_timer_[static_cast<GPUEvent*>(events_[current_event_idx_].get())
                                         ->stream];

        auto measured_time_ms = gpu_timer->get_measured_time_ms();
        if (measured_time_ms < 0.0) {
          throw internal_runtime_error(HugeCTR::Error_t::UnspecificError,
                                       gpu_event_strfy(events_[current_event_idx_].get()));
        }
        events_[current_event_idx_]->measured_times_ms.push_back(measured_time_ms);
        events_[current_event_idx_]->iter_start_to_event_start_times_ms.push_back(
            gpu_timer->get_iter_start_to_event_start_ms());
        auto extra_info_start = gpu_timer->extra_info_start;
        auto extra_info_stop = gpu_timer->extra_info_stop;
        if (!extra_info_start.empty()) {
          events_[current_event_idx_]->extra_infos_start.push_back(extra_info_start);
        }
        if (!extra_info_stop.empty()) {
          events_[current_event_idx_]->extra_infos_stop.push_back(extra_info_stop);
        }
      }
    }
    current_iteration_ += 1;
    current_reapted_times_ += 1;

    if (current_reapted_times_ >= repeat_times_) {
      current_reapted_times_ = 0;
      current_event_idx_ += 1;
    }

    if (current_event_idx_ >= int(events_.size())) {
      write_result();
      MESSAGE_("Profiling complete!");
      if (exit_when_finished_) {
        MESSAGE_("Program exit.");
        std::exit(0);
      }
      return true;
    }
    prepare_iter_start();
  }
  return false;
  // MESSAGE_(std::string("Iter: ") + std::to_string(current_iteration_) + " event_idx: " +
  //          std::to_string(current_event_idx_) + " repeat_times: " +
  //          std::to_string(current_reapted_times_) + " init cudagraph: " +
  //          std::to_string(init_cuda_graph_this_iter));
}

void FineGrainedProfiler::prepare_iter_start() {
  if (use_cuda_graph_) {
    if (current_reapted_times_ == 0) {
      MESSAGE_(std::string("Iteration: ") + std::to_string(current_iteration_) +
               std::string(". Profiler re-instantiate cuda graph for ") +
               gpu_event_strfy(events_[current_event_idx_].get()));
      init_cuda_graph_this_iter = true;
    } else {
      init_cuda_graph_this_iter = false;
    }
  } else {
    if (current_reapted_times_ == 0) {
      MESSAGE_(std::string("Iteration: ") + std::to_string(current_iteration_) + ". " +
               std::to_string(current_event_idx_) + " : " +
               events_[current_event_idx_]->event_name + " " +
               std::to_string(static_cast<GPUEvent*>(events_[current_event_idx_].get())
                                  ->met_times_within_this_stream) +
               " on " +
               stream_str(static_cast<GPUEvent*>(events_[current_event_idx_].get())->stream));
    }
    init_cuda_graph_this_iter = false;
  }

  for (auto& s_and_gt : map_stream_to_gpu_timer_) {
    s_and_gt.second->extra_info_start.clear();
    s_and_gt.second->extra_info_stop.clear();
  }

  auto target_stream = static_cast<GPUEvent*>(events_[current_event_idx_].get())->stream;
  map_stream_to_gpu_timer_[target_stream]->iter_start(target_stream);
  iter_check_ = std::chrono::steady_clock::now();
}

void FineGrainedProfiler::record_event(const char* event_label_char, cudaStream_t stream,
                                       bool could_be_in_cuda_graph, int device_id,
                                       const std::string& extra_info) {
  try {
    // auto t_start = std::chrono::steady_clock::now();
    auto name_and_type = get_event_name_and_type(event_label_char);
    auto event_name = std::get<0>(name_and_type);
    auto event_type = std::get<1>(name_and_type);
    // above string operation cost 0.000xxx ms on DGXA100. x usually is 1 - 2.
    if (current_iteration_ <= warmup_iterations_) {
      mtx_.lock();
      thread_local int original_device_id = set_and_keep_original_device(device_id);
      int met_times_within_this_stream =
          access_or_insert_in_event_met_times_in_stream(stream, event_name);
      int read_device_id = device_id < 0 ? original_device_id : device_id;
      try_create_one_gpu_event(event_name, event_type, read_device_id, stream);
      auto map_iter = map_stream_to_gpu_timer_.find(stream);
      if (map_iter == map_stream_to_gpu_timer_.end()) {
        map_stream_to_gpu_timer_[stream] = std::make_shared<GPUTimer>();
      }
      if (event_type == "stop") {
        map_internal_[stream]->operator[](event_name) = met_times_within_this_stream + 1;
      }
      restore_original_device(original_device_id, device_id);
      mtx_.unlock();
    } else {
      int met_times_within_this_stream = map_internal_[stream]->at(event_name);
      if (events_[current_event_idx_]->event_name != event_name ||
          static_cast<GPUEvent*>(events_[current_event_idx_].get())->stream != stream ||
          static_cast<GPUEvent*>(events_[current_event_idx_].get())->met_times_within_this_stream !=
              met_times_within_this_stream) {
        if (event_type == "stop") {
          map_internal_[stream]->operator[](event_name) = met_times_within_this_stream + 1;
        }
        return;
      }
      // above map and if compare costs 0.000x ms on DGXA100, x is usually 1 - 7.
      thread_local int original_device_id = Profiler::set_and_keep_original_device(device_id);
      auto gpu_timer = map_stream_to_gpu_timer_[stream];
      // above getdevice and mapping costs 0.000x ms on DGXA100, x is usually 1 - 2.

      if (event_type == "start") {
        gpu_timer->extra_info_start = extra_info;
        gpu_timer->event_start(stream, use_cuda_graph_);
      } else {
        gpu_timer->extra_info_stop = extra_info;
        gpu_timer->event_stop(stream, use_cuda_graph_);
        // event_start and event_stop usually costs 0.002ms on DGXA100
        map_internal_[stream]->operator[](event_name) = met_times_within_this_stream + 1;
        // Above post event record operation costs 0.00x on DGXA100, usually x is 1 - 2.
      }
      Profiler::restore_original_device(original_device_id, device_id);
    }
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

void FineGrainedProfiler::write_result(const char* file_path) {
  int ret = std::system((std::string("mkdir -p ") + profiling_dir).c_str());
  if (ret != 0) {
    MESSAGE_("Creating PROFILING_DIR failed?");
  }

  std::string result_file;
  if (file_path) {
    result_file = profiling_dir + '/' + file_path;
  } else {
    result_file = profiling_dir + '/' + host_name_ + ".fine_grained.json";
  }

  MESSAGE_(std::string("Result json file is wrote in ") + result_file + ".");

  json result;
  result["host_name"] = host_name_;
  result["iter_time_ms"] = iter_time_ms_;
  result["events"] = json::array();
  for (auto& event_p : events_) {
    GPUEvent* gep = static_cast<GPUEvent*>(event_p.get());
    json j;
    j["event_name"] = gep->event_name;
    j["device_id"] = gep->device_id;
    j["stream"] = stream_str(gep->stream);
    j["start_index"] = gep->start_index;
    j["end_index"] = gep->end_index;
    j["met_times_within_this_stream"] = gep->met_times_within_this_stream;
    j["measured_times_ms"] = gep->measured_times_ms;
    j["iter_start_to_event_start_times_ms"] = gep->iter_start_to_event_start_times_ms;
    j["extra_infos_start"] = gep->extra_infos_start;
    j["extra_infos_stop"] = gep->extra_infos_stop;
    result["events"].push_back(j);
  }

  std::string result_jstring = result.dump();
  std::ofstream outfile;
  outfile.open(result_file.c_str());
  outfile << result_jstring;
  outfile.close();
}

}  //  namespace Profiler

Profiler::FineGrainedProfiler global_fine_grained_profiler;

}  //  namespace HugeCTR