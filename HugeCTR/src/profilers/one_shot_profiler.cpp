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

void OneShotProfiler::initialize(bool use_cuda_graph, bool exit_when_finished) {
  BaseProfiler::initialize(use_cuda_graph, exit_when_finished);

  MESSAGE_("Profiler using PROFILING_MODE: one_shot");

  char* repeat_iters_str = std::getenv("PROFILING_REPEAT_ITERS");
  if (repeat_iters_str == NULL) {
    repeat_iters_ = 1000;
  } else {
    repeat_iters_ = std::atoi(repeat_iters_str);
  }
  MESSAGE_(std::string("Profiler using PROFILING_REPEAT_ITERS: ") + std::to_string(repeat_iters_));
  repeat_iters_ += warmup_iterations_;

  char* record_every_n_str = std::getenv("PROFILING_RECORD_EVERY_N");
  if (record_every_n_str == NULL) {
    record_every_n_ = 5;
  } else {
    record_every_n_ = std::atoi(record_every_n_str);
  }
  MESSAGE_(std::string("Profiler using PROFILING_RECORD_EVERY_N: ") +
           std::to_string(record_every_n_));

  init_cuda_graph_this_iter = true;
  map_event_key_to_gpu_timer_.clear();
}

bool OneShotProfiler::iter_check(int gpus) {
  if (current_iteration_ < warmup_iterations_) {
    sync_all_gpus(gpus);
    // global_data_reader_one_shot_profiler.phase = 1;
    current_iteration_ += 1;
  } else if (current_iteration_ == warmup_iterations_) {
    sync_all_gpus(gpus);
    if (events_.size() == 0) {
      MESSAGE_(
          "No profiling labels found int code or they are not in prof.events. Please have a check. "
          "Program exit.");
      std::exit(0);
    }
    // global_data_reader_one_shot_profiler.phase = 2;
    current_iteration_ += 1;
  } else {
    bool time_to_sync_and_record =
        current_iteration_ > warmup_iterations_ && current_iteration_ % record_every_n_ == 0;
    if (time_to_sync_and_record) {
      MESSAGE_(std::string("Profiler sync and record in iteration: ") +
               std::to_string(current_iteration_));
      sync_all_gpus(gpus);
      // record the result
      // sync and record the event
      for (auto it = map_event_key_to_gpu_timer_.begin(); it != map_event_key_to_gpu_timer_.end();
           it++) {
        auto event_key = it->first;
        auto gpu_timer = it->second;
        auto event_idx = map_event_key_to_event_idx_.at(event_key);

        auto device_id = ((GPUEvent*)(events_[event_idx].get()))->device_id;
        auto iter_start_event = find_iter_start_event(device_id);

        float iter_start_to_event_start_time_ms;
        CK_CUDA_THROW_(cudaEventElapsedTime(&iter_start_to_event_start_time_ms, iter_start_event,
                                            gpu_timer->start_));
        if (iter_start_to_event_start_time_ms < 0.0) {
          throw internal_runtime_error(HugeCTR::Error_t::UnspecificError,
                                       gpu_event_strfy(events_[event_idx].get()));
        }
        events_[event_idx]->iter_start_to_event_start_times_ms.push_back(
            iter_start_to_event_start_time_ms);

        auto measured_time_ms = gpu_timer->get_measured_time_ms();
        if (measured_time_ms < 0.0) {
          throw internal_runtime_error(HugeCTR::Error_t::UnspecificError,
                                       gpu_event_strfy(events_[event_idx].get()));
        }
        events_[event_idx]->measured_times_ms.push_back(measured_time_ms);

        auto extra_info_start = gpu_timer->extra_info_start;
        auto extra_info_stop = gpu_timer->extra_info_stop;
        if (!extra_info_start.empty()) {
          events_[event_idx]->extra_infos_start.push_back(extra_info_start);
        }
        if (!extra_info_stop.empty()) {
          events_[event_idx]->extra_infos_stop.push_back(extra_info_stop);
        }
      }
      // global_data_reader_one_shot_profiler.iter_check();
    }

    if (current_iteration_ > repeat_iters_) {
      sync_all_gpus(gpus);
      write_result();
      MESSAGE_("Profiling complete!");
      if (exit_when_finished_) {
        MESSAGE_("Program exit.");
        std::exit(0);
      }
      return true;
    }
    current_iteration_ += 1;
  }
  determine_init_cuda_graph();
  clear_map_interal();
  return false;
}

void OneShotProfiler::determine_init_cuda_graph() {
  if (use_cuda_graph_ && current_iteration_ <= warmup_iterations_ + 1) {
    init_cuda_graph_this_iter = true;
    if (current_iteration_ == warmup_iterations_ + 1) {
      MESSAGE_(std::string("Profiler parsed ") + std::to_string(events_.size()) +
               " events on different streams");
    }
  } else {
    init_cuda_graph_this_iter = false;
  }
}

void OneShotProfiler::record_event(const char* event_label_char, cudaStream_t stream,
                                   bool could_be_in_cuda_graph, int device_id,
                                   const std::string& extra_info) {
  try {
    auto name_and_type = get_event_name_and_type(event_label_char);
    auto event_name = std::get<0>(name_and_type);
    auto event_type = std::get<1>(name_and_type);

    if (current_iteration_ <= warmup_iterations_) {
      mtx_.lock();
      thread_local int original_device_id = set_and_keep_original_device(device_id);
      int met_times_within_this_stream =
          access_or_insert_in_event_met_times_in_stream(stream, event_name);
      int read_device_id = device_id < 0 ? original_device_id : device_id;
      bool created = try_create_one_gpu_event(event_name, event_type, read_device_id, stream);
      auto event_key = gen_event_key(event_name, stream, met_times_within_this_stream);
      if (created && event_type == "start") {
        map_event_key_to_gpu_timer_[event_key] = std::make_shared<GPUTimer>();
      }
      if (event_type == "stop") {
        map_internal_[stream]->operator[](event_name) = met_times_within_this_stream + 1;
      }
      restore_original_device(original_device_id, device_id);
      mtx_.unlock();
    } else {
      int met_times_within_this_stream = map_internal_[stream]->at(event_name);
      auto event_key = gen_event_key(event_name, stream, met_times_within_this_stream);
      int event_idx = find_event(event_key);
      if (event_idx < 0) {
        if (event_type == "stop") {
          map_internal_[stream]->operator[](event_name) = met_times_within_this_stream + 1;
        }
        return;
      }
      thread_local int original_device_id = Profiler::set_and_keep_original_device(device_id);
      auto gpu_timer = map_event_key_to_gpu_timer_[event_key];
      if (event_type == "start") {
        gpu_timer->event_start(stream, use_cuda_graph_);
        gpu_timer->extra_info_start = extra_info;
      } else {
        gpu_timer->event_stop(stream, use_cuda_graph_);
        gpu_timer->extra_info_stop = extra_info;
        map_internal_[stream]->operator[](event_name) = met_times_within_this_stream + 1;
      }
      Profiler::restore_original_device(original_device_id, device_id);
    }
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

void OneShotProfiler::write_result(const char* file_path) {
  int ret = std::system((std::string("mkdir -p ") + profiling_dir).c_str());
  if (ret != 0) {
    MESSAGE_("Creating PROFILING_DIR failed?");
  }

  std::string result_file;
  if (file_path) {
    result_file = profiling_dir + '/' + file_path;
  } else {
    result_file = profiling_dir + '/' + host_name_ + ".one_shot.json";
  }

  MESSAGE_(std::string("Result json file is wrote in ") + result_file + ".");

  json result;
  result["host_name"] = host_name_;
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

  // data reader
  // for (auto& x : global_data_reader_one_shot_profiler.map_stream_to_stream_recorder) {
  //   auto recorder = x.second;
  //   json j_h2d;
  //   j_h2d["event_name"] = "data_reader_memcpy_h2d";
  //   j_h2d["device_id"] = recorder->device_id;
  //   j_h2d["stream"] = stream_str(recorder->stream);
  //   j_h2d["start_index"] = -1;
  //   j_h2d["end_index"] = -1;
  //   j_h2d["met_times_within_this_stream"] = -1;
  //   j_h2d["measured_times_ms"] = recorder->measured_times_ms_h2d;
  //   j_h2d["iter_start_to_event_start_times_ms"] = recorder->iter_start_to_event_start_times_ms_h2d;
  //   j_h2d["extra_infos_start"] = "";
  //   j_h2d["extra_infos_stop"] = "";
  //   result["events"].push_back(j_h2d);
  //   json j_p2p;
  //   j_p2p["event_name"] = "data_reader_memcpy_p2p";
  //   j_p2p["device_id"] = recorder->device_id;
  //   j_p2p["stream"] = stream_str(recorder->stream);
  //   j_p2p["start_index"] = -1;
  //   j_p2p["end_index"] = -1;
  //   j_p2p["met_times_within_this_stream"] = -1;
  //   j_p2p["measured_times_ms"] = recorder->measured_times_ms_p2p;
  //   j_p2p["iter_start_to_event_start_times_ms"] = recorder->iter_start_to_event_start_times_ms_p2p;
  //   j_p2p["extra_infos_start"] = "";
  //   j_p2p["extra_infos_stop"] = "";
  //   result["events"].push_back(j_p2p);
  // }

  std::string result_jstring = result.dump();
  std::ofstream outfile;
  outfile.open(result_file.c_str());
  outfile << result_jstring;
  outfile.close();
}

cudaEvent_t OneShotProfiler::find_iter_start_event(int device_id) {
  cudaEvent_t iter_start_event = nullptr;
  try {
    iter_start_event = map_device_id_to_iter_gpu_timer_.at(device_id)->start_;
  } catch (const std::out_of_range& e) {
    for (auto& event_p : events_) {
      auto id = ((GPUEvent*)(event_p.get()))->device_id;
      auto event_name = ((GPUEvent*)(event_p.get()))->event_name;
      if (id == device_id && event_name == "iteration") {
        auto event_key = gen_event_key("iteration", ((GPUEvent*)(event_p.get()))->stream, 0);
        auto timer = map_event_key_to_gpu_timer_[event_key];
        map_device_id_to_iter_gpu_timer_[device_id] = timer;
        iter_start_event = timer->start_;
        break;
      }
    }
  }
  return iter_start_event;
}

}  //  namespace Profiler

Profiler::OneShotProfiler global_one_shot_profiler;

}  //  namespace HugeCTR