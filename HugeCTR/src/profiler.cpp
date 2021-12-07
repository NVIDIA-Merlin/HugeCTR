#include <cuda.h>
#include <cuda_runtime.h>
#include <limits.h>
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

#if CUDA_VERSION < 11010
// this won't work for cuda graph, just to pass the compiling
#define CUDA_GRAPH_EVENT_RECORD(...) cudaEventRecord(__VA_ARGS__)
#else
#define CUDA_GRAPH_EVENT_RECORD(...) cudaEventRecordWithFlags(__VA_ARGS__, cudaEventRecordExternal)
#endif

using nlohmann::json;

namespace HugeCTR {

Profiler::GPUTimer::GPUTimer() {
  CK_CUDA_THROW_(cudaEventCreateWithFlags(&iter_start_, cudaEventBlockingSync));
  CK_CUDA_THROW_(cudaEventCreateWithFlags(&start_, cudaEventBlockingSync));
  CK_CUDA_THROW_(cudaEventCreateWithFlags(&stop_, cudaEventBlockingSync));
}

Profiler::GPUTimer::~GPUTimer() {
  cudaEventDestroy(iter_start_);
  cudaEventDestroy(start_);
  cudaEventDestroy(stop_);
}

void Profiler::GPUTimer::iter_start(cudaStream_t stream) {
  CK_CUDA_THROW_(cudaEventRecord(iter_start_, stream));
}

void Profiler::GPUTimer::event_start(cudaStream_t stream, bool in_cuda_graph) {
  if (in_cuda_graph) {
    CK_CUDA_THROW_(CUDA_GRAPH_EVENT_RECORD(start_, stream));
  } else {
    CK_CUDA_THROW_(cudaEventRecord(start_, stream));
  }
}

void Profiler::GPUTimer::event_stop(cudaStream_t stream, bool in_cuda_graph) {
  if (in_cuda_graph) {
    CK_CUDA_THROW_(CUDA_GRAPH_EVENT_RECORD(stop_, stream));
  } else {
    CK_CUDA_THROW_(cudaEventRecord(stop_, stream));
  }
}

void Profiler::GPUTimer::sync_stop() { CK_CUDA_THROW_(cudaEventSynchronize(stop_)); }

float Profiler::GPUTimer::get_iter_start_to_event_start_ms() {
  float iter_start_to_event_start_ms;
  CK_CUDA_THROW_(cudaEventElapsedTime(&iter_start_to_event_start_ms, iter_start_, start_));
  return iter_start_to_event_start_ms;
}

float Profiler::GPUTimer::get_measured_time_ms() {
  float measured_time_ms;
  CK_CUDA_THROW_(cudaEventElapsedTime(&measured_time_ms, start_, stop_));
  return measured_time_ms;
}

void Profiler::initialize(bool use_cuda_graph, bool exit_when_finished) {
  char* pd = std::getenv("PROFILING_DIR");
  if (pd == NULL) {
    std::string msg(
        "Got empty for env PROFILING_DIR. You must specify if when using this profiler");
    MESSAGE_(msg);
    throw std::invalid_argument(msg);
  }
  profiling_dir = std::string(pd);
  MESSAGE_(std::string("Profiler using PROFILING_DIR: ") + profiling_dir);

  interested_events_.clear();
  std::ifstream interested_events_file((profiling_dir + "/" + "prof.events").c_str());
  if (interested_events_file.good()) {
    MESSAGE_("Profiler using prof.events inside PROFILING_DIR");
    for (std::string line; getline(interested_events_file, line);) {
      interested_events_.push_back(line);
    }
  }

  char* warmup_iterations_str = std::getenv("PROFILING_WARMUP_ITERS");
  if (warmup_iterations_str == NULL) {
    warmup_iterations_ = 11;
  } else {
    warmup_iterations_ = std::atoi(warmup_iterations_str) + 1;
  }
  MESSAGE_(std::string("Profiler using WARMUP_ITERS: ") + std::to_string(warmup_iterations_ - 1));
  char* repeat_times_str = std::getenv("PROFILING_REPEAT_TIMES_PER_EVENT");
  if (repeat_times_str == NULL) {
    repeat_times_ = 50;
  } else {
    repeat_times_ = std::atoi(repeat_times_str);
  }
  MESSAGE_(std::string("Profiler using REPEAT_TIMES_PER_EVENT: ") + std::to_string(repeat_times_));
  char* warmup_after_cudagraph_reinit_str = std::getenv("PROFILING_WARMUP_AFTER_CUDAGRAPH_REINIT");
  if (warmup_after_cudagraph_reinit_str == NULL) {
    warmup_after_cudagraph_reinit_ = 10;
  } else {
    warmup_after_cudagraph_reinit_ = std::atoi(warmup_after_cudagraph_reinit_str);
  }
  if (use_cuda_graph) {
    MESSAGE_(std::string("Profiler using WARMUP_AFTER_CUDAGRAPH_REINIT: ") +
             std::to_string(warmup_after_cudagraph_reinit_));
    // for extra cuda graph init iter, it won't count
    repeat_times_ += warmup_after_cudagraph_reinit_;
  }
  char* data_collection_iterations_str = std::getenv("PROFILING_DATA_COLLECTION_ITERS");
  if (data_collection_iterations_str == NULL) {
    data_collection_iterations_ = 0;
  } else {
    data_collection_iterations_ = std::atoi(data_collection_iterations_str);
  }
  MESSAGE_(std::string("Profiler using DATA_COLLECTION_ITERS: ") +
           std::to_string(data_collection_iterations_));

  repeat_times_ += 1;
  current_reapted_times_ = 0;

  char host_name[HOST_NAME_MAX + 1];
  gethostname(host_name, sizeof(host_name));
  host_name_ = std::string(host_name);
  use_cuda_graph_ = use_cuda_graph;
  if (CUDA_VERSION < 11010 && use_cuda_graph_) {
    MESSAGE_(std::string("CUDA version is ") + std::to_string(CUDA_VERSION) +
             ". Profiler do not support use_cuda_graph = true and CUDA version < 11.1 at the same "
             "time." +
             " Consider use higher CUDA version or use_cuda_graph = false. Program exit.");
    std::exit(0);
  }

  MESSAGE_(std::string("Profiler using cuda graph: ") + std::to_string(use_cuda_graph_));
  if (use_cuda_graph_) {
    MESSAGE_(
        "Profiler Warning. 'extra_info' arg in the PROFILE_RECORD maybe ignored, if the event is "
        "executed in cuda graph.");
    if (data_collection_iterations_ > 0) {
      MESSAGE_("Profiler Warning. Data collection may not fuction when cuda graph is ON!");
    }
  }

  exit_when_finished_ = exit_when_finished;
  current_iteration_ = 0;
  current_event_idx_ = 0;
  events_num_ = 0;
  init_cuda_graph_this_iter = false;
  unit_test_mode = false;
  current_data_collection_iteration_ = 0;
  record_data_phase = false;
  iter_time_ms_.clear();
  map_stream_to_gpu_timer_.clear();
  events_.clear();
  map_event_key_to_event_idx_.clear();
  map_internal_.clear();
  runtime_data_.clear();
}

bool Profiler::iter_check() {
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
    if (!init_cuda_graph_this_iter && !record_data_phase) {
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
        events_[current_event_idx_]->measured_times_ms.push_back(gpu_timer->get_measured_time_ms());
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
      if (current_data_collection_iteration_ >= data_collection_iterations_) {
        write_result();
        MESSAGE_("Profiling complete!");
        if (exit_when_finished_) {
          MESSAGE_("Program exit.");
          std::exit(0);
        }
        return true;
      } else {
        if (current_data_collection_iteration_ % 100 == 0) {
          MESSAGE_(std::string("Iteration: ") + std::to_string(current_iteration_) +
                   ". Profiler collecting run time data ...");
        }
        record_data_phase = true;
        current_data_collection_iteration_++;
        return false;
      }
    }
    prepare_iter_start();
  }
  return false;
  // MESSAGE_(std::string("Iter: ") + std::to_string(current_iteration_) + " event_idx: " +
  //          std::to_string(current_event_idx_) + " repeat_times: " +
  //          std::to_string(current_reapted_times_) + " init cudagraph: " +
  //          std::to_string(init_cuda_graph_this_iter));
}

void Profiler::prepare_iter_start() {
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

void Profiler::record_event(const char* event_label_char, cudaStream_t stream,
                            bool could_be_in_cuda_graph, int device_id,
                            const std::string& extra_info) {
  try {
    // auto t_start = std::chrono::steady_clock::now();
    std::string event_label = std::string(event_label_char);
    int dot_pos = event_label.find_last_of(std::string("."));
    std::string event_type = event_label.substr(dot_pos + 1);
    if (event_type != "start" && event_type != "stop") {
      throw internal_runtime_error(
          HugeCTR::Error_t::UnspecificError,
          std::string("Invalid event name. Should end with .start or .stop"));
    }
    std::string event_name = event_label.substr(0, dot_pos);
    // above string operation cost 0.000xxx ms on DGXA100. x usually is 1 - 2.
    if (current_iteration_ <= warmup_iterations_) {
      mtx_.lock();

      thread_local int current_device_id;
      CK_CUDA_THROW_(cudaGetDevice(&current_device_id));
      if (device_id < 0) {
        device_id = current_device_id;
      }
      if (current_device_id != device_id) {
        CK_CUDA_THROW_(cudaSetDevice(device_id));
      }

      auto map_iter = map_stream_to_gpu_timer_.find(stream);
      if (map_iter == map_stream_to_gpu_timer_.end()) {
        auto gpu_timer = std::make_shared<GPUTimer>();
        map_stream_to_gpu_timer_[stream] = gpu_timer;
        map_internal_[stream] = std::make_shared<std::map<std::string, int>>();
      }
      int met_times_within_this_stream;
      try {
        met_times_within_this_stream = map_internal_[stream]->at(event_name);
      } catch (const std::out_of_range& e) {
        map_internal_[stream]->insert({event_name, 0});
        met_times_within_this_stream = 0;
      }

      if (interested_events_.size() > 0) {
        if (std::find(interested_events_.begin(), interested_events_.end(), event_name) ==
            interested_events_.end()) {
          if (event_type == "stop") {
            map_internal_[stream]->operator[](event_name) = met_times_within_this_stream + 1;
          }
          if (current_device_id != device_id) {
            CK_CUDA_THROW_(cudaSetDevice(current_device_id));
          }
          mtx_.unlock();
          return;
        }
      }
      std::string event_key = gen_event_key(event_name, stream, met_times_within_this_stream);
      int event_idx = find_event(event_key);

      if (event_type == "start") {
        if (event_idx >= 0) {
          // event exist!
          if (current_device_id != device_id) {
            CK_CUDA_THROW_(cudaSetDevice(current_device_id));
          }
          mtx_.unlock();
          return;
        }

        // create new event
        auto gpu_event = new GPUEvent;
        gpu_event->event_name = event_name;
        gpu_event->met_times_within_this_stream = met_times_within_this_stream;
        gpu_event->start_index = events_num_;
        gpu_event->end_index = -1;  // wait for stop event to set,
        gpu_event->measured_times_ms = std::vector<float>();
        gpu_event->iter_start_to_event_start_times_ms = std::vector<float>();
        gpu_event->device_id = device_id;
        gpu_event->stream = stream;
        gpu_event->extra_infos_start = std::vector<std::string>();
        gpu_event->extra_infos_stop = std::vector<std::string>();
        events_.push_back(std::shared_ptr<Event>(static_cast<Event*>(gpu_event)));
        map_event_key_to_event_idx_[event_key] = events_.size() - 1;
        events_num_++;
        // PROFILER_DEBUG_(std::string("Parsed a new GPU event ") + event_label + " occured_time " +
        // std::to_string(met_times_within_this_stream));
        if (current_device_id != device_id) {
          CK_CUDA_THROW_(cudaSetDevice(current_device_id));
        }
      } else {
        // event_name == "stop"
        // only update the end_index
        if (event_idx >= 0) {
          auto event = events_[event_idx];
          if (event->end_index < 0) {
            event->end_index = events_num_;
            events_num_++;
          }
          map_internal_[stream]->operator[](event_name) = met_times_within_this_stream + 1;
          // PROFILER_DEBUG_(std::string("Parsed a new GPU event ") + event_label + " occured_time "
          // + std::to_string(met_times_within_this_stream));
        } else {
          throw internal_runtime_error(
              HugeCTR::Error_t::UnspecificError,
              std::string("Event ") + event_name + std::string(" has stop but no start"));
        }
      }
      if (current_device_id != device_id) {
        CK_CUDA_THROW_(cudaSetDevice(current_device_id));
      }
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
      thread_local int current_device_id;
      CK_CUDA_THROW_(cudaGetDevice(&current_device_id));
      if (device_id < 0) {
        device_id = current_device_id;
      }
      if (current_device_id != device_id) {
        CK_CUDA_THROW_(cudaSetDevice(device_id));
      }
      auto gpu_timer = map_stream_to_gpu_timer_[stream];
      // above getdevice and mapping costs 0.000x ms on DGXA100, x is usually 1 - 2.

      if (event_type == "start") {
        gpu_timer->extra_info_start = extra_info;
        gpu_timer->event_start(stream, use_cuda_graph_ && could_be_in_cuda_graph);
      } else {
        gpu_timer->extra_info_stop = extra_info;
        gpu_timer->event_stop(stream, use_cuda_graph_ && could_be_in_cuda_graph);
        // event_start and event_stop usually costs 0.002ms on DGXA100
        map_internal_[stream]->operator[](event_name) = met_times_within_this_stream + 1;
        // Above post event record operation costs 0.00x on DGXA100, usually x is 1 - 2.
      }
      if (current_device_id != device_id) {
        CK_CUDA_THROW_(cudaSetDevice(current_device_id));
      }
    }
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

bool Profiler::record_data(const char* data_label_char, cudaStream_t stream,
                           const std::string& data, int device_id) {
  try {
    std::string data_label = std::string(data_label_char);
    int dot_pos = data_label.find_last_of(std::string("."));
    std::string type = data_label.substr(dot_pos + 1);
    if (type != "start" && type != "stop") {
      throw internal_runtime_error(
          HugeCTR::Error_t::UnspecificError,
          std::string("Invalid data name. Should end with .start or .stop"));
    }
    std::string data_name = data_label.substr(0, dot_pos);
    if (!record_data_phase) {
      return false;
    }
    if (type == "start") {
      return true;
    }
    thread_local int current_device_id;
    CK_CUDA_THROW_(cudaGetDevice(&current_device_id));
    if (device_id < 0) {
      device_id = current_device_id;
    }
    mtx_.lock();
    int i = 0;
    for (; i < int(runtime_data_.size()); i++) {
      if (runtime_data_[i]->data_name == data_name && runtime_data_[i]->device_id == device_id &&
          runtime_data_[i]->stream == stream) {
        runtime_data_[i]->data.push_back(data);
        break;
      }
    }

    if (i == int(runtime_data_.size())) {
      auto rdp = std::make_shared<RuntimeData>();
      rdp->data_name = data_name;
      rdp->device_id = device_id;
      rdp->stream = stream;
      rdp->data = {data};
      runtime_data_.push_back(rdp);
    }

    mtx_.unlock();
    return false;
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

void Profiler::record_event_unit_test(const char* event_label_char, cudaStream_t stream,
                                      bool could_be_in_cuda_graph, int device_id,
                                      const std::string& extra_info) {
  try {
    // should only be working in single thread
    thread_local int current_device_id;
    CK_CUDA_THROW_(cudaGetDevice(&current_device_id));
    if (device_id < 0) {
      device_id = current_device_id;
    }
    if (current_device_id != device_id) {
      CK_CUDA_THROW_(cudaSetDevice(device_id));
    }
    mtx_.lock();
    cudaEvent_t check_point;
    unit_test_events_.push_back(check_point);
    cudaEvent_t* p = &(*(unit_test_events_.end() - 1));
    CK_CUDA_THROW_(cudaEventCreateWithFlags(p, cudaEventBlockingSync));
    unit_test_labels_.push_back(std::string(event_label_char));
    unit_test_streams_.push_back(stream);
    unit_test_devices_.push_back(device_id);
    unit_test_extra_infos_.push_back(extra_info);
    mtx_.unlock();

    CK_CUDA_THROW_(cudaEventRecord(*p, stream));

    if (current_device_id != device_id) {
      CK_CUDA_THROW_(cudaSetDevice(current_device_id));
    }

  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

void Profiler::unit_test_start(const char* test_name) {
  unit_test_mode = true;
  MESSAGE_("Just a info. Profiler does not support cuda graph in unit test.");
  char* pd = std::getenv("PROFILING_DIR");
  if (pd == NULL) {
    std::string msg(
        "Got empty for env PROFILING_DIR. You must specify if when using this profiler");
    MESSAGE_(msg);
    throw std::invalid_argument(msg);
  }
  profiling_dir = std::string(pd);
  MESSAGE_(std::string("Profiler using PROFILING_DIR: ") + profiling_dir);

  char host_name[HOST_NAME_MAX + 1];
  gethostname(host_name, sizeof(host_name));
  host_name_ = std::string(host_name);
  map_internal_.clear();
  unit_test_labels_.clear();
  unit_test_events_.clear();
  unit_test_devices_.clear();
  unit_test_labels_.clear();
  unit_test_extra_infos_.clear();

  map_event_key_to_event_idx_.clear();
  events_.clear();
  test_name_ = std::string(test_name);
}

void Profiler::unit_test_end() {
  std::vector<cudaStream_t> streams;
  std::vector<int> devices;
  std::vector<cudaEvent_t> stream_first_points;

  thread_local int current_device_id;
  CK_CUDA_THROW_(cudaGetDevice(&current_device_id));

  for (int i = 0; i < int(unit_test_labels_.size()); i++) {
    if (std::find(streams.begin(), streams.end(), unit_test_streams_[i]) == streams.end()) {
      streams.push_back(unit_test_streams_[i]);
      stream_first_points.push_back(unit_test_events_[i]);
    }
    if (std::find(devices.begin(), devices.end(), unit_test_devices_[i]) == devices.end()) {
      devices.push_back(unit_test_devices_[i]);
      CK_CUDA_THROW_(cudaSetDevice(unit_test_devices_[i]));
      cudaDeviceSynchronize();
    }
  }

  CK_CUDA_THROW_(cudaSetDevice(current_device_id));

  for (int i = 0; i < int(unit_test_labels_.size()); i++) {
    auto event_label = unit_test_labels_[i];
    auto stream = unit_test_streams_[i];
    auto device_id = unit_test_devices_[i];
    auto extra_info = unit_test_extra_infos_[i];

    int dot_pos = event_label.find_last_of(std::string("."));
    std::string event_type = event_label.substr(dot_pos + 1);
    if (event_type != "start" && event_type != "stop") {
      throw internal_runtime_error(
          HugeCTR::Error_t::UnspecificError,
          std::string("Invalid event name. Should end with .start or .stop"));
    }
    std::string event_name = event_label.substr(0, dot_pos);
    // above string operation cost 0.000xxx ms on DGXA100. x usually is 1 - 2.

    int met_times_within_this_stream;
    try {
      if (map_internal_.count(stream) <= 0) {
        map_internal_[stream] = std::make_shared<std::map<std::string, int>>();
      }
      met_times_within_this_stream = map_internal_[stream]->at(event_name);
    } catch (const std::out_of_range& e) {
      map_internal_[stream]->insert({event_name, 0});
      met_times_within_this_stream = 0;
    }
    std::string event_key = gen_event_key(event_name, stream, met_times_within_this_stream);
    int event_idx = find_event(event_key);

    if (event_type == "start") {
      if (event_idx >= 0) {
        // event exist!
        throw internal_runtime_error(HugeCTR::Error_t::UnspecificError,
                                     std::string("Event exist! Must be error in code."));
      }
      // create new event
      auto gpu_event = new GPUEvent;
      gpu_event->event_name = event_name;
      gpu_event->met_times_within_this_stream = met_times_within_this_stream;
      gpu_event->start_index = events_num_;
      gpu_event->end_index = -1;  // wait for stop event to set,
      gpu_event->measured_times_ms = std::vector<float>();
      gpu_event->iter_start_to_event_start_times_ms = std::vector<float>();
      gpu_event->device_id = device_id;
      gpu_event->stream = stream;
      if (!extra_info.empty()) {
        gpu_event->extra_infos_start.push_back(extra_info);
      }
      events_.push_back(std::shared_ptr<Event>(static_cast<Event*>(gpu_event)));
      events_num_++;
      map_event_key_to_event_idx_[event_key] = events_.size() - 1;
      // PROFILER_DEBUG_(std::string("Parsed a new GPU event ") + event_label + " occured_time " +
      // std::to_string(met_times_within_this_stream));
    } else {
      // event_name == "stop"
      // only update the end_index
      if (event_idx >= 0) {
        auto event = events_[event_idx];
        event->end_index = i;
        auto it =
            std::find(streams.begin(), streams.end(), static_cast<GPUEvent*>(event.get())->stream);
        int stream_index = std::distance(streams.begin(), it);
        if (stream_index < 0) {
          throw internal_runtime_error(HugeCTR::Error_t::UnspecificError,
                                       std::string("Some errors happened."));
        }
        float start_to_event_start_ms;
        float measured_time_ms;
        CK_CUDA_THROW_(cudaSetDevice(static_cast<GPUEvent*>(event.get())->device_id));
        CK_CUDA_THROW_(cudaEventElapsedTime(
            &measured_time_ms, unit_test_events_[event->start_index], unit_test_events_[i]));
        CK_CUDA_THROW_(cudaEventElapsedTime(&start_to_event_start_ms,
                                            stream_first_points[stream_index],
                                            unit_test_events_[event->start_index]));
        event->measured_times_ms.push_back(measured_time_ms);
        event->iter_start_to_event_start_times_ms.push_back(start_to_event_start_ms);
        if (!extra_info.empty()) {
          event->extra_infos_stop.push_back(extra_info);
        }
        events_num_++;
        map_internal_[stream]->operator[](event_name) = met_times_within_this_stream + 1;
        // PROFILER_DEBUG_(std::string("Parsed a new GPU event ") + event_label + " occured_time " +
        // std::to_string(met_times_within_this_stream));
      } else {
        throw internal_runtime_error(
            HugeCTR::Error_t::UnspecificError,
            std::string("Event ") + event_name + std::string(" has stop but no start"));
      }
    }
  }
  std::string file_path = host_name_ + "_" + test_name_ + ".prof.json";
  write_result(file_path.c_str());
}

int Profiler::event_met_times_within_stream(const char* event_name, cudaStream_t stream) {
  int met_times = 0;
  try {
    met_times = map_internal_.at(stream)->at(std::string(event_name));
  } catch (const std::out_of_range& e) {
  }
  return met_times;
}

int Profiler::find_event(std::string& event_key) {
  int idx = -1;
  try {
    idx = map_event_key_to_event_idx_.at(event_key);
  } catch (const std::out_of_range& e) {
  }
  return idx;
}

void Profiler::write_result(const char* file_path) {
  int ret = std::system((std::string("mkdir -p ") + profiling_dir).c_str());
  if (ret != 0) {
    MESSAGE_("Creating PROFILING_DIR failed?");
  }

  std::string result_file;
  if (file_path) {
    result_file = profiling_dir + '/' + file_path;
  } else {
    result_file = profiling_dir + '/' + host_name_ + ".prof.json";
  }

  MESSAGE_(std::string("Result json file is wrote in ") + result_file + ".");

  json result;
  result["host_name"] = host_name_;
  result["iter_time_ms"] = iter_time_ms_;
  result["events"] = json::array();
  result["runtime_data"] = json::array();
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
  for (auto& data : runtime_data_) {
    json j;
    j["data_name"] = data->data_name;
    j["device_id"] = data->device_id;
    j["stream"] = stream_str(data->stream);
    j["data"] = data->data;
    result["runtime_data"].push_back(j);
  }

  std::string result_jstring = result.dump();
  std::ofstream outfile;
  outfile.open(result_file.c_str());
  outfile << result_jstring;
  outfile.close();
}

// A global variable
Profiler global_profiler;

}  // namespace HugeCTR
