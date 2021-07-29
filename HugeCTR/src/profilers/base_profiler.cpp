#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <profiler.hpp>

namespace HugeCTR {
namespace Profiler {

void BaseProfiler::initialize(bool use_cuda_graph, bool exit_when_finished) {
  std::string mode;
  if (HugeCTR::global_profiler_train_eval_mode == 0) {
    mode = "train";
  } else if (HugeCTR::global_profiler_train_eval_mode == 1) {
    mode = "eval";
  }
  MESSAGE_(std::string("Profiler using PROFILING_TRAIN_EVAL_MODE : ") + mode);

  char* pd = std::getenv("PROFILING_DIR");
  if (pd == NULL) {
    std::string msg(
        "Got empty for env PROFILING_DIR. You must specify if when using this profiler");
    MESSAGE_(msg);
    throw std::invalid_argument(msg);
  }
  profiling_dir = std::string(pd);
  MESSAGE_(std::string("Profiler using PROFILING_DIR: ") + profiling_dir);

  char* warmup_iterations_str = std::getenv("PROFILING_WARMUP_ITERS");
  if (warmup_iterations_str == NULL) {
    warmup_iterations_ = 11;
  } else {
    warmup_iterations_ = std::atoi(warmup_iterations_str) + 1;
  }
  MESSAGE_(std::string("Profiler using WARMUP_ITERS: ") + std::to_string(warmup_iterations_ - 1));

  char* warmup_after_cudagraph_reinit_str = std::getenv("PROFILING_WARMUP_AFTER_CUDAGRAPH_REINIT");
  if (warmup_after_cudagraph_reinit_str == NULL) {
    warmup_after_cudagraph_reinit_ = 10;
  } else {
    warmup_after_cudagraph_reinit_ = std::atoi(warmup_after_cudagraph_reinit_str);
  }
  if (use_cuda_graph) {
    MESSAGE_(std::string("Profiler using WARMUP_AFTER_CUDAGRAPH_REINIT: ") +
             std::to_string(warmup_after_cudagraph_reinit_));
  }

  char host_name[50];
  gethostname(host_name, 50);
  host_name_ = std::string(host_name);

  use_cuda_graph_ = use_cuda_graph;
  exit_when_finished_ = exit_when_finished;
  events_num_ = 0;
  current_iteration_ = 0;
  events_.clear();
  interested_events_.clear();
  iter_time_ms_.clear();
  map_event_key_to_event_idx_.clear();
  map_internal_.clear();

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
  }

  std::ifstream interested_events_file((profiling_dir + "/" + "prof.events").c_str());
  if (interested_events_file.good()) {
    MESSAGE_("Profiler using prof.events inside PROFILING_DIR");
    interested_events_.push_back("iteration");  // default event
    for (std::string line; getline(interested_events_file, line);) {
      interested_events_.push_back(line);
    }
  }
}

int BaseProfiler::access_or_insert_in_event_met_times_in_stream(cudaStream_t stream,
                                                                const std::string& event_name) {
  auto map_iter = map_internal_.find(stream);
  if (map_iter == map_internal_.end()) {
    map_internal_[stream] = std::make_shared<std::map<std::string, int>>();
  }
  int met_times_within_this_stream = 0;
  try {
    met_times_within_this_stream = map_internal_[stream]->at(event_name);
  } catch (const std::out_of_range& e) {
    map_internal_[stream]->insert({event_name, 0});
  }
  return met_times_within_this_stream;
}

int BaseProfiler::event_met_times_within_stream_safe(cudaStream_t stream,
                                                     const std::string& event_name) {
  int met_times = 0;
  try {
    met_times = map_internal_.at(stream)->at(event_name);
  } catch (const std::out_of_range& e) {
  }
  return met_times;
}  // namespace Profiler

int BaseProfiler::find_event(const std::string& event_key) {
  int idx = -1;
  try {
    idx = map_event_key_to_event_idx_.at(event_key);
  } catch (const std::out_of_range& e) {
  }
  return idx;
}

bool BaseProfiler::find_in_interested_events(const std::string& event_name) {
  if (interested_events_.empty()) {
    return true;
  }
  if (std::find(interested_events_.begin(), interested_events_.end(), event_name) ==
      interested_events_.end()) {
    return false;
  }
  return true;
}

bool BaseProfiler::try_create_one_gpu_event(const std::string& event_name,
                                            const std::string& event_type, int device_id,
                                            cudaStream_t stream) {
  if (!find_in_interested_events(event_name)) {
    return false;
  }

  int met_times_within_this_stream = map_internal_[stream]->at(event_name);
  std::string event_key = gen_event_key(event_name, stream, met_times_within_this_stream);
  int event_idx = find_event(event_key);

  if (event_type == "start") {
    if (event_idx >= 0) {
      // event exist!
      return false;
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
  } else {
    // event_name == "stop"
    // only update the end_index
    if (event_idx >= 0) {
      auto event = events_[event_idx];
      if (event->end_index < 0) {
        event->end_index = events_num_;
        events_num_++;
      }
      // PROFILER_DEBUG_(std::string("Parsed a new GPU event ") + event_label + " occured_time "
      // + std::to_string(met_times_within_this_stream));
    } else {
      throw internal_runtime_error(
          HugeCTR::Error_t::UnspecificError,
          std::string("Event ") + event_name + std::string(" has stop but no start"));
    }
  }
  return true;
}

void BaseProfiler::sync_all_gpus(int gpus) {
#pragma omp parallel for num_threads(gpus)
  for (int id = 0; id < gpus; id++) {
    CK_CUDA_THROW_(cudaSetDevice(id));
    CK_CUDA_THROW_(cudaDeviceSynchronize());
  }
}

void BaseProfiler::clear_map_interal() {
  for (auto& x : map_internal_) {
    for (auto it = x.second->begin(); it != x.second->end(); it++) {
      it->second = 0;
    }
  }
}

std::string BaseProfiler::stream_str(cudaStream_t stream) {
  const void* address = static_cast<const void*>(stream);
  std::stringstream ss;
  ss << address;
  return ss.str();
}

std::string BaseProfiler::gen_event_key(const std::string& event_name, cudaStream_t stream,
                                        int met_times_within_this_stream) {
  return event_name + "_" + stream_str(stream) + "_" + std::to_string(met_times_within_this_stream);
}

std::string BaseProfiler::gpu_event_strfy(Event* event) {
  GPUEvent* gpuevent = static_cast<GPUEvent*>(event);
  return std::string("Event name: ") + gpuevent->event_name +
         ". Met time: " + std::to_string(gpuevent->met_times_within_this_stream) +
         ". Device: " + std::to_string(gpuevent->device_id) +
         " . Stream: " + stream_str(gpuevent->stream);
}

std::pair<std::string, std::string> BaseProfiler::get_event_name_and_type(
    const char* event_label_char) {
  std::string event_label = std::string(event_label_char);
  int dot_pos = event_label.find_last_of(std::string("."));
  std::pair<std::string, std::string> name_and_type;

  name_and_type.second = event_label.substr(dot_pos + 1);
  if (name_and_type.second != "start" && name_and_type.second != "stop") {
    throw internal_runtime_error(
        HugeCTR::Error_t::UnspecificError,
        std::string("Invalid event name. Should end with .start or .stop"));
  }
  name_and_type.first = event_label.substr(0, dot_pos);
  return name_and_type;
}

int parse_train_eval_mode() {
  char* pd = std::getenv("PROFILING_TRAIN_EVAL_MODE");
  if (pd == NULL) {
    return 0;
  }
  int ret_mode = -1;
  std::string mode = std::string(pd);
  if (mode == "train") {
    ret_mode = 0;
  } else if (mode == "eval") {
    ret_mode = 1;
  } else {
    std::string msg("Invalid PROFILING_TRAIN_EVAL_MODE");
    throw std::invalid_argument(msg);
  }
  return ret_mode;
}

int parse_record_event_mode() {
#ifdef ENABLE_PROFILING
  char* pd = std::getenv("PROFILING_MODE");
  if (pd == NULL) {
    return 1;
  }
  int ret_mode = -1;
  std::string mode = std::string(pd);
  if (mode == "fine_grained") {
    ret_mode = 0;
  } else if (mode == "one_shot") {
    ret_mode = 1;
  } else if (mode == "unit_test") {
    ret_mode = 2;
  } else {
    std::string msg("Invalid PROFILING_MODE");
    throw std::invalid_argument(msg);
  }
  return ret_mode;
#else
  return 0;
#endif
}

}  //  namespace Profiler

bool profiler_init_cuda_graph_this_iter() {
  if (HugeCTR::global_profiling_mode == 0) {
    return HugeCTR::global_fine_grained_profiler.init_cuda_graph_this_iter;
  } else if (HugeCTR::global_profiling_mode == 1) {
    return HugeCTR::global_one_shot_profiler.init_cuda_graph_this_iter;
  } else {
    std::string msg("Not implemented yet!");
    ERROR_MESSAGE_(msg);
    throw std::invalid_argument(msg);
  }
}

const int global_profiler_train_eval_mode = Profiler::parse_train_eval_mode();
const int global_profiling_mode = Profiler::parse_record_event_mode();

}  //  namespace HugeCTR