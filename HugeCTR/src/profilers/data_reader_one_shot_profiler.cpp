#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <pthread.h>
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

DataReaderOneShotProfiler::StreamRecorder::StreamRecorder(int cuevent_per_event, cudaStream_t s,
                                                          int did) {
  current_h2d_idx = 0;
  current_p2p_idx = 0;
  cuevent_per_event_ = cuevent_per_event;
  stream = s;
  device_id = did;
  int ret = pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE);
  if (ret != 0) {
    throw internal_runtime_error(
        HugeCTR::Error_t::UnspecificError,
        "DataReaderOneShotProfiler::StreamRecorder spin lock creation error");
  }
  memcpy_h2d_cuevents.resize(2 * cuevent_per_event_);
  memcpy_p2p_cuevents.resize(2 * cuevent_per_event_);
  for (int i = 0; i < 2 * cuevent_per_event_; i++) {
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&memcpy_h2d_cuevents[i], cudaEventBlockingSync));
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&memcpy_p2p_cuevents[i], cudaEventBlockingSync));
  }
}

void DataReaderOneShotProfiler::StreamRecorder::record(const std::string& event_name,
                                                       const std::string& event_type,
                                                       cudaStream_t stream) {
  if (event_name == "data_reader_memcpy_h2d") {
    if (event_type == "start") {
      pthread_spin_lock(&lock);
      CK_CUDA_THROW_(cudaEventRecord(memcpy_h2d_cuevents[current_h2d_idx], stream));
    } else {
      CK_CUDA_THROW_(cudaEventRecord(memcpy_h2d_cuevents[current_h2d_idx + 1], stream));
      current_h2d_idx = (current_h2d_idx + 2) % (2 * cuevent_per_event_);
      pthread_spin_unlock(&lock);
    }
  } else if (event_name == "data_reader_memcpy_p2p") {
    if (event_type == "start") {
      pthread_spin_lock(&lock);
      CK_CUDA_THROW_(cudaEventRecord(memcpy_p2p_cuevents[current_p2p_idx], stream));
    } else {
      CK_CUDA_THROW_(cudaEventRecord(memcpy_p2p_cuevents[current_p2p_idx + 1], stream));
      current_p2p_idx = (current_p2p_idx + 2) % (2 * cuevent_per_event_);
      pthread_spin_unlock(&lock);
    }
  } else {
    throw internal_runtime_error(HugeCTR::Error_t::UnspecificError,
                                 "DataReaderOneShotProfiler unsupported event name.");
  }
}

void DataReaderOneShotProfiler::initialize() {
  // according to data reader's code, there is one stream per device
  // for cuda memcopy p2p and htod, so assume that.
  MESSAGE_("Profiler activate: DataReaderOneShotProfiler");
  char* cuevent_per_event_str = std::getenv("PROFILING_DATA_READER_ONE_SHOT_CUEVENT_NUM");
  if (cuevent_per_event_str == NULL) {
    cuevent_per_event_ = 5;
  } else {
    cuevent_per_event_ = std::atoi(cuevent_per_event_str);
  }
  MESSAGE_(std::string("Profiler using PROFILING_DATA_READER_ONE_SHOT_CUEVENT_NUM: ") +
           std::to_string(cuevent_per_event_));

  phase = 0;
}

void DataReaderOneShotProfiler::record_event(const char* event_label_char, cudaStream_t stream) {
  try {
    if (phase == 1) {
      mtx_.lock();
      if (map_stream_to_stream_recorder.find(stream) == map_stream_to_stream_recorder.end()) {
        int device_id;
        CK_CUDA_THROW_(cudaGetDevice(&device_id));
        map_stream_to_stream_recorder[stream] =
            std::make_shared<StreamRecorder>(cuevent_per_event_, stream, device_id);
      }
      mtx_.unlock();
    } else if (phase == 2) {
      auto recorder = map_stream_to_stream_recorder[stream];
      auto name_and_type = get_event_name_and_type(event_label_char);
      auto event_name = std::get<0>(name_and_type);
      auto event_type = std::get<1>(name_and_type);
      recorder->record(event_name, event_type, stream);
    } else if (phase != 0) {
      throw internal_runtime_error(HugeCTR::Error_t::UnspecificError,
                                   "DataReaderOneShotProfiler wrong phase.");
    }
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

void DataReaderOneShotProfiler::iter_check() {
  float measured_time_ms;
  float iter_start_to_event_start_time_ms;
  std::vector<float> measured_time_ms_h2d;
  std::vector<float> measured_time_ms_p2p;
  std::vector<float> iter_start_to_event_start_ms_h2d;
  std::vector<float> iter_start_to_event_start_ms_p2p;
  for (auto& x : map_stream_to_stream_recorder) {
    auto recorder = x.second;
    auto iter_start_event = global_one_shot_profiler.find_iter_start_event(recorder->device_id);
    measured_time_ms_h2d.clear();
    measured_time_ms_p2p.clear();
    iter_start_to_event_start_ms_h2d.clear();
    iter_start_to_event_start_ms_p2p.clear();

    for (int i = 0; i < cuevent_per_event_; i++) {
      try {
        CK_CUDA_THROW_(cudaEventElapsedTime(&iter_start_to_event_start_time_ms, iter_start_event,
                                            recorder->memcpy_h2d_cuevents[2 * i]));
        CK_CUDA_THROW_(cudaEventElapsedTime(&measured_time_ms, recorder->memcpy_h2d_cuevents[2 * i],
                                            recorder->memcpy_h2d_cuevents[2 * i + 1]));
        if (iter_start_to_event_start_time_ms > 0.0f) {
          iter_start_to_event_start_ms_h2d.push_back(iter_start_to_event_start_time_ms);
          measured_time_ms_h2d.push_back(measured_time_ms);
        }
      } catch (const std::runtime_error& rt_err) {
      }
      try {
        CK_CUDA_THROW_(cudaEventElapsedTime(&iter_start_to_event_start_time_ms, iter_start_event,
                                            recorder->memcpy_p2p_cuevents[2 * i]));
        CK_CUDA_THROW_(cudaEventElapsedTime(&measured_time_ms, recorder->memcpy_p2p_cuevents[2 * i],
                                            recorder->memcpy_p2p_cuevents[2 * i + 1]));
        if (iter_start_to_event_start_time_ms > 0.0f) {
          iter_start_to_event_start_ms_p2p.push_back(iter_start_to_event_start_time_ms);
          measured_time_ms_p2p.push_back(measured_time_ms);
        }
      } catch (const std::runtime_error& rt_err) {
      }
    }

    recorder->iter_start_to_event_start_times_ms_h2d.push_back(iter_start_to_event_start_ms_h2d);
    recorder->measured_times_ms_h2d.push_back(measured_time_ms_h2d);
    recorder->iter_start_to_event_start_times_ms_p2p.push_back(iter_start_to_event_start_ms_p2p);
    recorder->measured_times_ms_p2p.push_back(measured_time_ms_p2p);
  }
}

}  //  namespace Profiler

Profiler::DataReaderOneShotProfiler global_data_reader_one_shot_profiler;
}  //  namespace HugeCTR