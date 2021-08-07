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

DataReaderOneShotProfiler::StreamRecorder::StreamRecorder(int cuevent_per_event, cudaStream_t s) {
  current_h2d_idx = 0;
  current_p2p_idx = 0;
  cuevent_per_event_ = cuevent_per_event;
  stream = s;
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
  if (event_name == "memcpy_h2d") {
    if (event_type == "start") {
      pthread_spin_lock(&lock);
      CK_CUDA_THROW_(cudaEventRecord(memcpy_h2d_cuevents[current_h2d_idx], stream));
    } else {
      K_CUDA_THROW_(cudaEventRecord(memcpy_h2d_cuevents[current_h2d_idx + 1], stream));
      current_h2d_idx = (current_h2d_idx + 2) % (2 * cuevent_per_event_);
      pthread_spin_unlock(&lock);
    }
  } else if (event_name == "memcpy_p2p") {
    if (event_type == "start") {
      pthread_spin_lock(&lock);
      CK_CUDA_THROW_(cudaEventRecord(memcpy_p2p_cuevents[current_p2p_idx], stream));
    } else {
      K_CUDA_THROW_(cudaEventRecord(memcpy_p2p_cuevents[current_p2p_idx + 1], stream));
      current_h2d_idx = (current_p2p_idx + 2) % (2 * cuevent_per_event_);
      pthread_spin_unlock(&lock);
    }
  } else {
    throw internal_runtime_error(HugeCTR::Error_t::UnspecificError,
                                 "DataReaderOneShotProfiler unsupported event name.");
  }
}
else {
}
CK_CUDA_THROW_(cudaEventRecord(memcpy_h2d_cuevents[current_h2d_idx], stream));
}

void DataReaderOneShotProfiler::initialize() {
  // according to data reader's code, there is one stream per device
  // for cuda memcopy p2p and htod, so assume that.
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
      try {
        map_stream_to_stream_recorder_[stream];
      } catch (const std::out_of_range& e) {
        map_stream_to_stream_recorder_[stream] = StreamRecorder(cuevent_per_event_, stream);
      }
      mtx_.unlock();
    } else if (phase == 2) {
      auto stream_recorder = map_stream_to_stream_recorder_[stream];
      auto name_and_type = get_event_name_and_type(event_label_char);
      auto event_name = std::get<0>(name_and_type);
      auto event_type = std::get<1>(name_and_type);
      stream_recorder->record(event_name, event_type, )
    } else {
      throw internal_runtime_error(HugeCTR::Error_t::UnspecificError,
                                   "DataReaderOneShotProfiler wrong phase.");
    }
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }
}

bool DataReaderOneShotProfiler::iter_check() {}

}  //  namespace Profiler

Profiler::OneShotProfiler global_data_reader_one_shot_profiler;
}  //  namespace HugeCTR