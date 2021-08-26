#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include <chrono>
#include <common.hpp>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#if CUDA_VERSION < 11010
// this won't work for cuda graph, just to pass the compiling
#define CUDA_GRAPH_EVENT_RECORD(...) cudaEventRecord(__VA_ARGS__)
#else
#define CUDA_GRAPH_EVENT_RECORD(...) cudaEventRecordWithFlags(__VA_ARGS__, cudaEventRecordExternal)
#endif

#ifdef ENABLE_PROFILING
#define INITIALIZE_PROFILER(...)                                     \
  do {                                                               \
    if (HugeCTR::global_profiling_mode == 0) {                       \
      HugeCTR::global_fine_grained_profiler.initialize(__VA_ARGS__); \
    } else if (HugeCTR::global_profiling_mode == 1) {                \
      HugeCTR::global_one_shot_profiler.initialize(__VA_ARGS__);     \
      HugeCTR::global_data_reader_one_shot_profiler.initialize();    \
    } else if (HugeCTR::global_profiling_mode == 2) {                \
      HugeCTR::global_unit_test_profiler.initialize(__VA_ARGS__);    \
    }                                                                \
    HugeCTR::global_data_profiler.initialize(__VA_ARGS__);           \
  } while (0)

#define PROFILE_RECORD(...)                                            \
  do {                                                                 \
    if (HugeCTR::global_profiling_mode == 0) {                         \
      HugeCTR::global_fine_grained_profiler.record_event(__VA_ARGS__); \
    } else if (HugeCTR::global_profiling_mode == 1) {                  \
      HugeCTR::global_one_shot_profiler.record_event(__VA_ARGS__);     \
    } else if (HugeCTR::global_profiling_mode == 2) {                  \
      HugeCTR::global_unit_test_profiler.record_event(__VA_ARGS__);    \
    } else {                                                           \
      std::string msg("Invalid global_profiling_mode");                \
      ERROR_MESSAGE_(msg);                                             \
      throw std::invalid_argument(msg);                                \
    }                                                                  \
  } while (0)

#define PROFILE_RECORD_DATA(...) global_data_profiler.record_data(__VA_ARGS__);

#define PROFILE_RECORD_DATA_READER(...) \
  global_data_reader_one_shot_profiler.record_event(__VA_ARGS__);

#define PROFILE_UNIT_TEST_START(...)                                 \
  do {                                                               \
    HugeCTR::global_unit_test_profiler.unit_test_start(__VA_ARGS__); \
  } while (0)

#define PROFILE_UNIT_TEST_STOP(...)                     \
  do {                                                  \
    HugeCTR::global_unit_test_profiler.unit_test_end(); \
  } while (0)

#define PROFILER_ITER_CHECK(...)                                 \
  do {                                                           \
    if (HugeCTR::global_profiling_mode == 0) {                   \
      HugeCTR::global_fine_grained_profiler.iter_check();        \
    } else if (HugeCTR::global_profiling_mode == 1) {            \
      HugeCTR::global_one_shot_profiler.iter_check(__VA_ARGS__); \
    } else {                                                     \
      std::string msg("Invalid global_profiling_mode");          \
      ERROR_MESSAGE_(msg);                                       \
      throw std::invalid_argument(msg);                          \
    }                                                            \
  } while (0)

#else

#define INITIALIZE_PROFILER(...) \
  do {                           \
  } while (0)

#define PROFILE_RECORD(...) \
  do {                      \
  } while (0)

#define PROFILE_RECORD_DATA(...) \
  do {                           \
  } while (0)

#define PROFILE_RECORD_DATA_READER(...) \
  do {                                  \
  } while (0)

#define PROFILE_UNIT_TEST_START(...) \
  do {                               \
  } while (0)

#define PROFILE_UNIT_TEST_STOP(...) \
  do {                              \
  } while (0)

#define PROFILER_ITER_CHECK(...) \
  do {                           \
  } while (0)

#endif

#define PROFILER_DEBUG_(msg)                                                                    \
  do {                                                                                          \
    MESSAGE_(std::string(msg) + " on thread " + std::to_string(omp_get_thread_num()) +          \
             ", on stream " + stream_str(stream) + ", on device " + std::to_string(device_id) + \
             ", iter " + std::to_string(current_iteration_));                                   \
  } while (0)

namespace HugeCTR {
namespace Profiler {

struct Event {
  std::string event_name;
  int start_index;
  int end_index;
  std::vector<float> iter_start_to_event_start_times_ms;
  std::vector<float> measured_times_ms;
  std::vector<std::string> extra_infos_start;
  std::vector<std::string> extra_infos_stop;
};

struct GPUEvent : Event {
  int device_id;
  int met_times_within_this_stream;
  cudaStream_t stream;
};

struct RuntimeData {
  std::string data_name;
  int device_id;
  cudaStream_t stream;
  std::vector<std::string> data;
};

class GPUTimer {
 public:
  cudaEvent_t start_;
  cudaEvent_t stop_;
  cudaEvent_t iter_start_;
  std::string extra_info_start;
  std::string extra_info_stop;

  GPUTimer() {
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&iter_start_, cudaEventBlockingSync));
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&start_, cudaEventBlockingSync));
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&stop_, cudaEventBlockingSync));
  }
  ~GPUTimer() {
    cudaEventDestroy(iter_start_);
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  void iter_start(cudaStream_t stream) { CK_CUDA_THROW_(cudaEventRecord(iter_start_, stream)); };

  void event_start(cudaStream_t stream, bool could_be_in_cuda_graph) {
    if (could_be_in_cuda_graph) {
      cudaStreamCaptureStatus* capture_status;
      CK_CUDA_THROW_(cudaStreamIsCapturing(stream, &capture_status));
      if (*capture_status == cudaStreamCaptureStatusActive) {
        CK_CUDA_THROW_(CUDA_GRAPH_EVENT_RECORD(start_, stream));
        return;
      }
    }
    CK_CUDA_THROW_(cudaEventRecord(start_, stream));
  };

  void event_stop(cudaStream_t stream, bool could_be_in_cuda_graph) {
    if (could_be_in_cuda_graph) {
      cudaStreamCaptureStatus* capture_status;
      CK_CUDA_THROW_(cudaStreamIsCapturing(stream, &capture_status));
      if (*capture_status == cudaStreamCaptureStatusActive) {
        CK_CUDA_THROW_(CUDA_GRAPH_EVENT_RECORD(stop_, stream));
        return;
      }
    }
    CK_CUDA_THROW_(cudaEventRecord(stop_, stream));
  };

  float get_measured_time_ms() {
    float measured_time_ms;
    CK_CUDA_THROW_(cudaEventElapsedTime(&measured_time_ms, start_, stop_));
    return measured_time_ms;
  };

  float get_iter_start_to_event_start_ms() {
    float iter_start_to_event_start_ms;
    CK_CUDA_THROW_(cudaEventElapsedTime(&iter_start_to_event_start_ms, iter_start_, start_));
    return iter_start_to_event_start_ms;
  };
};

class BaseProfiler {
 protected:
  std::string host_name_;
  bool use_cuda_graph_;
  bool exit_when_finished_;
  int warmup_iterations_;
  int warmup_after_cudagraph_reinit_;
  int current_iteration_;

  std::vector<float> iter_time_ms_;
  std::chrono::time_point<std::chrono::steady_clock> iter_check_;

  std::vector<std::string> interested_events_;
  std::vector<std::shared_ptr<Event>> events_;
  int events_num_;

  std::map<std::string, int> map_event_key_to_event_idx_;
  std::map<cudaStream_t, std::shared_ptr<std::map<std::string, int>>> map_internal_;
  // for thread safe
  std::mutex mtx_;

 public:
  std::string profiling_dir;
  bool init_cuda_graph_this_iter;

  void initialize(bool use_cuda_graph, bool exit_when_finished = true);
  int access_or_insert_in_event_met_times_in_stream(cudaStream_t stream,
                                                    const std::string& event_name);
  int event_met_times_within_stream_safe(cudaStream_t stream, const std::string& event_name);
  int find_event(const std::string& event_key);
  bool find_in_interested_events(const std::string& event_name);
  bool try_create_one_gpu_event(const std::string& event_name, const std::string& event_type,
                                int device_id, cudaStream_t stream);
  void sync_all_gpus(int gpus);
  void clear_map_interal();
  static std::string stream_str(cudaStream_t stream);
  static std::string gen_event_key(const std::string& event_name, cudaStream_t stream,
                                   int met_times_within_this_stream);
  static std::string gpu_event_strfy(Event* event);
  static std::pair<std::string, std::string> get_event_name_and_type(const char* event_label_char);
};

class FineGrainedProfiler : public BaseProfiler {
 private:
  int repeat_times_;
  int current_reapted_times_;
  int current_event_idx_;

  std::map<cudaStream_t, std::shared_ptr<GPUTimer>> map_stream_to_gpu_timer_;

 public:
  void initialize(bool use_cuda_graph, bool exit_when_finished = true);
  void record_event(const char* event_label_char, cudaStream_t stream,
                    bool could_be_in_cuda_graph = true, int device_id = -1,
                    const std::string& extra_info = std::string());
  bool iter_check();
  void prepare_iter_start();
  void write_result(const char* file_path = nullptr);
};

class OneShotProfiler : public BaseProfiler {
 private:
  int repeat_iters_;
  int record_every_n_;
  std::map<int, std::shared_ptr<GPUTimer>> map_device_id_to_iter_gpu_timer_;

  std::map<std::string, std::shared_ptr<GPUTimer>> map_event_key_to_gpu_timer_;
  std::map<cudaStream_t, cudaEvent_t> map_stream_to_iter_start_cuevent_;

 public:
  void initialize(bool use_cuda_graph, bool exit_when_finished = true);
  bool iter_check(int gpus);
  void record_event(const char* event_label_char, cudaStream_t stream,
                    bool could_be_in_cuda_graph = true, int device_id = -1,
                    const std::string& extra_info = std::string());
  void prepare_iter_start();
  cudaEvent_t find_iter_start_event(int device_id);
  void determine_init_cuda_graph();
  void write_result(const char* file_path = nullptr);
};

class DataProfiler : public BaseProfiler {
 private:
 public:
  void initialize(bool use_cuda_graph, bool exit_when_finished = true);
  bool record_data(const char* data_label_char, cudaStream_t stream,
                   const std::string& data = std::string(), int device_id = -1);
};

class UnitTestProfiler : public BaseProfiler {
 private:
 public:
  void initialize(bool use_cuda_graph, bool exit_when_finished = true);
  void record_event(const char* event_label_char, cudaStream_t stream,
                    bool could_be_in_cuda_graph = true, int device_id = -1,
                    const std::string& extra_info = std::string());
  bool iter_check();
};

class DataReaderOneShotProfiler : public BaseProfiler {
 private:
  int cuevent_per_event_;

 public:
  class StreamRecorder {
   private:
    int cuevent_per_event_;

   public:
    pthread_spinlock_t lock;
    cudaStream_t stream;
    int device_id;
    int current_h2d_idx;
    int current_p2p_idx;
    std::vector<cudaEvent_t> memcpy_h2d_cuevents;
    std::vector<cudaEvent_t> memcpy_p2p_cuevents;
    std::vector<std::vector<float>> iter_start_to_event_start_times_ms_h2d;
    std::vector<std::vector<float>> iter_start_to_event_start_times_ms_p2p;
    std::vector<std::vector<float>> measured_times_ms_h2d;
    std::vector<std::vector<float>> measured_times_ms_p2p;

    StreamRecorder(int cuevent_per_event, cudaStream_t s, int did);
    void record(const std::string& event_name, const std::string& event_type, cudaStream_t stream);
  };

  int phase;
  std::map<cudaStream_t, std::shared_ptr<StreamRecorder>> map_stream_to_stream_recorder;

  void initialize();
  void record_event(const char* event_label_char, cudaStream_t stream);
  void iter_check();
};

inline int set_and_keep_original_device(int target_device_id) {
  int original_device_id;
  CK_CUDA_THROW_(cudaGetDevice(&original_device_id));
  if (target_device_id < 0) {
    target_device_id = original_device_id;
  }
  if (original_device_id != target_device_id) {
    CK_CUDA_THROW_(cudaSetDevice(target_device_id));
  }
  return original_device_id;
}

inline void restore_original_device(const int original_device_id, const int current_device_id) {
  if (current_device_id != original_device_id) {
    CK_CUDA_THROW_(cudaSetDevice(original_device_id));
  }
}

}  //  namespace Profiler

// A global variables
extern const int global_profiling_mode;

extern Profiler::FineGrainedProfiler global_fine_grained_profiler;
extern Profiler::OneShotProfiler global_one_shot_profiler;
extern Profiler::DataProfiler global_data_profiler;
extern Profiler::UnitTestProfiler global_unit_test_profiler;
extern Profiler::DataReaderOneShotProfiler global_data_reader_one_shot_profiler;

bool profiler_init_cuda_graph_this_iter();

}  //  namespace HugeCTR
