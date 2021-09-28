#include "HugeCTR/include/embeddings/hybrid_embedding/stream_manager.hpp"
#include "HugeCTR/include/common.hpp"

namespace HugeCTR {

StreamManager::StreamManager(int num_devices) :
    stream_map(num_devices),
    event_map (num_devices) {
}

cudaStream_t &StreamManager::get_stream(uint32_t device_id, const std::string &key) {
  if (stream_map[device_id].find(key) == stream_map[device_id].end()) {
    cudaStream_t stream;
    CK_CUDA_THROW_(cudaStreamCreate(&stream));
    stream_map[device_id][key] = stream;
  }
  return stream_map[device_id][key];
}

cudaEvent_t &StreamManager::get_event(uint32_t device_id, const std::string &key) {
  if (event_map[device_id].find(key) == event_map[device_id].end()) {
    cudaEvent_t event;
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    event_map[device_id][key] = event;
  }
  return event_map[device_id][key];
}

StreamManager::~StreamManager() {
  for (auto &sm : stream_map) {
    for (auto &s : sm) {
      cudaStreamDestroy(s.second);
    }
  }
  for (auto &em : event_map) {
    for (auto &e : em) {
      cudaEventDestroy(e.second);
    }
  }
}

}
