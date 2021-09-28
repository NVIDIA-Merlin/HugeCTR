#pragma once

#include <vector>
#include <unordered_map>
#include <string>

namespace HugeCTR {

class StreamManager {
  std::vector<std::unordered_map<std::string, cudaStream_t>> stream_map;
  std::vector<std::unordered_map<std::string, cudaEvent_t>> event_map;

public:
  StreamManager(int num_devices);
  cudaStream_t& get_stream(uint32_t device_id, const std::string& key);
  cudaEvent_t& get_event(uint32_t device_id, const std::string& key);
  
  ~StreamManager();
};

}
