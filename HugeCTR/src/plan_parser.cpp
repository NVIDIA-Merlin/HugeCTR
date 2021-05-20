#include "gossip/include/plan_parser.hpp"

#include <assert.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "nlohmann/json.hpp"
using json = nlohmann::json;

using gpu_id_t = gossip::gpu_id_t;

gossip::transfer_plan_t parse_plan(const char* filename) {
  std::string type = "";
  gpu_id_t num_gpus = 0;
  gpu_id_t main_gpu = -1;
  size_t num_chunks = 0;
  std::vector<std::vector<gpu_id_t>> transfer_sequences = {};
  std::vector<size_t> transfer_sizes = {};

  std::ifstream ifs(filename);
  json json_plan;

  if (ifs.good())
    ifs >> json_plan;
  else {
    std::cerr << "error reading " << filename << std::endl;
    auto plan = gossip::transfer_plan_t{type, num_gpus, transfer_sequences};
    return plan;
  }

  // get plan from json
  auto it = json_plan.find("type");
  if (it != json_plan.end()) type = *it;

  it = json_plan.find("num_gpus");
  if (it != json_plan.end()) num_gpus = *it;

  it = json_plan.find("main_gpu");
  if (it != json_plan.end()) main_gpu = *it;

  it = json_plan.find("num_chunks");
  if (it != json_plan.end()) num_chunks = *it;

  it = json_plan.find("plan");
  if (it != json_plan.end())
    for (const auto& seq : *it) {
      transfer_sequences.push_back(seq);
      // TODO cut surplus items from seq
    }

  it = json_plan.find("chunks");
  if (it != json_plan.end())
    for (const auto& seq : *it) {
      transfer_sizes.push_back(seq);
    }

  auto plan =
      gossip::transfer_plan_t{type, num_gpus, transfer_sequences, num_chunks, transfer_sizes};

  plan.main_gpu(main_gpu);

  return plan;
}
