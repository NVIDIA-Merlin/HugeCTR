#pragma once
#include <vector>

#include "HugeCTR/core/core.hpp"
#include "common.hpp"
#include "embedding_table.hpp"
#include "operators/operator.hpp"

// struct State {
//   enum class ScheduleHint {
//     H2D,
//     Compute,
//     Communication
//   };
//   ScheduleHint schedule_hint_;
//   std::optional<cudaEvent_t> event_; // device-device sync
//   volatile int *host_deivce_lock_; // device-host sync
//   std::atomic<int> host_lock_; // host-host sync
//   int remaining_stage_;
//   int iteration_;
//   bool is_train_;
//   int priority_;
// };

// using DataBag = std::vector<std::any>;
// class Pipelineable {
//  public:
//   virtual void prepare_data(DataBag &input_bag) = 0;
//   virtual State progress(std::optional<State> state) = 0;
//   virtual std::vector<State> get_pipeline() = 0;
// };

// // Per Device. Workable for datareader/embedding/layer/optimizer
// class Pipeline {

//   struct GraphWrapper {
//     bool initialized = false;
//     cudaGraph_t graph;
//     cudaGraphExec_t graph_exec;

//     void capture(std::function<void(cudaStream_t)> workload, cudaStream_t stream);
//     void exec(cudaStream_t stream);
//   };
//   GraphWrapper graph_;
//   using DataOperatorPair = std::pair<DataBag, Pipelineable*>;
//   std::vector<DataOperatorPair> pipelineable_list_;
//   std::unordered_map<Pipelineable*, Pipelineable*> dependency_;

//   Device device_;
//  public:
//   void construct_pipeline(Session &session);

//   void exec(State init_state);
// };