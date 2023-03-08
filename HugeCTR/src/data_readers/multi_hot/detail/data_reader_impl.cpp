/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cassert>
#include <data_readers/multi_hot/detail/data_reader_impl.hpp>
#include <filesystem>
#include <set>

namespace HugeCTR {
namespace MultiHot {

DataReaderImpl::DataReaderImpl(const std::vector<FileSource>& source_files,
                               const std::shared_ptr<ResourceManager>& resource_manager,
                               size_t batch_size, size_t num_reader_threads_per_device,
                               size_t num_batches_per_thread, bool shuffle, bool schedule_uploads)
    : resource_manager_(resource_manager), schedule_uploads_(schedule_uploads) {
  const size_t local_gpu_count = resource_manager->get_local_gpu_count();
  const size_t global_gpu_count = resource_manager->get_global_gpu_count();
  const size_t num_slots = source_files.size();

  for (auto source : source_files) {
    if (batch_size % global_gpu_count) {
      throw std::invalid_argument("Batch size not divisible by number of GPUs");
    }

    std::unique_ptr<IBatchLocations> locations = configure_locations(source, batch_size, shuffle);
    if (num_batches_ == 0) {
      num_batches_ = locations->count();
    } else if (num_batches_ != locations->count()) {
      throw std::invalid_argument("files do not contain the same number of batches");
    }

    // TODO: refactor this for dynamic pooling
    const size_t local_batch_size_bytes =
        (batch_size / global_gpu_count) * source.sample_size_bytes;

    auto device_locations = locations->shard(global_gpu_count, local_batch_size_bytes);

    for (size_t i = 0; i < local_gpu_count; ++i) {
      // move thread to correct numa
      CudaCPUDeviceContext ctx(resource_manager->get_local_gpu(i)->get_device_id());

      int global_gpu_id = resource_manager->get_local_gpu(i)->get_global_id();
      assert(global_gpu_id < device_locations.size() && "Invalid global gpu id");
      auto thread_locations =
          device_locations[global_gpu_id]->distribute(num_reader_threads_per_device);

      for (size_t thread = 0; thread < thread_locations.size(); ++thread) {
        auto reader = new BatchFileReader(source.name, source.slot_id, num_batches_per_thread,
                                          std::move(thread_locations[thread]));
        file_readers_[i].emplace_back(reader);
      }
    }
  }

  size_t num_inflight_batches =
      std::min(num_reader_threads_per_device * num_batches_per_thread, num_batches_);

  // Init batches
  batch_buffers_.resize(num_inflight_batches);
  for (size_t i = 0; i < batch_buffers_.size(); ++i) {
    auto batch = std::make_unique<Batch>();

    batch->id = -1;  // Invalid
    batch->total_ios = local_gpu_count * num_slots;
    batch->state = BatchState::NOT_READY;
    batch->num_completed_io = {0};
    batch->num_completed_uploads = {0};
    batch->in_use_count = {resource_manager->get_local_gpu_count()};

    batch->local_batches.resize(resource_manager->get_local_gpu_count());
    size_t gpu = 0;
    for (auto& local_batch : batch->local_batches) {
      CudaDeviceContext ctx(resource_manager->get_local_gpu(gpu)->get_device_id());

      local_batch.io_batches.resize(num_slots);
      local_batch.device_transfers.resize(num_slots);
      local_batch.num_transfers = 0;

      // Allocate buffer for each slot
      for (auto source : source_files) {
        uint8_t* ptr = nullptr;
        size_t local_batch_size_bytes = (batch_size / global_gpu_count) * source.sample_size_bytes;
        HCTR_LIB_THROW(cudaMalloc(&ptr, local_batch_size_bytes));
        HCTR_LIB_THROW(cudaMemset(ptr, 0, local_batch_size_bytes));
        local_batch.device_data.push_back(ptr);
      }

      gpu++;
    }

    batch_buffers_[i] = std::move(batch);
  }

  pending_transfers_.resize(resource_manager->get_local_gpu_count());

  for (size_t i = 0; i < resource_manager->get_local_gpu_count(); ++i) {
    CudaCPUDeviceContext ctx(resource_manager->get_local_gpu(i)->get_device_id());
    cudaStream_t stream;
    // Needs to be highest priority to ensure transfers get executed as soon as they are unblocked
    HCTR_LIB_THROW(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -100));
    placement_streams_.push_back(stream);

    cudaEvent_t event;
    HCTR_LIB_THROW(cudaEventCreate(&event));
    placement_events_.push_back(event);
  }
}

DataReaderImpl::~DataReaderImpl() {
  running_ = false;
  for (auto& thread : file_reader_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  for (auto& thread : placement_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  std::cout << "DataReaderImpl exit!" << std::endl;
  std::cout << std::fixed << "Batch latency (min/max/avg) (s): (" << io_stats.batch_min_latency
            << ", " << io_stats.batch_max_latency << ", " << io_stats.batch_avg_latency << ")\n";
  // TODO: free GPU mem
}

std::unique_ptr<IBatchLocations> DataReaderImpl::configure_locations(FileSource source,
                                                                     size_t batch_size,
                                                                     bool shuffle) const {
  const size_t file_size = std::filesystem::file_size(source.name);
  assert(file_size > 0);

  auto locations = std::make_unique<BatchLocations>(
      batch_size * source.sample_size_bytes, 0, file_size, shuffle,
      resource_manager_->get_local_cpu()->get_replica_uniform_seed());
  return locations;
}

void DataReaderImpl::start() {
  running_ = true;
  for (const auto& entry : file_readers_) {
    int device_id = entry.first;
    for (const auto& file_reader : entry.second) {
      file_reader_threads_.emplace_back(&DataReaderImpl::read_batches, this,
                                        std::ref(*file_reader.get()), device_id);
    }
  }

#ifndef BENCH_IO
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); ++i) {
    placement_threads_.emplace_back(&DataReaderImpl::upload_batches, this, i);
  }
#endif

  // Make sure first batch is ready to consume
  while (batch_buffers_[0]->state != BatchState::READY_TO_CONSUME) {
    // spin
  }
}

const DataReaderImpl::Batch& DataReaderImpl::get_batch() {
  const size_t buf_pos = batch_i_ % batch_buffers_.size();

  Batch* batch = batch_buffers_[buf_pos].get();

  // needs to be set to NOT_READY on calling thread, not callback thread, otherwise there will be
  // race condition where CPU runs ahead and the next batch could be ready to consume from the
  // previous iteration.
  while (batch->state.load(std::memory_order_acquire) != BatchState::READY_TO_CONSUME) {
    // spin
  }
  batch->state = BatchState::NOT_READY;

  compute_batch_stats(batch);

  batch_i_ = (batch_i_ + 1) % num_batches_;

  last_batch_ = batch;
  return *last_batch_;
}

void DataReaderImpl::device_release_last_batch_here(cudaStream_t stream) const {
  HCTR_LIB_THROW(cudaStreamAddCallback(stream, &DataReaderImpl::release_batch_callback,
                                       (void*)last_batch_, 0));
}

void DataReaderImpl::schedule_upload_here(int raw_device_id, cudaStream_t stream, bool from_graph) {
  if (schedule_uploads_) {
    unsigned int flags = from_graph ? cudaEventRecordExternal : 0;
    HCTR_LIB_THROW(cudaEventRecordWithFlags(placement_events_[raw_device_id], stream, flags));
  }
}

void DataReaderImpl::upload_notify(int raw_device_id) {
  if (schedule_uploads_) {
    pending_transfers_[raw_device_id].raw++;
  }
}

size_t DataReaderImpl::get_total_inflight_batches() const { return batch_buffers_.size(); }

// QUEUE_SIZE (num_inflight_batches): 4

// BATCH_IDS READER A (numa 0):   0 1 2 3 4 0 1 2 3 4
// QUEUE_IDS READER A (numa 0):   0 1 2 3 0 0 1 2 3
//                                        ^

// BATCH_IDS READER B (numa 1):   0 1 2 3 0 1 2 3
// QUEUE_IDS READER B (numa 1):   0 1 2 3 0 1 2 3
//                                        ^
//                                        This is bad! READER A batch(4), and READER B batch(0)
//                                        can be inflight at the same time causing race condition
//                                        because they fall in the same queue id slot.

DataReaderImpl::Batch& DataReaderImpl::get_parent(size_t batch_i) {
  return *batch_buffers_[batch_i % batch_buffers_.size()].get();
}

void DataReaderImpl::read_batches(BatchFileReader& file_reader, int device_id) {
  CudaCPUDeviceContext ctx(device_id);  // move thread to appropriate numa

  while (running_) {
    const std::vector<const BatchFileReader::Batch*>& io_batches = file_reader.read_batches(10);
    for (const auto io_batch : io_batches) {
      Batch& batch = get_parent(io_batch->batch_i);
      auto& local_batch = batch.local_batches[device_id];

      local_batch.io_batches[io_batch->slot_id] = io_batch;

      DeviceTransfer* transfer = nullptr;
      if (io_batch->shard_size_bytes > 0)  // incomplete batch may not have local batch on all GPUs
      {
        transfer = new DeviceTransfer(device_id,
                                      io_batch->data,                              // src
                                      local_batch.device_data[io_batch->slot_id],  // dst
                                      io_batch->shard_size_bytes);
      }

      size_t buf_idx = local_batch.num_transfers++;  // atomic
      local_batch.device_transfers[buf_idx] = transfer;

      if (++batch.num_completed_io == batch.total_ios) {  // All IOs complete

#ifdef BENCH_IO
        const bool batch_uploaded = true;
#else
        /// TODO
        const bool batch_uploaded = false;  // batch.id == local_batch.io_batches.at(0)->batch_id;
#endif
        if (batch_uploaded) {
          // unblock upload threads and allow them to move into next batch
          // TODO: batch.state.store(BatchState::READY_TO_UPLOAD, std::memory_order_release);
        } else {
          // Batch ID will be the same across all devices and all io_batches
          batch.id = local_batch.io_batches[0]->batch_id;

          // Release fence, and perform atomic write to indicate that the batch is ready to upload
          // we need a fence to ensure previous writes are visible (i.e device transfers)
          batch.state.store(BatchState::READY_TO_UPLOAD, std::memory_order_release);
        }
      }
    }
  }
}

void DataReaderImpl::upload_batches(size_t device_id) {
  // move thread to correct numa
  CudaCPUDeviceContext ctx(resource_manager_->get_local_gpu(device_id)->get_device_id());

  // FIXME: Need to initialize in thread, otherwise cache coherency initialization problem.
  //  placement_barriers_[device_i].sem_[0] = 1;
  pending_transfers_[device_id].raw =
      std::min(2ul, batch_buffers_.size());  // TODO: allocate based on credit

  cudaStream_t& stream = placement_streams_[device_id];

  size_t batch_i = 0;

  while (running_) {
    // Process uploads in order
    Batch* batch = batch_buffers_[batch_i % batch_buffers_.size()].get();
    auto& local_batch = batch->local_batches[device_id];

    // acquire so we guarantee all previous writes before memory_order_release of state are visible
    if (batch->state.load(std::memory_order_acquire) == BatchState::READY_TO_UPLOAD &&
        local_batch.num_transfers.raw.load(std::memory_order_relaxed) > 0) {
      // Schedule transfers at correct place in iteration
      if (schedule_uploads_) {
        while (pending_transfers_[device_id].raw == 0) {
          // spin
        }

        pending_transfers_[device_id].raw--;
        HCTR_LIB_THROW(cudaStreamWaitEvent(stream, placement_events_[device_id]));
      }

      // H2D for each slot
      size_t num_transfers = local_batch.num_transfers.raw;
      assert(num_transfers <= local_batch.device_transfers.size());
      for (size_t i = 0; i < num_transfers; ++i) {
        DeviceTransfer* transfer = local_batch.device_transfers[i];
        if (transfer) {  // might be empty on incomplete batch
          transfer->execute(stream);
          delete transfer;
        }
      }

      // It's possible that this local batch has been uploaded but the other local batches haven't.
      // In this case we will attempt to upload again because state == READY_TO_UPLOAD. Therefore,
      // set num_transfers to 0 to prevent uploading this local batch again.
      local_batch.num_transfers = 0;

      // necessary for decrement of num_completed_uploads because it is checked on the host
      HCTR_LIB_THROW(cudaStreamSynchronize(stream));

      // all devices have uploaded their local batch, main thread can now consume
      if (++batch->num_completed_uploads == batch->local_batches.size()) {
        batch->state.store(BatchState::READY_TO_CONSUME, std::memory_order_release);
      }

      batch_i = (batch_i + 1) % num_batches_;  // move to next batch
    } else {
      std::this_thread::yield();
    }
  }

  HCTR_LIB_THROW(cudaStreamDestroy(stream));
}

void DataReaderImpl::release_batch_callback(cudaStream_t stream, cudaError_t status,
                                            void* user_data) {
  auto batch = reinterpret_cast<Batch*>(user_data);
  if (--(batch->in_use_count) == 0) {
    batch->num_completed_io = 0;
    batch->num_completed_uploads = 0;
    batch->in_use_count = batch->local_batches.size();

    for (auto& local_batch : batch->local_batches) {
      for (auto& io_batch : local_batch.io_batches) {
        const_cast<BatchFileReader::Batch*>(io_batch)->release();
      }
    }
  }
}

void DataReaderImpl::compute_batch_stats(Batch* batch) {
  static uint64_t n = 0;
  static double batch_avg = 0.f;
  n++;

  auto running_average = [](uint64_t n, double old_avg, double new_value) {
    return old_avg * (n - 1) / n + (new_value / n);
  };

  double earliest_time = batch->local_batches[0].io_batches[0]->start_time;
  double latest_time = batch->local_batches[0].io_batches[0]->end_time;

  for (size_t i = 1; i < batch->local_batches.size(); ++i) {
    earliest_time = std::min(earliest_time, batch->local_batches[i].io_batches[0]->start_time);
    latest_time = std::max(latest_time, batch->local_batches[i].io_batches[0]->end_time);
  }

  double latency = latest_time - earliest_time;

  batch_avg = running_average(n, batch_avg, latency);
  io_stats.batch_min_latency = n == 1 ? latency : std::min(io_stats.batch_min_latency, latency);
  io_stats.batch_max_latency = n == 1 ? latency : std::max(io_stats.batch_max_latency, latency);
  io_stats.batch_avg_latency = batch_avg;
}

}  // namespace MultiHot
}  // namespace HugeCTR
