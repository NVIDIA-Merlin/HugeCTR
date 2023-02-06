#include "data_readers/multi_hot/detail/data_reader_impl.hpp"

#include <cassert>
#include <filesystem>
#include <set>

namespace HugeCTR {
namespace MultiHot {

DataReaderImpl::DataReaderImpl(const std::vector<FileSource>& data_files,
                               const std::shared_ptr<ResourceManager>& resource_manager,
                               size_t batch_size, size_t num_reader_threads_per_device,
                               size_t num_batches_per_thread, bool shuffle, bool schedule_uploads)
    : resource_manager_(resource_manager), schedule_uploads_(schedule_uploads) {
  size_t ios_per_batch = 0;
  size_t local_gpu_count = resource_manager->get_local_gpu_count();

  for (auto data_file : data_files) {
    if (batch_size % local_gpu_count) {
      throw std::invalid_argument("Batch size not divisible by number of GPUs");
    }

    std::unique_ptr<IBatchLocations> locations =
        configure_locations(data_file, batch_size, shuffle);
    if (num_batches_ == 0) {
      num_batches_ = locations->count();
    } else if (num_batches_ != locations->count()) {
      throw std::invalid_argument("files do not contain the same number of batches");
    }

    size_t batch_size_bytes_per_device =
        (batch_size / local_gpu_count) * data_file.sample_size_bytes;

    auto device_locations = locations->shard(local_gpu_count, batch_size_bytes_per_device);
    ios_per_batch += device_locations.size();
    for (size_t i = 0; i < device_locations.size(); ++i) {
      CudaCPUDeviceContext ctx(
          resource_manager->get_local_gpu(i)->get_device_id());  // moves thread to correct numa
      auto thread_locations = device_locations[i]->distribute(num_reader_threads_per_device);

      for (size_t thread = 0; thread < thread_locations.size(); ++thread) {
        auto reader = new BatchFileReader(data_file.name, data_file.slot_id, num_batches_per_thread,
                                          std::move(thread_locations[thread]));
        file_readers_[i].emplace_back(reader);
      }
    }
  }

  // queue depth will be same across all devices so only query one of the devices
  const auto& device_0_readers = file_readers_.begin()->second;
  const size_t num_inflight_batches =
      std::accumulate(device_0_readers.begin(), device_0_readers.end(), 0,
                      [](size_t sum, auto& fr) { return sum + fr->get_queue_depth(); });

  // Init batches
  batch_buffers_.resize(num_inflight_batches);
  for (size_t i = 0; i < batch_buffers_.size(); ++i) {
    auto batch = std::make_unique<Batch>();

    batch->id = -1;  // Invalid
    batch->ready_to_upload = false;
    batch->ready_to_consume = SystemLatch(resource_manager->get_local_gpu_count());
    batch->total_ios = ios_per_batch;
    batch->num_completed_io = {0};
    batch->in_use_count = {resource_manager->get_local_gpu_count()};

    batch->local_batches.resize(resource_manager->get_local_gpu_count());
    size_t gpu = 0;
    for (auto& local_batch : batch->local_batches) {
      CudaDeviceContext ctx(resource_manager->get_local_gpu(gpu)->get_device_id());

      local_batch.io_batches.resize(data_files.size());
      local_batch.device_transfers.reserve(data_files.size());

      // Allocate buffer for each slot
      for (auto data_file : data_files) {
        uint8_t* ptr = nullptr;
        size_t batch_size_bytes_per_dev =
            (batch_size / local_gpu_count) * data_file.sample_size_bytes;
        HCTR_LIB_THROW(cudaMalloc(&ptr, batch_size_bytes_per_dev));
        HCTR_LIB_THROW(cudaMemset(ptr, 0, batch_size_bytes_per_dev));
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

std::unique_ptr<IBatchLocations> DataReaderImpl::configure_locations(FileSource data_file,
                                                                     size_t batch_size,
                                                                     bool shuffle) const {
  const size_t file_size = std::filesystem::file_size(data_file.name);
  assert(file_size > 0);

  auto locations = std::make_unique<BatchLocations>(
      batch_size * data_file.sample_size_bytes, 0, file_size, shuffle,
      resource_manager_->get_local_cpu()->get_replica_uniform_seed());
  return locations;
}

void DataReaderImpl::start() {
  running_ = true;
  for (const auto& entry : file_readers_) {
    int device = entry.first;
    for (const auto& file_reader : entry.second) {
      file_reader_threads_.emplace_back(&DataReaderImpl::read_batches, this,
                                        std::ref(*file_reader.get()), device);
    }
  }

#ifndef BENCH_IO
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); ++i) {
    placement_threads_.emplace_back(&DataReaderImpl::place_batches, this, i);
  }
#endif

  // Make sure first batch is ready to consume
  batch_buffers_[0]->ready_to_consume.wait();

  printf("finish start\n");
}

const DataReaderImpl::Batch& DataReaderImpl::get_batch() {
  const size_t buf_pos = batch_i_ % batch_buffers_.size();

  Batch* batch = batch_buffers_[buf_pos].get();

  batch->ready_to_consume.wait();
  // needs to be reset on calling thread, not callback thread, otherwise there will be race
  // condition where CPU runs ahead and the next batch could be ready to consume from the previous
  // iteration.
  batch->ready_to_consume.reset();

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

      // thread safe assigment
      const size_t num_completed_ios = ++batch.num_completed_io;
      batch.local_batches[device_id].io_batches[io_batch->slot_id] = io_batch;

      if (num_completed_ios == batch.total_ios) {  // All IOs complete

#ifdef BENCH_IO
        const bool batch_uploaded = true;
#else
        const bool batch_uploaded =
            batch.id == batch.local_batches[device_id].io_batches.at(0)->batch_id;
#endif
        if (batch_uploaded) {
          // don't upload again
          batch.ready_to_consume.reset(1);
          batch.ready_to_consume.count_down();
        } else {
          // Batch ID will be the same across all devices and all io_batches
          batch.id = batch.local_batches[device_id].io_batches[0]->batch_id;

          size_t num_devices_to_transfer = 0;
          for (auto& local_batch : batch.local_batches) {
            // create transfer for each slot
            size_t idx = 0;
            for (auto& io_batch : local_batch.io_batches) {
              auto transfer = new DeviceTransfer(
                  device_id, io_batch->data, local_batch.device_data[idx], io_batch->size_bytes);
              local_batch.device_transfers.emplace_back(transfer);
              idx++;
            }
            num_devices_to_transfer += !local_batch.device_transfers.empty();
          }

          // Set the number of dependencies / devices to wait on
          batch.ready_to_consume.reset(num_devices_to_transfer);

          // Release fence, and perform atomic write to indicate that the batch is ready to upload
          // we need a fence to ensure the device transfers are visible (i.e not re-ordered)
          std::atomic_store_explicit(&batch.ready_to_upload, true, std::memory_order_release);
        }
      }
    }
  }
}

void DataReaderImpl::place_batches(size_t device_i) {
  // move thread to correct numa
  CudaCPUDeviceContext ctx(resource_manager_->get_local_gpu(device_i)->get_device_id());

  // FIXME: Need to initialize in thread, otherwise cache coherency initialization problem.
  //  placement_barriers_[device_i].sem_[0] = 1;
  pending_transfers_[device_i].raw =
      std::min(2ul, batch_buffers_.size());  // TODO: allocate based on credit

  cudaStream_t& stream = placement_streams_[device_i];

  size_t batch_i = 0;

  while (running_) {
    // Process uploads in order
    Batch* batch = batch_buffers_[batch_i % batch_buffers_.size()].get();

    if (batch->ready_to_upload.load(std::memory_order_relaxed)) {
      // guaranteed to observe everything done in the writer thread before the
      // atomic_store_explicit()
      std::atomic_thread_fence(std::memory_order_acquire);

      // Schedule transfers at correct place in iteration
      if (schedule_uploads_) {
        while (pending_transfers_[device_i].raw == 0) {
          // spin
        }

        pending_transfers_[device_i].raw--;
        HCTR_LIB_THROW(cudaStreamWaitEvent(stream, placement_events_[device_i]));
      }

      auto& transfers = batch->local_batches[device_i].device_transfers;
      if (!transfers.empty()) {
        // upload all slots
        for (DeviceTransfer* transfer : transfers) {
          transfer->execute(stream);
          delete transfer;
        }

        // reset so we don't iterate again and upload this batch
        transfers.clear();

        // necessary for barrier.can_acquire() because it is checked on the host
        HCTR_LIB_THROW(cudaStreamSynchronize(stream));
        // since we have streamSync we can use host count_down() instead of
        // device_count_down(stream)
        batch->ready_to_consume.count_down();
      }

      batch_i++;  // move to next batch
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
    batch->in_use_count.store(batch->local_batches.size());
    batch->ready_to_upload = false;

    for (auto& local_batch : batch->local_batches) {
      local_batch.device_transfers.clear();
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