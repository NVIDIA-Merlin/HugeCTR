/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <base/debug/logger.hpp>
#include <random>
#include <resource_managers/resource_manager_ext.hpp>
#include <utils.hpp>

namespace HugeCTR {

#ifdef ENABLE_MPI
class MPILifetimeService {
 private:
  MPILifetimeService() {
    HCTR_MPI_THROW(MPI_Initialized(&mpi_preinitialized_));
    if (mpi_preinitialized_) {
      HCTR_LOG(WARNING, ROOT,
               "MPI was already initialized somewhere elese. Lifetime service disabled.\n");
    } else {
      HCTR_MPI_THROW(MPI_Init(nullptr, nullptr));
      HCTR_LOG(INFO, ROOT, "MPI initialized from native backend.\n");
      HCTR_LOG(WARNING, ROOT,
               "\nMPI can only be initialized once per process. If HugeCTR is run on top of\n"
               "Python, unexpected things might happen if other other packages are used that rely\n"
               "on MPI. If you experience problems, try adding \"from mpi4py import MPI\" in your\n"
               "Python script before importing HugeCTR.\n");
    }
  }

  MPILifetimeService(const MPILifetimeService&) = delete;
  MPILifetimeService& operator=(const MPILifetimeService&) = delete;

 public:
  virtual ~MPILifetimeService() {
    if (!mpi_preinitialized_) {
      HCTR_MPI_THROW(MPI_Finalize());
      HCTR_LOG(INFO, ROOT, "MPI finalization done.\n");
    }
  }

  static void init() {
    static std::unique_ptr<MPILifetimeService> instance;
    static std::once_flag once_flag;
    std::call_once(once_flag, []() { instance.reset(new MPILifetimeService()); });
  }

 private:
  int mpi_preinitialized_ = 0;
};
#endif

std::shared_ptr<ResourceManager> ResourceManagerExt::create(
    const std::vector<std::vector<int>>& visible_devices, unsigned long long seed,
    DeviceMap::Layout layout) {
  int size = 1, rank = 0;

#ifdef ENABLE_MPI
  MPILifetimeService::init();
  HCTR_MPI_THROW(MPI_Comm_size(MPI_COMM_WORLD, &size));
  HCTR_MPI_THROW(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
#endif

  DeviceMap device_map(visible_devices, rank, layout);

  std::random_device rd;
  if (seed == 0) {
    seed = rd();
  }

#ifdef ENABLE_MPI
  HCTR_MPI_THROW(MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD));
#endif

  HCTR_LOG(INFO, ROOT, "Global seed is %llu\n", seed);

  std::shared_ptr<ResourceManager> core(
      new ResourceManagerCore(size, rank, std::move(device_map), seed));

  return std::shared_ptr<ResourceManager>(new ResourceManagerExt(core));
}

#ifdef ENABLE_MPI
void ResourceManagerExt::init_ib_comm() {
  int num_process = get_num_process();
  if (num_process > 1) {
    int process_id = get_process_id();
    ib_comm_ = std::make_unique<IbComm>();
    ib_comm_->init(num_process, get_local_gpu_count(), process_id, get_local_gpu_device_id_list());
  }
}
#endif

void ResourceManagerExt::set_ar_comm(AllReduceAlgo algo, bool use_mixed_precision) {
  int num_process = get_num_process();
#ifdef ENABLE_MPI
  IbComm* ib_comm_ptr = nullptr;
  if (algo == AllReduceAlgo::ONESHOT) {
    init_ib_comm();
    ib_comm_ptr = ib_comm_.get();
  }
  ar_comm_ = AllReduceInPlaceComm::create(num_process, algo, use_mixed_precision, get_local_gpus(),
                                          ib_comm_ptr);
#else
  ar_comm_ = AllReduceInPlaceComm::create(num_process, algo, use_mixed_precision, get_local_gpus());
#endif
}

}  // namespace HugeCTR
