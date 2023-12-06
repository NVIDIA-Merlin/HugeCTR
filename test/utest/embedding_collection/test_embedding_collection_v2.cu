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

#include <curand.h>
#include <curand_kernel.h>
#include <gtest/gtest.h>
#include <sys/time.h>

#include <core/hctr_impl/hctr_backend.hpp>
#include <core23/buffer_factory.hpp>
#include <embedding/embedding.hpp>
#include <embeddings/embedding_collection.hpp>
#include <filesystem>
#include <numeric>
#include <resource_managers/resource_manager_ext.hpp>
#include <utest/embedding_collection/configuration.hpp>
#include <utest/embedding_collection/reference_embedding.hpp>
#include <utils.cuh>
using namespace embedding;

template <typename KeyType, typename OffsetType, typename EmbType>
struct HostEmbeddingCollectionInput {
  std::vector<std::vector<KeyType>> h_dp_keys;
  std::vector<std::vector<OffsetType>> h_dp_bucket_range;
};

template <typename KeyType, typename OffsetType, typename EmbType>
void generate_input_from_file(
    const Configuration &config, const EmbeddingCollectionParam &ebc_param,
    HostEmbeddingCollectionInput<KeyType, OffsetType, EmbType> *host_ebc_input) {
  int batch_size_per_gpu = config.runtime_configuration.batch_size_per_gpu;

  HCTR_CHECK(config.input_data_configuration.fixed_hotness == true);
  const auto &raw_format_param = config.input_data_configuration.raw_format_param;

  size_t dense_bytes = sizeof(float) * (raw_format_param.dense_dim + raw_format_param.label_dim);
  size_t sparse_bytes = 0;
  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    auto &lookup_param = ebc_param.lookup_params[lookup_id];
    int max_hotness = lookup_param.max_hotness;

    sparse_bytes += max_hotness * sizeof(int);
  }

  size_t sample_bytes = dense_bytes + sparse_bytes;

  size_t file_size = std::filesystem::file_size(raw_format_param.input_file);

  HCTR_CHECK_HINT(file_size % sample_bytes == 0,
                  "file size %lu is not divisible by sample_bytes %lu", file_size, sample_bytes);
  size_t num_samples = file_size / sample_bytes;

  size_t sample_id = rand() % num_samples;

  std::ifstream input_fs(raw_format_param.input_file, std::ios::binary);
  input_fs.seekg(sample_id * sample_bytes);

  auto &dp_keys = host_ebc_input->h_dp_keys;
  auto &dp_bucket_range = host_ebc_input->h_dp_bucket_range;
  dp_keys.resize(ebc_param.num_lookup);
  dp_bucket_range.resize(ebc_param.num_lookup);
  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    dp_bucket_range[lookup_id].push_back(0);
  }

  std::vector<char> data_vec(sample_bytes);
  for (int b = 0; b < batch_size_per_gpu; ++b) {
    if (static_cast<size_t>(input_fs.tellg()) == file_size) {
      input_fs.seekg(0);
    }
    input_fs.read(data_vec.data(), sample_bytes);

    int *sparse_data = (int *)(data_vec.data() + dense_bytes);
    int i = 0;
    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      auto &lookup_param = ebc_param.lookup_params[lookup_id];
      int max_hotness = lookup_param.max_hotness;

      dp_bucket_range[lookup_id].push_back(max_hotness);
      for (int h = 0; h < max_hotness; ++h) {
        KeyType key = (KeyType)sparse_data[i];

        dp_keys[lookup_id].push_back(key);
        ++i;
      }
    }
  }
  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    std::inclusive_scan(dp_bucket_range[lookup_id].begin(), dp_bucket_range[lookup_id].end(),
                        dp_bucket_range[lookup_id].begin());
  }
}

template <typename KeyType, typename OffsetType, typename EmbType>
void generate_random_input(
    const Configuration &config, const EmbeddingCollectionParam &ebc_param,
    const std::vector<embedding::EmbeddingTableParam> &table_params,
    HostEmbeddingCollectionInput<KeyType, OffsetType, EmbType> *host_ebc_input) {
  int batch_size_per_gpu = config.runtime_configuration.batch_size_per_gpu;

  auto &dp_keys = host_ebc_input->h_dp_keys;
  auto &dp_bucket_range = host_ebc_input->h_dp_bucket_range;
  dp_keys.resize(ebc_param.num_lookup);
  dp_bucket_range.resize(ebc_param.num_lookup);
  for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
    auto &lookup_param = ebc_param.lookup_params[lookup_id];
    int table_id = lookup_param.table_id;
    int max_hotness = lookup_param.max_hotness;
    int64_t max_vocabulary_size = table_params[table_id].max_vocabulary_size;

    dp_bucket_range[lookup_id].push_back(0);

    for (int b = 0; b < batch_size_per_gpu; ++b) {
      int nnz = config.input_data_configuration.fixed_hotness ? max_hotness : rand() % max_hotness;
      dp_bucket_range[lookup_id].push_back(nnz);

      for (int i = 0; i < nnz; ++i) {
        KeyType key = rand() % max_vocabulary_size;
        dp_keys[lookup_id].push_back(key);
      }
    }
    std::inclusive_scan(dp_bucket_range[lookup_id].begin(), dp_bucket_range[lookup_id].end(),
                        dp_bucket_range[lookup_id].begin());
  }
}

template <typename KeyType, typename OffsetType, typename EmbType>
void generate_input(const Configuration &config, const EmbeddingCollectionParam &ebc_param,
                    const std::vector<embedding::EmbeddingTableParam> &table_params,
                    HostEmbeddingCollectionInput<KeyType, OffsetType, EmbType> *host_ebc_input) {
  if (config.input_data_configuration.input_data_type == InputDataType::Uniform) {
    generate_random_input<KeyType, OffsetType, EmbType>(config, ebc_param, table_params,
                                                        host_ebc_input);
  } else if (config.input_data_configuration.input_data_type == InputDataType::RawFormat) {
    generate_input_from_file<KeyType, OffsetType, EmbType>(config, ebc_param, host_ebc_input);
  } else {
    HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput, "input data type not supported");
  }
}

template <typename KeyType, typename OffsetType, typename EmbType>
void generate_host_embedding_collection_input(
    int global_gpu_id, const Configuration &config, const EmbeddingCollectionParam &ebc_param,
    const std::vector<embedding::EmbeddingTableParam> &table_params,
    HostEmbeddingCollectionInput<KeyType, OffsetType, EmbType> *host_ebc_input) {
  timeval t1;
  gettimeofday(&t1, NULL);
  srand(t1.tv_usec * t1.tv_sec * global_gpu_id);

  generate_input<KeyType, OffsetType, EmbType>(config, ebc_param, table_params, host_ebc_input);
}

void allocate_data_distributor_input(
    std::vector<std::shared_ptr<core::CoreResourceManager>> core_resource_manager_list,
    const EmbeddingCollectionParam &ebc_param, int num_global_gpus,
    std::vector<std::vector<core23::Tensor>> *sparse_dp_tensors,
    std::vector<std::vector<core23::Tensor>> *sparse_dp_bucket_range,
    std::vector<core23::Tensor> *ebc_top_grads, std::vector<core23::Tensor> *ebc_outptut) {
  int num_local_gpus = core_resource_manager_list.size();
  int batch_size_per_gpu = ebc_param.universal_batch_size / num_global_gpus;
  for (int gpu_id = 0; gpu_id < num_local_gpus; ++gpu_id) {
    HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());
    core23::Device device(core23::DeviceType::GPU,
                          core_resource_manager_list[gpu_id]->get_device_id());
    core23::TensorParams params = core23::TensorParams().device(device);

    std::vector<core23::Tensor> sparse_dp_tensors_on_current_gpu;
    std::vector<core23::Tensor> sparse_dp_bucket_range_on_current_gpu;
    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      auto &lookup_param = ebc_param.lookup_params[lookup_id];
      int max_hotness = lookup_param.max_hotness;
      sparse_dp_tensors_on_current_gpu.emplace_back(
          params.shape({batch_size_per_gpu, max_hotness}).data_type(ebc_param.key_type));
      sparse_dp_bucket_range_on_current_gpu.emplace_back(
          params.shape({batch_size_per_gpu + 1}).data_type(ebc_param.offset_type));
    }
    sparse_dp_tensors->push_back(sparse_dp_tensors_on_current_gpu);
    sparse_dp_bucket_range->push_back(sparse_dp_bucket_range_on_current_gpu);

    int64_t num_ev = 0;
    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
      auto &lookup_param = ebc_param.lookup_params[lookup_id];
      num_ev += (lookup_param.combiner == Combiner::Concat)
                    ? lookup_param.ev_size * lookup_param.max_hotness
                    : lookup_param.ev_size;
    }
    num_ev *= batch_size_per_gpu;
    ebc_top_grads->emplace_back(params.shape({num_ev}).data_type(ebc_param.emb_type));
    ebc_outptut->emplace_back(params.shape({num_ev}).data_type(ebc_param.emb_type));
  }
}

namespace {

template <typename EmbType>
__global__ void generate_random_decimal_kernel(curandState *state, EmbType *result, int num) {
  CUDA_1D_KERNEL_LOOP(i, num) {
    curand_init(1234, i, 0, &state[i]);
    float randf = curand_normal(state + i);
    randf -= 0.5f;
    randf *= 2;
    result[i] = HugeCTR::TypeConvertFunc<EmbType, float>::convert(randf);
  }
}
}  // namespace

template <typename EmbType>
void generate_device_top_grads(core23::Tensor &tensor) {
  curandState *d_state;
  HCTR_LIB_THROW(cudaMalloc(&d_state, sizeof(curandState) * tensor.num_elements()));
  generate_random_decimal_kernel<<<1024, 1024>>>(d_state, tensor.data<EmbType>(),
                                                 tensor.num_elements());
  HCTR_LIB_THROW(cudaFree(d_state));
}

std::vector<EmbeddingTableParam> get_table_params(const Configuration &config) {
  std::vector<EmbeddingTableParam> table_params;
  int table_id = 0;
  for (auto &embedding_config : config.embedding_config) {
    int ev_size = embedding_config.emb_vec_size;
    int max_vocabulary_size = embedding_config.max_vocabulary_size;
    for (int tid = 0; tid < embedding_config.num_table; ++tid) {
      table_params.emplace_back(table_id, max_vocabulary_size, ev_size, config.opt);
      table_id += 1;
    }
  }
  return table_params;
}

std::vector<LookupParam> get_lookup_params(const Configuration &config) {
  std::vector<LookupParam> lookup_params;
  int lookup_id = 0;
  int table_id = 0;
  for (auto &embedding_config : config.embedding_config) {
    int ev_size = embedding_config.emb_vec_size;
    for (int tid = 0; tid < embedding_config.num_table; ++tid) {
      for (auto max_hotness : embedding_config.max_hotness_list) {
        lookup_params.emplace_back(lookup_id, table_id, embedding_config.combiner, max_hotness,
                                   ev_size);
        lookup_id += 1;
      }
      table_id += 1;
    }
  }
  return lookup_params;
}

template <typename KeyType, typename OffsetType, typename IndexType, typename EmbType>
void embedding_collection_e2e(const Configuration &config) {
  auto table_param_list = get_table_params(config);
  int num_table = static_cast<int>(table_param_list.size());

  const auto &lookup_params = get_lookup_params(config);
  int num_lookup = static_cast<int>(lookup_params.size());
  const auto &shard_matrix = config.shard_configuration.shard_matrix;
  const auto &grouped_emb_params = config.shard_configuration.grouped_table_params;

  int batch_size_per_gpu = config.runtime_configuration.batch_size_per_gpu;
  int num_nodes = config.runtime_configuration.num_node;
  int num_local_gpus = config.runtime_configuration.num_gpus_per_node;
  int num_global_gpus = num_nodes * num_local_gpus;

  int batch_size = batch_size_per_gpu * num_global_gpus;
  int niters = config.niters;

  std::vector<int> device_list_per_node(num_local_gpus);
  std::iota(device_list_per_node.begin(), device_list_per_node.end(), 0);
  std::vector<std::vector<int>> device_list(num_nodes, device_list_per_node);

  auto resource_manager = HugeCTR::ResourceManagerExt::create(device_list, 0);

  std::vector<std::shared_ptr<core::CoreResourceManager>> core_resource_manager_list;
  for (int gpu_id = 0; gpu_id < num_local_gpus; ++gpu_id) {
    auto core = std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, gpu_id);

    core_resource_manager_list.push_back(core);
  }

  auto sync_gpus = [&]() {
    for (auto core : core_resource_manager_list) {
      CudaDeviceContext __(core->get_device_id());
      HCTR_LIB_THROW(cudaStreamSynchronize(core->get_local_gpu()->get_stream()));
    }
  };

  auto key_type = HugeCTR::core23::ToScalarType<KeyType>::value;
  auto index_type = HugeCTR::core23::ToScalarType<IndexType>::value;
  auto offset_type = HugeCTR::core23::ToScalarType<OffsetType>::value;
  auto emb_type = HugeCTR::core23::ToScalarType<EmbType>::value;
  auto wgrad_type = HugeCTR::core23::ToScalarType<EmbType>::value;

  std::vector<EmbeddingCollectionParam> ebc_params;
  for (int option_id = 0; option_id < static_cast<int>(config.options.size()); ++option_id) {
    auto &option = config.options[option_id];

    auto input_layout = option.input_layout;
    HCTR_CHECK_HINT(input_layout == embedding::EmbeddingLayout::FeatureMajor,
                    "only support feature major input layout");
    auto output_layout = option.output_layout;
    auto keys_preprocess_strategy = option.keys_preprocess_strategy;
    auto sort_strategy = option.sort_strategy;
    auto allreduce_strategy = option.allreduce_strategy;
    auto comm_strategy = option.comm_strategy;

    ebc_params.push_back(EmbeddingCollectionParam(
        num_table, num_lookup, lookup_params, shard_matrix, grouped_emb_params, batch_size,
        key_type, index_type, offset_type, emb_type, wgrad_type, input_layout, output_layout,
        sort_strategy, keys_preprocess_strategy, allreduce_strategy, comm_strategy,
        config.shard_configuration.compression_param));
  }

  HCTR_LOG(INFO, ROOT, "start preparing host data\n");
  std::vector<std::vector<HostEmbeddingCollectionInput<KeyType, OffsetType, EmbType>>>
      host_embedding_collection_input;  // niters x num_local_gpus
  {
    auto &ebc_param = ebc_params[0];
    for (int iter_id = 0; iter_id < niters; ++iter_id) {
      std::vector<HostEmbeddingCollectionInput<KeyType, OffsetType, EmbType>>
          host_ebc_input_for_all_gpus(num_local_gpus);
#pragma omp parallel for num_threads(num_local_gpus)
      for (int gpu_id = 0; gpu_id < num_local_gpus; ++gpu_id) {
        HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());

        generate_host_embedding_collection_input<KeyType, OffsetType, EmbType>(
            core_resource_manager_list[gpu_id]->get_global_gpu_id(), config, ebc_param,
            table_param_list, &host_ebc_input_for_all_gpus[gpu_id]);
      }
      host_embedding_collection_input.push_back(std::move(host_ebc_input_for_all_gpus));
    }
  }
  HCTR_LOG(INFO, ROOT, "finish preparing host data\n");

  HCTR_LOG(INFO, ROOT, "start preparing device data\n");
  std::vector<std::vector<core23::Tensor>> sparse_dp_tensors;
  std::vector<std::vector<core23::Tensor>> sparse_dp_bucket_range;
  std::vector<core23::Tensor> ebc_top_grads;
  std::vector<core23::Tensor> ebc_outptut;
  {
    auto &ebc_param = ebc_params[0];

    allocate_data_distributor_input(core_resource_manager_list, ebc_param, num_global_gpus,
                                    &sparse_dp_tensors, &sparse_dp_bucket_range, &ebc_top_grads,
                                    &ebc_outptut);

#pragma omp parallel for num_threads(num_local_gpus)
    for (int gpu_id = 0; gpu_id < num_local_gpus; ++gpu_id) {
      HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());

      generate_device_top_grads<EmbType>(ebc_top_grads[gpu_id]);
    }
  }
  sync_gpus();
  HCTR_LOG(INFO, ROOT, "finish preparing device data\n");

  for (int option_id = 0; option_id < static_cast<int>(config.options.size()); ++option_id) {
    auto &option = config.options[option_id];

    std::cout << "start testing embedding\n";
    std::cout << "shard_matrix:\n";
    for (auto &v : shard_matrix) {
      for (auto i : v) {
        std::cout << i << ",";
      }
      std::cout << std::endl;
    }
    std::cout << "grouped embedding params:\n";
    for (auto &v : grouped_emb_params) {
      if (v.table_placement_strategy == TablePlacementStrategy::ModelParallel) {
        std::cout << "model parallel:";
      } else if (v.table_placement_strategy == TablePlacementStrategy::DataParallel) {
        std::cout << "data parallel:";
      } else {
        HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError, "error");
      }
      for (auto i : v.table_ids) {
        std::cout << i << ",";
      }
      std::cout << std::endl;
    }

    std::cout << "compression_param:" << config.shard_configuration.compression_param << "\n";
    std::cout << "key_type:" << key_type << "\n";
    std::cout << "offset_type:" << offset_type << "\n";
    std::cout << "emb type:" << emb_type << "\n";
    std::cout << "option:" << option << std::endl;

    auto &ebc_param = ebc_params[option_id];

    std::vector<int> dr_lookup_ids(ebc_param.num_lookup);
    std::iota(dr_lookup_ids.begin(), dr_lookup_ids.end(), 0);
    std::shared_ptr<HugeCTR::DataDistributor> data_distributor =
        std::make_shared<HugeCTR::DataDistributor>(core_resource_manager_list, ebc_param,
                                                   table_param_list, dr_lookup_ids);

    std::vector<HugeCTR::DataDistributor::Result> data_distributor_outputs;
    for (int gpu_id = 0; gpu_id < num_local_gpus; ++gpu_id) {
      HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());

      data_distributor_outputs.push_back(HugeCTR::allocate_output_for_data_distributor(
          core_resource_manager_list[gpu_id], ebc_param));
    }

    std::unique_ptr<embedding::EmbeddingCollection> ebc =
        std::make_unique<embedding::EmbeddingCollection>(
            resource_manager, core_resource_manager_list, ebc_param, ebc_param, table_param_list);

    std::vector<std::vector<IGroupedEmbeddingTable *>> grouped_emb_table_ptr_list =
        ebc->get_grouped_embedding_tables();

    std::vector<EmbeddingReferenceGPU<KeyType, OffsetType, EmbType>> emb_ref;
    if (config.reference_check) {
      for (int gpu_id = 0; gpu_id < num_local_gpus; ++gpu_id) {
        HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());

        emb_ref.emplace_back(core_resource_manager_list[gpu_id], ebc_param, table_param_list,
                             grouped_emb_table_ptr_list[gpu_id]);
      }
    }

    // sync for emb table init
    // TODO: this is a WAR; need to find a way to remove the preallocation
    for (int local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
      auto device_id = resource_manager->get_local_gpu(local_gpu_id)->get_device_id();
      core23::Device device(core23::DeviceType::GPU, device_id);
      bool success = core23::AllocateBuffers(device);
      if (!success) {
        HCTR_LOG_S(DEBUG, ROOT) << "Nothing to preallocate" << std::endl;
      }
    }
    core23::Device device_h(core23::DeviceType::CPU);
    bool success = core23::AllocateBuffers(device_h);
    if (!success) {
      HCTR_LOG_S(DEBUG, ROOT) << "Nothing to preallocate" << std::endl;
    }
    sync_gpus();
    for (int iter = 0; iter < config.niters; ++iter) {
      std::cout << "iter:" << iter << "\n";
#pragma omp parallel for num_threads(num_local_gpus)
      for (int gpu_id = 0; gpu_id < num_local_gpus; ++gpu_id) {
        HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());

        for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
          core23::copy_sync(sparse_dp_tensors[gpu_id][lookup_id],
                            host_embedding_collection_input[iter][gpu_id].h_dp_keys[lookup_id]);
          core23::copy_sync(
              sparse_dp_bucket_range[gpu_id][lookup_id],
              host_embedding_collection_input[iter][gpu_id].h_dp_bucket_range[lookup_id]);
        }
      }
      sync_gpus();

      // gpu forward + backward
#pragma omp parallel for num_threads(num_local_gpus)
      for (int gpu_id = 0; gpu_id < num_local_gpus; ++gpu_id) {
        HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());

        data_distributor->distribute(gpu_id, sparse_dp_tensors[gpu_id],
                                     sparse_dp_bucket_range[gpu_id],
                                     data_distributor_outputs[gpu_id], batch_size);
        ebc->forward_per_gpu(true, gpu_id, data_distributor_outputs[gpu_id], ebc_outptut[gpu_id],
                             batch_size);
        ebc->backward_per_gpu(gpu_id, data_distributor_outputs[gpu_id], ebc_top_grads[gpu_id],
                              batch_size);
      }
      sync_gpus();

#ifdef ENABLE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      // cpu reference
#pragma omp parallel for num_threads(num_local_gpus)
      for (int gpu_id = 0; gpu_id < num_local_gpus; ++gpu_id) {
        HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());

        if (config.reference_check) {
          emb_ref[gpu_id].embedding_forward_cpu(sparse_dp_bucket_range[gpu_id],
                                                sparse_dp_tensors[gpu_id]);
          emb_ref[gpu_id].embedding_backward_cpu(ebc_top_grads[gpu_id]);
          emb_ref[gpu_id].compare_forward_result(ebc_outptut[gpu_id]);
          emb_ref[gpu_id].compare_backward_result(ebc->get_wgrad(gpu_id));
        }
      }
      sync_gpus();

      // gpu update
#pragma omp parallel for num_threads(num_local_gpus)
      for (int gpu_id = 0; gpu_id < num_local_gpus; ++gpu_id) {
        HugeCTR::CudaDeviceContext context(core_resource_manager_list[gpu_id]->get_device_id());

        ebc->update_per_gpu(gpu_id);
      }
      sync_gpus();
    }
  }
}

TEST(test_embedding_collection, utest_1node) {
  for (auto &config : get_ebc_single_node_utest_configuration()) {
    embedding_collection_e2e<uint32_t, uint32_t, uint32_t, float>(config);
    embedding_collection_e2e<uint64_t, uint64_t, uint32_t, float>(config);
  }
}

TEST(test_embedding_collection, utest_2node) {
  for (auto &config : get_ebc_two_node_utest_configuration()) {
    embedding_collection_e2e<uint32_t, uint32_t, uint32_t, float>(config);
  }
}