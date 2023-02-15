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

#include <embedding_storage/weight_io/parameter_IO.hpp>

using namespace HugeCTR;
namespace embedding {

EmbeddingParameterIO::EmbeddingParameterIO(
    std::shared_ptr<HugeCTR::ResourceManager> resource_manager) {
  resource_manager_ = resource_manager.get();
  int num_local_gpus = resource_manager->get_local_gpu_count();
  for (int local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
    auto core_resource_manager =
        std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager, local_gpu_id);
    core_list_.push_back(core_resource_manager);
  }
}

void EmbeddingParameterIO::add_embedding_collection(EmbeddingCollection* embedding_collection) {
  embedding_collections_.push_back(embedding_collection);
}

void EmbeddingParameterIO::load_metadata(const std::string& parameters_folder_path, int ebc_id,
                                         struct EmbeddingParameterInfo& epi) {
  // for now we can write binary first , we can change it to HDF5 in future;
  auto file_system = get_fs_object(parameters_folder_path, SparseFSType::FS);
  std::string ebc_path = parameters_folder_path + "/embedding_collection_" + std::to_string(ebc_id);
  std::string ebc_meta_path = ebc_path + "/meta_data";

  epi.parameter_folder_path = ebc_path;
  size_t meta_file_length = file_system->get_file_size(ebc_meta_path);
  if (meta_file_length < MetaDataValidLength) {
    HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError, "metadata is too small , could not be valid");
  }
  char* buffer = (char*)malloc(meta_file_length);
  file_system->read_from(ebc_meta_path, buffer, meta_file_length, 0);
  // FIX:use table nums check metadata is valid
  int start_offset = 5;
  int* buffer_head = (int*)buffer;
  epi.table_nums = buffer_head[0];
  size_t* buffer_key_num = (size_t*)(buffer + (start_offset + epi.table_nums) * sizeof(int));
  int* buffer_ev_length = (int*)(buffer + (start_offset + epi.table_nums) * sizeof(int) +
                                 epi.table_nums * sizeof(size_t));
  if (buffer_head[1] == 0) {
    epi.key_type = core::DataType(core::TensorScalarType::UInt32);
  } else if (buffer_head[1] == 1) {
    epi.key_type = core::DataType(core::TensorScalarType::Int64);
  }

  if (buffer_head[2] == 0) {
    epi.embedding_value_type = core::DataType(core::TensorScalarType::Float32);
  } else if (buffer_head[2] == 1) {
    epi.embedding_value_type = core::DataType(core::TensorScalarType::Float16);
  }
  epi.max_embedding_vector_length = buffer_head[4];

  for (int i = 0; i < epi.table_nums; ++i) {
    int table_id = buffer_head[start_offset + i];
    size_t table_key_num = buffer_key_num[i];
    int ev_length = buffer_ev_length[i];
    epi.table_ids.push_back(table_id);
    epi.table_key_nums[table_id] = table_key_num;
    epi.table_embedding_vector_lengths[table_id] = ev_length;
  }
  free(buffer);

  return;
}

void EmbeddingParameterIO::get_parameter_info_from_model(
    const std::string& path, std::vector<struct EmbeddingParameterInfo>& epis) {
  int collections_num = embedding_collections_.size();
  if (collections_num == 0) {
    HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError,
                   "don't have any embedding collection , please check your model");
  }
  int num_local_gpus = resource_manager_->get_local_gpu_count();
  int num_global_gpus = core_list_[0]->get_global_gpu_count();

  // get embedding distribute collection from EmbeddingCollection
  for (int i = 0; i < collections_num; ++i) {
    EmbeddingCollectionParam tmp_ebc_param = embedding_collections_[i]->ebc_param_;
    std::vector<std::vector<int>> tmp_shard_matrix = tmp_ebc_param.shard_matrix;
    struct EmbeddingParameterInfo tmp_epi;
    tmp_epi.embedding_collection_id = i;
    tmp_epi.table_nums = tmp_ebc_param.num_table;
    tmp_epi.key_type = tmp_ebc_param.key_type;
    tmp_epi.embedding_value_type = tmp_ebc_param.emb_type;

    if (embedding_collections_[i]->embedding_optimizers_.size() > 0)
      tmp_epi.optimizer_type = embedding_collections_[i]->embedding_optimizers_[0];
    else
      HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError,
                     "embedding_collection do not have optimizer ,please check "
                     "embedding_collection is inited correctly");

    // get gpu infomation in local
    tmp_epi.gemb_distribution =
        std::make_shared<struct GlobalEmbeddingDistribution>(num_global_gpus, tmp_epi.table_nums);
    for (int gpu_id = 0; gpu_id < num_local_gpus; ++gpu_id) {
      HugeCTR::CudaDeviceContext context(core_list_[gpu_id]->get_device_id());
      int global_gpu_id = core_list_[gpu_id]->get_global_gpu_id();
      int embedding_group_num = tmp_ebc_param.grouped_emb_params.size();
      if (embedding_group_num == 0) {
        HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError,
                       "don't have any grouped embedding table  , please check your model");
      }
      auto& tmp_embedding_tables_per_gpu = embedding_collections_[i]->embedding_tables_[gpu_id];

      for (int grouped_id = 0; grouped_id < embedding_group_num; ++grouped_id) {
        auto group_table_ids = tmp_embedding_tables_per_gpu[grouped_id]->table_ids();
        auto group_table_kns = tmp_embedding_tables_per_gpu[grouped_id]->key_num_per_table();

        for (int tmp_table_index = 0; tmp_table_index < group_table_ids.size(); ++tmp_table_index) {
          int tmp_table_id = group_table_ids[tmp_table_index];
          size_t tmp_key_num = group_table_kns[tmp_table_index];
          tmp_epi.gemb_distribution->set(tmp_key_num, global_gpu_id, tmp_table_id);

          if (tmp_ebc_param.grouped_emb_params[grouped_id].table_placement_strategy ==
              TablePlacementStrategy::DataParallel) {
            tmp_epi.gemb_distribution->set_parallel(tmp_table_id, 1);
          } else if (tmp_ebc_param.grouped_emb_params[grouped_id].table_placement_strategy ==
                     TablePlacementStrategy::ModelParallel) {
            tmp_epi.gemb_distribution->set_parallel(tmp_table_id, 2);
          } else {
            HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError,
                           "For now , 3G embedding don't support this parallel model");
          }
        }
      }
    }

    for (auto tmp_lookup_param : tmp_ebc_param.lookup_params) {
      if (tmp_epi.table_embedding_vector_lengths.find(tmp_lookup_param.table_id) ==
          tmp_epi.table_embedding_vector_lengths.end()) {
        tmp_epi.table_embedding_vector_lengths.insert(
            {tmp_lookup_param.table_id, tmp_lookup_param.ev_size});
        tmp_epi.table_ids.push_back(tmp_lookup_param.table_id);
      }
    }

    std::sort(tmp_epi.table_ids.begin(), tmp_epi.table_ids.end());

    if (resource_manager_->get_num_process() > 1) {
#ifdef ENABLE_MPI
      size_t* distribute_ptr = tmp_epi.gemb_distribution->get_buffer();
      HCTR_MPI_THROW(MPI_Allreduce(distribute_ptr, distribute_ptr,
                                   num_global_gpus * tmp_epi.table_nums, MPI_SIZE_T, MPI_SUM,
                                   MPI_COMM_WORLD));
#else
      HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError,
                     "when you have multi node ,you must define macro ENABLE_MPI");
#endif
    }

    for (size_t tmp_table_id = 0; tmp_table_id < tmp_epi.table_ids.size(); ++tmp_table_id) {
      size_t key_num_sum = tmp_epi.gemb_distribution->get_table_keynum(tmp_table_id);
      tmp_epi.table_key_nums.insert({tmp_table_id, key_num_sum});
    }

    epis.push_back(tmp_epi);
  }
}

void EmbeddingParameterIO::dump_metadata(const std::string& parameters_folder_path,
                                         const struct EmbeddingParameterInfo& epi,
                                         const std::vector<int>& table_ids) {
  int myrank = resource_manager_->get_process_id();
  // for now we can write binary first , we can change it to HDF5 in future;
  auto file_system = get_fs_object(parameters_folder_path, SparseFSType::FS);
  file_system->delete_dir(parameters_folder_path);
  file_system->make_dir(parameters_folder_path);
  std::string ebc_path = parameters_folder_path + "/embedding_collection_" +
                         std::to_string(epi.embedding_collection_id);
  file_system->make_dir(ebc_path);

  std::string ebc_meta_path = ebc_path + "/meta_data";
  // first calculate meta data lenth;
  int buffer_length = MetaDataHeadLength;  // nbytes
  std::vector<int> table_ids_update;
  if (table_ids.empty()) {
    for (int table_id = 0; table_id < epi.table_nums; ++table_id) {
      table_ids_update.push_back(table_id);
    }
  } else {
    for (int i = 0; i < table_ids.size(); ++i) {
      if (table_ids[i] < 0 || table_ids[i] >= epi.table_nums) {
        HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError, "Input table id is out of range");
      }
      table_ids_update.push_back(table_ids[i]);
    }
  }

  std::sort(table_ids_update.begin(), table_ids_update.end());
  // add length to store table_ids
  buffer_length += table_ids_update.size() * sizeof(int);
  // add length to store key_num per table
  buffer_length += table_ids_update.size() * sizeof(size_t);
  // add length to store embedding vector length per table
  buffer_length += table_ids_update.size() * sizeof(int);

  char* buffer = (char*)malloc(buffer_length);
  ;
  memset(buffer, 0, buffer_length);

  int* buffer_head = (int*)buffer;
  buffer_head[0] = (int)table_ids_update.size();
  if (epi.key_type.type() == core::TensorScalarType::UInt32) {
    buffer_head[1] = 0;
  } else if (epi.key_type.type() == core::TensorScalarType::Int64) {
    buffer_head[1] = 1;
  }

  if (epi.embedding_value_type.type() == core::TensorScalarType::Float32) {
    buffer_head[2] = 0;
  } else if (epi.embedding_value_type.type() == core::TensorScalarType::Float16) {
    buffer_head[2] = 1;
  }

  int start_index = 5;
  for (auto table_id : table_ids_update) {
    buffer_head[start_index] = table_id;
    ++start_index;
  }

  size_t* buffer_key_num = (size_t*)(buffer_head + start_index);
  start_index = 0;

  for (auto table_id : table_ids_update) {
    buffer_key_num[start_index] = epi.table_key_nums.at(table_id);
    ++start_index;
  }

  int* buffer_ev_length = (int*)(buffer_key_num + start_index);
  start_index = 0;
  for (auto table_id : table_ids_update) {
    buffer_ev_length[start_index] = epi.table_embedding_vector_lengths.at(table_id);
    ++start_index;
  }

  if (myrank == 0) {
    file_system->write_to(ebc_meta_path, buffer, 0, buffer_length);
  }

  free(buffer);
}

void EmbeddingParameterIO::dump_embedding_weight(const std::string& parameters_folder_path,
                                                 struct EmbeddingParameterInfo& epi,
                                                 const std::vector<int>& table_ids) {
  int num_local_gpus = resource_manager_->get_local_gpu_count();
  int nrank = resource_manager_->get_num_process();
  int myrank = resource_manager_->get_process_id();

  auto file_system = get_fs_object(parameters_folder_path);
  file_system->make_dir(parameters_folder_path);
  std::string ebc_path = parameters_folder_path + "/embedding_collection_" +
                         std::to_string(epi.embedding_collection_id);
  file_system->make_dir(ebc_path);
  EmbeddingCollection* tmp_ebc = embedding_collections_[epi.embedding_collection_id];

  auto& group_embedding_tables = tmp_ebc->embedding_tables_;
  std::vector<int> table_ids_update;
  if (table_ids.empty()) {
    for (int table_id = 0; table_id < epi.table_nums; ++table_id) {
      table_ids_update.push_back(table_id);
    }
  } else {
    for (int i = 0; i < table_ids.size(); ++i) {
      if (table_ids[i] < 0 || table_ids[i] >= epi.table_nums) {
        HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError, "Input table id is out of range");
      }
      table_ids_update.push_back(table_ids[i]);
    }
  }

  DISPATCH_INTEGRAL_FUNCTION(epi.key_type.type(), key_t, [&] {
    for (int table_id = 0; table_id < table_ids_update.size(); ++table_id) {
      std::string ebc_key_path = ebc_path + "/key" + std::to_string(table_id);
      std::string ebc_weight_path = ebc_path + "/weight" + std::to_string(table_id);
      write_file_head(ebc_key_path, EmbeddingFileType::Key, table_id, file_system);
      write_file_head(ebc_weight_path, EmbeddingFileType::Weight, table_id, file_system);
      // FIX:to enum
      int parallel_mode = epi.gemb_distribution->get_parallel(table_id);
      // data parallel
      if (parallel_mode == 1) {
        HugeCTR::CudaDeviceContext context(core_list_[0]->get_device_id());
        int global_gpu_id = core_list_[0]->get_global_gpu_id();
        size_t table_ev_length = epi.table_embedding_vector_lengths.at(table_id);

        size_t table_key_num = epi.gemb_distribution->get(global_gpu_id, table_id);
        size_t weight_length = table_key_num * table_ev_length;

        auto buffer_ptr = core::GetBuffer(core_list_[0]);
        Tensor key_tensor_tmp = buffer_ptr->reserve(table_key_num, DeviceType::CPU, epi.key_type);
        Tensor weight_tensor_tmp =
            buffer_ptr->reserve(weight_length, DeviceType::CPU, epi.embedding_value_type);

        buffer_ptr->allocate();
        int group_nums = group_embedding_tables.size();
        int group_index = -1;
        for (int group_id = 0; group_id < group_nums; ++group_id) {
          std::vector<int>& group_table_ids =
              tmp_ebc->ebc_param_.grouped_emb_params[group_id].table_ids;
          auto find_iter = std::find(group_table_ids.begin(), group_table_ids.end(), table_id);
          if (find_iter != group_table_ids.end()) {
            group_index = group_id;
            break;
          }
        }
        if (group_index == -1) {
          HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError,
                         "can't find table id in any grouped tables");
        }
        group_embedding_tables[0][group_index]->dump_by_id(&key_tensor_tmp, &weight_tensor_tmp,
                                                           table_id);
        char* table_key_ptr = (char*)key_tensor_tmp.get();
        char* table_weight_ptr = (char*)weight_tensor_tmp.get();
#ifdef ENABLE_MPI
        if (resource_manager_->get_process_id() == 0) {
          file_system->write_to(ebc_key_path, table_key_ptr, 0, table_key_num * sizeof(key_t),
                                false);
          file_system->write_to(ebc_weight_path, table_weight_ptr, 0, weight_length * sizeof(float),
                                false);
        } else {
          file_system->write_to(ebc_key_path, table_key_ptr, FileHeadNbytes, 0, false);
          file_system->write_to(ebc_weight_path, table_weight_ptr, FileHeadNbytes, 0, false);
        }
#else
          file_system->write_to(ebc_key_path,table_key_ptr,0,table_key_num*sizeof(key_t),false);
          file_system->write_to(ebc_weight_path,table_weight_ptr,0,weight_length*sizeof(float),false);
#endif
      }
      // model parallel
      else if (parallel_mode == 2) {
        size_t table_ev_length = epi.table_embedding_vector_lengths.at(table_id);
        size_t table_key_num_local = 0;
        size_t weight_length_local = 0;

        int group_nums = group_embedding_tables.size();
        int group_index = -1;
        for (int group_id = 0; group_id < group_nums; ++group_id) {
          std::vector<int>& group_table_ids =
              tmp_ebc->ebc_param_.grouped_emb_params[group_id].table_ids;

          auto find_iter = std::find(group_table_ids.begin(), group_table_ids.end(), table_id);
          if (find_iter != group_table_ids.end()) {
            group_index = group_id;
            break;
          }
        }
        if (group_index == -1) {
          HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError,
                         "can't find table id in any grouped tables");
        }

        std::vector<int> local_gpu_id_hit;
        std::vector<int> table_key_num_hit;
        for (int local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
          HugeCTR::CudaDeviceContext context(core_list_[local_gpu_id]->get_device_id());
          int global_gpu_id = core_list_[local_gpu_id]->get_global_gpu_id();
          size_t tmp_key_num_gpu = epi.gemb_distribution->get(global_gpu_id, table_id);
          if (tmp_key_num_gpu > 0) {
            table_key_num_local += tmp_key_num_gpu;
            local_gpu_id_hit.push_back(local_gpu_id);
            table_key_num_hit.push_back(tmp_key_num_gpu);
          }
        }
        std::vector<size_t> offset_per_rank(nrank, 0);
        offset_per_rank[myrank] = table_key_num_local;
#ifdef ENABLE_MPI
        HCTR_MPI_THROW(MPI_Allgather(&table_key_num_local, 1, MPI_SIZE_T, offset_per_rank.data(), 1,
                                     MPI_SIZE_T, MPI_COMM_WORLD));
#endif

        std::exclusive_scan(offset_per_rank.begin(), offset_per_rank.end(), offset_per_rank.begin(),
                            0);
        weight_length_local = table_key_num_local * table_ev_length;
        size_t key_offset = offset_per_rank[myrank] * sizeof(key_t);
        size_t weight_offset = key_offset * table_ev_length * sizeof(float);

        key_t* table_key_ptr;
        float* table_weight_ptr;
        if (table_key_num_local > 0) {
          table_key_ptr = (key_t*)malloc(table_key_num_local * sizeof(key_t));
          table_weight_ptr = (float*)malloc(weight_length_local * sizeof(float));

          size_t tmp_offset = 0;
          for (int hit_index = 0; hit_index < local_gpu_id_hit.size(); ++hit_index) {
            int hit_gpu_id = local_gpu_id_hit[hit_index];
            size_t tmp_local_key_num = table_key_num_hit[hit_index];

            key_t* tmp_table_key_ptr = table_key_ptr + tmp_offset;
            float* tmp_table_weight_ptr = table_weight_ptr + tmp_offset * table_ev_length;
            auto buffer_ptr = core::GetBuffer(core_list_[hit_gpu_id]);
            Tensor key_tensor_tmp =
                buffer_ptr->reserve(tmp_local_key_num, DeviceType::CPU, epi.key_type);
            Tensor weight_tensor_tmp = buffer_ptr->reserve(
                tmp_local_key_num * table_ev_length, DeviceType::CPU, epi.embedding_value_type);

            buffer_ptr->allocate();

            HugeCTR::CudaDeviceContext context(core_list_[hit_gpu_id]->get_device_id());

            group_embedding_tables[hit_gpu_id][group_index]->dump_by_id(
                &key_tensor_tmp, &weight_tensor_tmp, table_id);
            key_t* tmp_table_key_ptr_part = key_tensor_tmp.get<key_t>();
            float* tmp_table_weight_ptr_part = weight_tensor_tmp.get<float>();

            memcpy(tmp_table_key_ptr, tmp_table_key_ptr_part, tmp_local_key_num * sizeof(key_t));
            memcpy(tmp_table_weight_ptr, tmp_table_weight_ptr_part,
                   tmp_local_key_num * table_ev_length * sizeof(float));
            tmp_offset += tmp_local_key_num;
          }
        }
        file_system->write_to(ebc_key_path, table_key_ptr, key_offset,
                              table_key_num_local * sizeof(key_t), false);
        file_system->write_to(ebc_weight_path, table_weight_ptr, weight_offset,
                              weight_length_local * sizeof(float), false);
        free(table_key_ptr);
        free(table_weight_ptr);
      } else {
        HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError,
                       "For now , 3G embedding don't support this parallel model");
      }
    }
  });
}

void EmbeddingParameterIO::dump_opt_state(const std::string& parameters_folder_path,
                                          struct EmbeddingParameterInfo& epi,
                                          const std::vector<int>& table_ids) {
  HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError, "wait 3G embedding optimizer complete");
}

std::shared_ptr<EmbeddingWeightIO> EmbeddingParameterIO::get_fs_object(const std::string& file_name,
                                                                       SparseFSType fs_type) {
  if (fs_type == SparseFSType::AUTO) {
#ifdef ENABLE_MPI
    return std::make_shared<EmbeddingWeightIOMpi>(file_name);
#endif
    return std::make_shared<EmbeddingWeightIOFS>(file_name);
  } else if (fs_type == SparseFSType::MPI) {
#ifdef ENABLE_MPI
    return std::make_shared<EmbeddingWeightIOMpi>(file_name);
#else
    HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError,
                   "when don't compile with MPI,can't specified MPI");
#endif
  }

  return std::make_shared<EmbeddingWeightIOFS>(file_name);
}

void EmbeddingParameterIO::load_embedding_weight(
    const struct EmbeddingParameterInfo& epi, int fs_table_id, Tensor& keys,
    Tensor& embedding_weights, embeddingFilter key_select,
    std::shared_ptr<core::CoreResourceManager> core_resource, const core::DataType& target_key_type,
    const core::DataType& target_value_type) {
  auto file_system = get_fs_object(epi.parameter_folder_path, SparseFSType::FS);
  std::string ebc_path = epi.parameter_folder_path + "/embedding_collection_" +
                         std::to_string(epi.embedding_collection_id);
  std::string ebc_key_path = epi.parameter_folder_path + "/key" + std::to_string(fs_table_id);
  std::string ebc_weight_path = epi.parameter_folder_path + "/weight" + std::to_string(fs_table_id);
  DISPATCH_INTEGRAL_FUNCTION(epi.key_type.type(), key_t, [&] {
    // TODO::need to check file head , safety check
    size_t ev_length = epi.table_embedding_vector_lengths.at(fs_table_id);
    size_t key_file_length = file_system->get_file_size(ebc_key_path);
    size_t weight_file_length = file_system->get_file_size(ebc_weight_path);
    size_t key_num = (key_file_length - FileHeadNbytes) / sizeof(key_t);
    size_t weight_num = (weight_file_length - FileHeadNbytes) / sizeof(float) / ev_length;
    if (key_num != weight_num)
      HCTR_OWN_THROW(HugeCTR::Error_t::WrongInput,
                     "Error: key num is not equal with embedding vector num");

    auto buffer_ptr = core::GetBuffer(core_resource);
    Tensor key_tensor_tmp = buffer_ptr->reserve(key_num, DeviceType::CPU, epi.key_type);
    buffer_ptr->allocate();
    key_t* key_tensor_ptr = key_tensor_tmp.get<key_t>();
    file_system->read_from(ebc_key_path, key_tensor_ptr, key_num * sizeof(key_t), FileHeadNbytes);
    size_t target_key_num = 0;
    for (int i = 0; i < key_num; ++i) {
      if (key_select((size_t)key_tensor_ptr[i])) {
        target_key_num++;
      }
    }

    Tensor weight_tensor_tmp =
        buffer_ptr->reserve(weight_num * ev_length, DeviceType::CPU, epi.embedding_value_type);
    Tensor target_key_tensor_tmp;
    Tensor target_weight_tensor_tmp;
    if (target_key_type.type() == core::TensorScalarType::None) {
      keys = buffer_ptr->reserve(target_key_num, DeviceType::CPU, epi.key_type);
    } else {
      keys = buffer_ptr->reserve(target_key_num, DeviceType::CPU, target_key_type);
    }

    if (target_value_type.type() == core::TensorScalarType::None) {
      embedding_weights = buffer_ptr->reserve(target_key_num * ev_length, DeviceType::CPU,
                                              epi.embedding_value_type);
    } else {
      embedding_weights =
          buffer_ptr->reserve(target_key_num * ev_length, DeviceType::CPU, target_value_type);
    }

    buffer_ptr->allocate();
    key_t* keys_ptr = keys.get<key_t>();
    float* weight_tensor_ptr = weight_tensor_tmp.get<float>();
    float* embedding_weights_ptr = embedding_weights.get<float>();

    file_system->read_from(ebc_weight_path, weight_tensor_ptr, key_num * ev_length * sizeof(key_t),
                           FileHeadNbytes);
    size_t tmp_target_key_offset = 0;
    // TODO::need use openmp optimize
    for (size_t i = 0; i < key_num; ++i) {
      if (key_select(key_tensor_ptr[i])) {
        float* tmp_embedding_weights_ptr =
            embedding_weights_ptr + tmp_target_key_offset * ev_length;
        float* tmp_weight_tensor_ptr = weight_tensor_ptr + i * ev_length;
        keys_ptr[tmp_target_key_offset] = key_tensor_ptr[i];
        for (size_t j = 0; j < ev_length; ++j) {
          tmp_embedding_weights_ptr[j] = tmp_weight_tensor_ptr[j];
        }
        tmp_target_key_offset++;
      }
    }
  });
}

void EmbeddingParameterIO::load_opt_state(const struct EmbeddingParameterInfo& epi, int fs_table_id,
                                          Tensor& keys, Tensor& optimizer_buffer,
                                          embeddingFilter key_select,
                                          std::shared_ptr<core::CoreResourceManager> core_resource,
                                          const core::DataType& target_key_type,
                                          const core::DataType& target_value_type) {
  HCTR_OWN_THROW(HugeCTR::Error_t::UnspecificError, "wait 3G embedding optimizer complete");
}
void EmbeddingParameterIO::write_file_head(const std::string& path, EmbeddingFileType file_type,
                                           int table_id, std::shared_ptr<EmbeddingWeightIO>& fs) {
  int* head_buffer = (int*)malloc(FileHeadLength * sizeof(int));
  memset(head_buffer, 0, FileHeadNbytes);
  switch (file_type) {
    case EmbeddingFileType::Key:
      head_buffer[0] = 1;
      break;
    case EmbeddingFileType::Weight:
      head_buffer[0] = 2;
      break;
    case EmbeddingFileType::Optimizer:
      head_buffer[0] = 3;
      break;
  }
  head_buffer[1] = table_id;
#ifdef ENABLE_MPI
  if (resource_manager_->get_process_id() == 0) {
    fs->write_to(path, head_buffer, 0, FileHeadNbytes);
  } else {
    fs->write_to(path, head_buffer, 0, 0);
  }
#else
  fs->write_to(path, head_buffer, 0, FileHeadNbytes);
#endif
  free(head_buffer);
  return;
}

}  // namespace embedding
