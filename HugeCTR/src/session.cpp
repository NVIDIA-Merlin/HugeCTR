/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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


#include "HugeCTR/include/session.hpp"
#include <nvToolsExt.h>
#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

/**
 * check if device is avaliable.
 * lowest avaliable CC is min_major.min_minor
 * @param device_id gpu id
 * @param min_major minimum compute compatibility required
 * @param min_minor minimum compute compatibility required
 */
static void check_device(int device_id, int min_major, int min_minor) {
  int device_count = 0;
  CK_CUDA_THROW_(cudaGetDeviceCount(&device_count));
  if (device_id >= device_count) {
    CK_THROW_(Error_t::WrongInput, "device is not avaliable");
  }
  int o_device = -1;
  CK_CUDA_THROW_(get_set_device(device_id, &o_device));
  cudaDeviceProp deviceProp;
  if (cudaGetDeviceProperties(&deviceProp, device_id) != cudaSuccess) {
    CK_THROW_(Error_t::InvalidEnv, "Invalid device:" + std::to_string(device_id));
    return;
  }
  std::cout << "Device " << device_id << ": " << deviceProp.name << std::endl;
  int major = deviceProp.major;
  int minor = deviceProp.minor;
  if (major < min_major) {
    CK_THROW_(Error_t::InvalidEnv, "Device Compute Compacity is low");
  } else if (major == min_major && minor < min_minor) {
    CK_THROW_(Error_t::InvalidEnv, "Device Compute Compacity is low");
  }
  CK_CUDA_THROW_(get_set_device(o_device));
  return;
}

Session::Session(int batch_size, const std::string& json_name, const DeviceMap& device_map)
    : gpu_resource_group_(device_map) {
  try {
    for (auto dev : gpu_resource_group_.get_device_list()) {
      check_device(dev, 6, 0);  // lowest supported device is CC=60
    }
    parser_ = new Parser(json_name, batch_size);
    DataReader<TypeKey>* data_reader_array[2];
    parser_->create_pipeline(data_reader_array, &embedding_, &networks_, gpu_resource_group_);
    data_reader_ = data_reader_array[0];
    data_reader_eval_ = data_reader_array[1];
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
  }
}

/**
 * load the model (binary) from model_file.
 * In model file, model should be saved as
 * the sequence as discribed in configure file.
 **/
Error_t Session::load_params(const std::string& model_file, const std::string& embedding_file) {
  try {
    float* weight = new float[networks_[0]->get_params_num()]();
    std::ifstream model_stream(model_file, std::ifstream::binary);
    if (!embedding_file.empty()) {
      std::ifstream embedding_stream(embedding_file, std::ifstream::binary);
      if (!embedding_stream.is_open()) {
        CK_THROW_(Error_t::WrongInput, "Cannot open model file");
      }
      embedding_->upload_params_to_device(embedding_stream);
      embedding_stream.close();
    }
    model_stream.read(reinterpret_cast<char*>(weight),
                      networks_[0]->get_params_num() * sizeof(float));
    for (auto network : networks_) {
      network->upload_params_to_device(weight);
    }
    delete[] weight;
    model_stream.close();
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    return rt_err.get_error();
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

Error_t Session::init_params(std::string model_file) {
  try {
    // model_file generation;
    std::ofstream out_stream(model_file, std::ofstream::binary);
    if (!out_stream.is_open()) {
      CK_THROW_(Error_t::WrongInput, "Cannot open model file");
    }
    // network init
    for (auto network : networks_) {
      network->init_params(out_stream);
    }
    out_stream.close();
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    return rt_err.get_error();
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

void network_train_helper(int id, std::vector<Network*> networks_) {
  try {
    int local_gpu_count = networks_.size();
    networks_[id]->train(local_gpu_count);
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
  }
  return;
}

Error_t Session::train() {
  try {
    data_reader_->read_a_batch_to_device();
    embedding_->forward();

    if (networks_.size() > 1) {
      // execute dense forward and backward with multi-cpu threads
      for (unsigned int i = 0; i < networks_.size(); i++) {
        gpu_resource_group_.results[i] = gpu_resource_group_.train_thread_pool.push(
            std::ref(network_train_helper), networks_);
      }
      for (unsigned int i = 0; i < networks_.size(); i++) {
        gpu_resource_group_.results[i].get();
      }
    } else if (networks_.size() == 1) {
      networks_[0]->train(networks_.size());
    } else {
      assert(!"networks_.size() should not less than 1.");
    }
    // wgrad exchange
    if (networks_.size() > 1) {
      CK_NCCL_THROW_(ncclGroupStart());
      for (auto network : networks_) {
        network->exchange_wgrad();
      }
      CK_NCCL_THROW_(ncclGroupEnd());
    }
    for (auto network : networks_) {
      network->update_params();
    }

    embedding_->backward();
    embedding_->update_params();
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    return rt_err.get_error();
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

void network_eval_helper(int id, Network* n) {
  try {
    n->eval();
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
  }
}

Error_t Session::eval() {
  try {
    if (data_reader_eval_ == nullptr) return Error_t::NotInitialized;
    data_reader_eval_->read_a_batch_to_device();
    embedding_->forward();

    if (networks_.size() > 1) {
      // execute dense forward and backward with multi-cpu threads
      for (unsigned int i = 0; i < networks_.size(); i++) {
        gpu_resource_group_.results[i] = gpu_resource_group_.train_thread_pool.push(
            std::ref(network_train_helper), networks_);
      }
      for (unsigned int i = 0; i < networks_.size(); i++) {
        gpu_resource_group_.results[i].get();
      }
    } else if (networks_.size() == 1) {
      networks_[0]->eval();
    } else {
      assert(!"networks_.size() should not less than 1.");
    }
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    return rt_err.get_error();
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

Error_t Session::download_params_to_file(std::string weights_file, std::string embedding_file) {
  try {
    std::ofstream out_stream_embedding(embedding_file, std::ofstream::binary);
    embedding_->download_params_to_host(out_stream_embedding);
    int numprocs = 1, pid = 0;
#ifdef ENABLE_MPI
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
#endif
    if (pid == 0) {
      std::ofstream out_stream_weight(weights_file, std::ofstream::binary);
      networks_[0]->download_params_to_host(out_stream_weight);
      std::string no_trained_params = networks_[0]->get_no_trained_params_in_string();
      if (no_trained_params.length() != 0) {
        std::string ntp_file = weights_file + ".ntp.json";
        std::ofstream out_stream_ntp(ntp_file, std::ofstream::out);
        out_stream_ntp.write(no_trained_params.c_str(), no_trained_params.length());
        out_stream_ntp.close();
      }
      out_stream_weight.close();
    }
    out_stream_embedding.close();
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    return rt_err.get_error();
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

Error_t Session::get_current_loss(float* loss) {
  try {
    float loss_sum = 0.f;
    float loss_reduced = 0.f;
    int numprocs = 1, pid = 0;
#ifdef ENABLE_MPI
    CK_MPI_THROW_(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
    CK_MPI_THROW_(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
#endif
    // Collect all the loss from every network and average
    for (auto network : networks_) {
      loss_sum += network->get_loss();
    }
    if (numprocs > 1) {
#ifdef ENABLE_MPI
      CK_MPI_THROW_(MPI_Reduce(&loss_sum, &loss_reduced, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));
#endif
    } else {
      loss_reduced = loss_sum;
    }
    *loss = loss_reduced / networks_.size() / numprocs;
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    return rt_err.get_error();
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    return Error_t::UnspecificError;
  }
  return Error_t::Success;
}

Session::~Session() {
  try {
    for (auto device : gpu_resource_group_.get_device_list()) {
      int o_device = -1;
      CK_CUDA_THROW_(get_set_device(device, &o_device));
      CK_CUDA_THROW_(cudaDeviceSynchronize());
      CK_CUDA_THROW_(get_set_device(o_device));
    }

    for (auto network : networks_) {
      assert(network != nullptr);
      delete network;
    }

    delete embedding_;
    delete data_reader_;
    delete parser_;
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
  }
}

}  // namespace HugeCTR
