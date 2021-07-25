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

#include <model_oversubscriber/parameter_server.hpp>

#include <fstream>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

namespace HugeCTR {

namespace {

void open_and_get_size(const std::string& file_name, std::ifstream& stream,
                       size_t& file_size_in_byte) {
  stream.open(file_name, std::ifstream::binary);
  if (!stream.is_open()) {
    CK_THROW_(Error_t::WrongInput, "Cannot open the file: " + file_name);
  }
  file_size_in_byte = fs::file_size(file_name);
}

}  // namespace

template <typename TypeKey>
ParameterServer<TypeKey>::ParameterServer(bool use_host_ps,
    const std::string &sparse_model_file, Embedding_t embedding_type,
    size_t emb_vec_size, std::shared_ptr<ResourceManager> resource_manager)
    : use_host_ps_(use_host_ps),
      sparse_model_entity_(SparseModelEntity<TypeKey>(use_host_ps,
      sparse_model_file, embedding_type, emb_vec_size, resource_manager)) {}

template <typename TypeKey>
void ParameterServer<TypeKey>::load_keyset_from_file(
	std::string keyset_file) {
  try {
    std::ifstream keyset_stream;
    size_t file_size_in_byte = 0;
    open_and_get_size(keyset_file, keyset_stream, file_size_in_byte);

    if (file_size_in_byte == 0) {
      CK_THROW_(Error_t::WrongInput, std::string(keyset_file) + " is empty");
    }

    size_t num_keys_in_file = file_size_in_byte / sizeof(TypeKey);
    keyset_.resize(num_keys_in_file);
    keyset_stream.read((char*)keyset_.data(), file_size_in_byte);
#ifdef ENABLE_MPI
    CK_MPI_THROW_(MPI_Barrier(MPI_COMM_WORLD));
#endif
  } catch (const internal_runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    throw;
  }
}

template <typename TypeKey>
void ParameterServer<TypeKey>::pull(BufferBag& buf_bag, size_t& hit_size) {
  if (keyset_.empty()) {
    CK_THROW_(Error_t::WrongInput, "keyset is empty");
  }
  sparse_model_entity_.load_vec_by_key(keyset_, buf_bag, hit_size);
}

template <typename TypeKey>
std::pair<std::vector<long long>, std::vector<float>>
ParameterServer<TypeKey>::pull(const std::vector<long long> &keys_to_load) {
  if (keys_to_load.empty()) {
    CK_THROW_(Error_t::WrongInput, "keyset is empty");
  }
  return sparse_model_entity_.load_vec_by_key(keys_to_load);
}

template <typename TypeKey>
void ParameterServer<TypeKey>::push(BufferBag &buf_bag, size_t dump_size) {
  if (dump_size == 0) return;
  sparse_model_entity_.dump_vec_by_key(buf_bag, dump_size);
}

template class ParameterServer<long long>;
template class ParameterServer<unsigned>;

}  // namespace HugeCTR
