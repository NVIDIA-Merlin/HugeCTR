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

#include <omp.h>
#include <time.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>

using namespace std;

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cout << "Please input num_of_floats , file_path! " << std::endl;
    exit(-1);
  }
  long num_of_keys = std::atoi(argv[1]);
  long embedding_vector_size = std::atoi(argv[2]);
  long num_of_floats = num_of_keys * embedding_vector_size;
  std::string filepath = argv[3];
  std::cout << "Generate num_of_floats is " << num_of_floats << std::endl;
  std::cout << "filepath is " << filepath << std::endl;
  uniform_real_distribution<float> u(-1, 1);
  default_random_engine e(time(NULL));
  auto start = chrono::system_clock::now();
  long bulk_size = 256000000;
  for (int i = 0; i <= num_of_floats / bulk_size; i++) {
    long batch_size = std::min(bulk_size, num_of_floats - (i * bulk_size));
    if (batch_size == 0) break;
    std::vector<float> emb_data(batch_size);
    size_t const num_thread(48);
#pragma omp parallel num_threads(num_thread)
    {
      size_t const tid(omp_get_thread_num());
      auto chunk_size{batch_size / num_thread};
      auto const index{chunk_size * tid};
      if (tid == num_thread - 1) chunk_size += (batch_size % num_thread);
      default_random_engine e(time(NULL));
      for (size_t i{index}; i < index + chunk_size; i++) {
        emb_data[i] = u(e);
      }
    }
    std::cout << "writing " << batch_size << " floats" << std::endl;
    //============ WRITING A VECTOR INTO A FILE ================
    if (i == 0) {
      std::ofstream emb_stream(filepath + "/emb_vector",
                               std::ofstream::binary | std::ofstream::trunc);
      emb_stream.write(reinterpret_cast<const char *>(&emb_data[0]), sizeof(float) * batch_size);
      emb_stream.close();
    } else {
      std::ofstream emb_stream(filepath + "/emb_vector",
                               std::ofstream::binary | std::ofstream::app);
      emb_stream.write(reinterpret_cast<const char *>(&emb_data[0]), sizeof(float) * batch_size);
      emb_stream.close();
    }
    //===========================================================
  }

  std::cout << "Generating keys is " << num_of_keys << std::endl;
  std::vector<long long> key_data;
  for (long long i = 0; i < num_of_keys; ++i) {
    key_data.emplace_back(i);
  }
  //============ WRITING A VECTOR INTO A FILE ================
  std::ofstream key_stream(filepath + "/key", std::ofstream::binary | std::ofstream::trunc);
  key_stream.write(reinterpret_cast<const char *>(&key_data[0]), sizeof(long long) * num_of_keys);
  key_stream.close();
  //===========================================================

  auto finish = chrono::system_clock::now();
  auto duration = chrono::duration_cast<chrono::microseconds>(finish - start);
  std::cout << "elapsed time:" << duration.count() / CLOCKS_PER_SEC << 's' << std::endl;
  return 0;
}

template <class RealType = double>
class uniform_real_distribution;
