/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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


#pragma once

#include <exception>
#include <string>
#include "cuda_runtime_api.h"

#define CUDA_CHECK(val) { FasterGossipCommUtil::cuda_check_((val), __FILE__, __LINE__); }

namespace FasterGossipCommUtil {

class CudaException: public std::runtime_error {
public:
    CudaException(const std::string& what): runtime_error(what) {}
};

inline void cuda_check_(cudaError_t val, const char *file, int line) {
    if (val != cudaSuccess) {
        throw CudaException(std::string(file)
                            + ":"
                            + std::to_string(line)
                            + ": CUDA error "
                            + std::to_string(val)
                            + ": "
                            + cudaGetErrorString(val)
            );
    }
}


class CudaDeviceRestorer {
public:
    CudaDeviceRestorer() {
        CUDA_CHECK(cudaGetDevice(&dev_));
    }
    ~CudaDeviceRestorer() {
        CUDA_CHECK(cudaSetDevice(dev_));
    }
private:
    int dev_;
};

}
