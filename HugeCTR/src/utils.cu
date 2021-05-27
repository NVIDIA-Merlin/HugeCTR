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

 #include <utils.cuh>
 #include <utils.hpp>
 
 namespace HugeCTR {
 
 template <typename TIN, typename TOUT>
 void convert_array_on_device(TOUT* out, const TIN* in, size_t num_elements, const cudaStream_t& stream) {
   convert_array<<<(num_elements - 1) / 1024 + 1, 1024, 0, stream>>>(out, in, num_elements);
 }
 
 template void convert_array_on_device<long long, int>(int*, const long long*, size_t, const cudaStream_t&);
 template void convert_array_on_device<unsigned int, int>(int*, const unsigned int*, size_t, const cudaStream_t&);
 template void convert_array_on_device<float, float>(float*, const float*, size_t, const cudaStream_t&);
 template void convert_array_on_device<float, __half>(__half*, const float*, size_t, const cudaStream_t&);
 template void convert_array_on_device<__half, float>(float*, const __half*, size_t, const cudaStream_t&);
 
 } // namespace HugeCTR