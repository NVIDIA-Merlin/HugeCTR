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

#ifndef EMBEDDING_UTIL_HPP
#define EMBEDDING_UTIL_HPP

#include <string>
#include <regex>
#include <vector>
#include <sstream>
#include "tensorflow/core/framework/tensor.h"

namespace Utils {

inline std::vector<std::string> split(const std::string& input_s, const std::string& pattern) {
    std::regex re(pattern);
    std::sregex_token_iterator p(input_s.begin(), input_s.end(), re, -1);
    std::sregex_token_iterator end;
    std::vector<std::string> result;
    while (p != end) {
        result.emplace_back(*p++);
    }
    return result;
}

inline std::string strs_concat(const std::vector<std::string>& str_v, const std::string& connect_symbol) {
    std::string result = "";
    for (size_t i = 0; i < str_v.size() - 1; ++i) {
        result += (str_v[i] + connect_symbol);
    }
    result += *(str_v.rbegin());
    return result;
}


inline int string2num(const std::string& input_s) {
    int result = -1; // this string cannot convert to number.

    std::stringstream ss(input_s);
    if (!(ss >> result)){
        return -1;
    } else {
        return result;
    }
}

template <typename T>
inline bool check_in_set(const std::set<T>& set, const T& item) {
    auto it = set.find(item);
    return it != set.end();
}

template <typename tensor_type>
void print_tensor(const tensorflow::Tensor* const tensor) {
    auto tensor_flat = tensor->flat<tensor_type>();
    for (long int i = 0; i < tensor_flat.size(); ++i) {
        std::cout << tensor_flat(i) << ", ";
    }
    std::cout << std::endl;
}

} // namespace Utils
#endif 