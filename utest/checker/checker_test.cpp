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

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/check_sum.hpp"
#include "HugeCTR/include/file_source.hpp"
#include "gtest/gtest.h"

using namespace HugeCTR;

TEST(checker, CheckSum) {
  auto func = [](std::string file, std::string str){
    int count = str.length();
    char sum = 0;
    std::ofstream out_stream(file, std::ofstream::binary);
    for(int i = 0; i < count; i++){
      sum += str[i];
    }
    out_stream.write(reinterpret_cast<char*>(&count), sizeof(int));
    out_stream.write(str.c_str(), count);
    out_stream.write(reinterpret_cast<char*>(&sum), sizeof(char));
    out_stream.write(reinterpret_cast<char*>(&count), sizeof(int));
    out_stream.write(str.c_str(), count);
    out_stream.write(reinterpret_cast<char*>(&sum), sizeof(char));

    out_stream.close();

    out_stream.open("file_list.txt", std::ofstream::out);
    out_stream << "1\n" << file;
    out_stream.close();
  };
  
  const int NUM_CHAR = 7;
  const char str[] = {"abcdefg"};
  func("file1.txt", str);

  FileList file_list("file_list.txt");
  FileSource file_source(file_list);
  CheckSum check_sum(file_source);
  char tmp1[NUM_CHAR], tmp2[NUM_CHAR];
  check_sum.next_source();
  EXPECT_EQ(check_sum.read(tmp1, NUM_CHAR), Error_t::Success);
  // for(int i=0; i< NUM_CHAR; i++){
  //   std::cout << tmp1[i];
  // }
  EXPECT_EQ(strncmp(tmp1,str,NUM_CHAR), 0);

  EXPECT_EQ(check_sum.read(tmp2, NUM_CHAR), Error_t::Success);
  // for(int i=0; i< NUM_CHAR; i++){
  //   std::cout << tmp2[i];
  // }
  EXPECT_EQ(strncmp(tmp1,str,NUM_CHAR), 0);
}
