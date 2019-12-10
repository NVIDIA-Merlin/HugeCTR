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

#pragma once

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/checker.hpp"
#include "HugeCTR/include/source.hpp"

namespace HugeCTR {

class CheckSum: public Checker {
private:
  const int MAX_TRY_{10};
  int counter_; /**< */
  char accum_; /**< */
public:
  CheckSum(Source& src):  Checker(src), counter_(0), accum_(0){}
  /**
   * Read "bytes_to_read" byte to the memory associated to ptr.
   * Users don't need to manualy maintain the check bit offset, just specify
   * number of bytes you really want to see in ptr.
   * @param ptr pointer to user located buffer
   * @param bytes_to_read bytes to read
   * @return `DataCheckError` `OutOfBound` `Success` `UnspecificError`
   */
  Error_t read(char* ptr, size_t bytes_to_read) noexcept{
    try{
      //if counter == 0 read int length and char check_sum
      if(counter_ == 0){
	Checker::src_.read(reinterpret_cast<char*>(&counter_), sizeof(int));
      }
      counter_ -= bytes_to_read;
      //if user read more data than expected, return `BrokenFile`. 
      //User should check this error and call next_source to new a source.
      if(counter_ < 0){
	CK_THROW_(Error_t::BrokenFile, "counter_ < 0");
      }
      else{
	Checker::src_.read(ptr, bytes_to_read);
	for(unsigned int i=0; i<bytes_to_read; i++){
	  accum_ += ptr[i];
	}
	//do checksum when counter_ == 0.
	if(counter_ == 0){
	  char check_sum = 0;
	  Checker::src_.read(reinterpret_cast<char*>(&check_sum), sizeof(char));
	  if(accum_ == check_sum){
	    accum_ = 0;
	    return Error_t::Success;
	  }
	  else{
	    //std::cout << "check_error:" << static_cast<int>(accum_) << std::endl;
	    accum_ = 0;
	    return Error_t::DataCheckError;
	  }
	}
	else{
	  return Error_t::Success;
	}
      }
      return Error_t::UnspecificError;
    }
    catch (const std::runtime_error& rt_err){
      std::cerr << rt_err.what() << std::endl;
      return Error_t::BrokenFile;
    }

  }

  /**
   * Start a new file to read.
   * @return `FileCannotOpen` or `UnspecificError`
   */
  void next_source(){
    // initialize
    counter_ = 0;
    accum_ = 0;
    for(int i = MAX_TRY_; i > 0; i++){
      if(Checker::src_.next_source() == Error_t::Success){
	return;
      }
    }
    CK_THROW_(Error_t::FileCannotOpen, "Checker::src_.next_source() == Error_t::Success failed");
  }
};

} //namespace HugeCTR
