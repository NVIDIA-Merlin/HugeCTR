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

#include <base/debug/logger.hpp>
#include <io/filesystem.hpp>
#include <io/hadoop_filesystem.hpp>

namespace HugeCTR {

FileSystem* DataSourceParams::create() const {
  switch (type) {
    case DataSourceType_t::HDFS:
#ifdef ENABLE_HDFS
      return new HadoopFileSystem{server, port};
#else
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Please install Hadoop and compile HugeCTR with ENABLE_HDFS to use HDFS "
                     "functionalities.");
#endif
    default:
      HCTR_OWN_THROW(Error_t::WrongInput, "Unsupproted filesystem.");
  }

  return nullptr;
}

}  // namespace HugeCTR
