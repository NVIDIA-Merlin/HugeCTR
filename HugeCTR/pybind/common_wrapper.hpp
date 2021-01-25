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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <HugeCTR/include/common.hpp>

namespace HugeCTR {

namespace python_lib {

void CommonPybind(pybind11::module& m) {
  m.attr("__version__") = std::to_string(HUGECTR_VERSION_MAJOR) 
	                  + "." + std::to_string(HUGECTR_VERSION_MINOR)
  	                  + "." + std::to_string(HUGECTR_VERSION_PATCH);
  pybind11::enum_<HugeCTR::Error_t>(m, "Error_t")
      .value("Success", HugeCTR::Error_t::Success)
      .value("FileCannotOpen", HugeCTR::Error_t::FileCannotOpen)
      .value("BrokenFile", HugeCTR::Error_t::BrokenFile)
      .value("OutOfMemory", HugeCTR::Error_t::OutOfMemory)
      .value("OutOfBound", HugeCTR::Error_t::OutOfBound)
      .value("WrongInput", HugeCTR::Error_t::WrongInput)
      .value("IllegalCall", HugeCTR::Error_t::IllegalCall)
      .value("NotInitialized", HugeCTR::Error_t::NotInitialized)
      .value("UnSupportedFormat", HugeCTR::Error_t::UnSupportedFormat)
      .value("InvalidEnv", HugeCTR::Error_t::InvalidEnv)
      .value("MpiError", HugeCTR::Error_t::MpiError)
      .value("CublasError", HugeCTR::Error_t::CublasError)
      .value("CudnnError", HugeCTR::Error_t::CudnnError)
      .value("CudaError", HugeCTR::Error_t::CudaError)
      .value("NcclError", HugeCTR::Error_t::NcclError)
      .value("DataCheckError", HugeCTR::Error_t::DataCheckError)
      .value("UnspecificError", HugeCTR::Error_t::UnspecificError)
      .export_values();
  pybind11::enum_<HugeCTR::Check_t>(m, "Check_t")
      .value("Sum", HugeCTR::Check_t::Sum)
      .value("None", HugeCTR::Check_t::None)
      .export_values();
  pybind11::enum_<HugeCTR::DataReaderSparse_t>(m, "DataReaderSparse_t")
      .value("Distributed", HugeCTR::DataReaderSparse_t::Distributed)
      .value("Localized", HugeCTR::DataReaderSparse_t::Localized)
      .export_values();
  pybind11::enum_<HugeCTR::DataReaderType_t>(m, "DataReaderType_t")
      .value("Norm", HugeCTR::DataReaderType_t::Norm)
      .value("Raw", HugeCTR::DataReaderType_t::Raw)
      .value("Parquet", HugeCTR::DataReaderType_t::Parquet)
      .export_values();
  pybind11::enum_<HugeCTR::SourceType_t>(m, "SourceType_t")
      .value("FileList", HugeCTR::SourceType_t::FileList)
      .value("Mmap", HugeCTR::SourceType_t::Mmap)
      .value("Parquet", HugeCTR::SourceType_t::Parquet)
      .export_values();
  pybind11::enum_<HugeCTR::Embedding_t>(m, "Embedding_t")
      .value("DistributedSlotSparseEmbeddingHash", HugeCTR::Embedding_t::DistributedSlotSparseEmbeddingHash)
      .value("LocalizedSlotSparseEmbeddingHash", HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingHash)
      .value("LocalizedSlotSparseEmbeddingOneHot", HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingOneHot)
      .export_values();
  pybind11::enum_<HugeCTR::Initializer_t>(m, "Initializer_t")
      .value("Default", HugeCTR::Initializer_t::Default)
      .value("Uniform", HugeCTR::Initializer_t::Uniform)
      .value("XavierNorm", HugeCTR::Initializer_t::XavierNorm)
      .value("XavierUniform", HugeCTR::Initializer_t::XavierUniform)
      .value("Zero", HugeCTR::Initializer_t::Zero)
      .export_values();
  pybind11::enum_<HugeCTR::Layer_t>(m, "Layer_t")
      .value("BatchNorm", HugeCTR::Layer_t::BatchNorm)
      .value("BinaryCrossEntropyLoss", HugeCTR::Layer_t::BinaryCrossEntropyLoss)
      .value("Reshape", HugeCTR::Layer_t::Reshape)
      .value("Concat", HugeCTR::Layer_t::Concat)
      .value("CrossEntropyLoss", HugeCTR::Layer_t::CrossEntropyLoss)
      .value("Dropout", HugeCTR::Layer_t::Dropout)
      .value("ELU", HugeCTR::Layer_t::ELU)
      .value("InnerProduct", HugeCTR::Layer_t::InnerProduct)
      .value("FusedInnerProduct", HugeCTR::Layer_t::FusedInnerProduct)
      .value("Interaction", HugeCTR::Layer_t::Interaction)
      .value("MultiCrossEntropyLoss", HugeCTR::Layer_t::MultiCrossEntropyLoss)
      .value("ReLU", HugeCTR::Layer_t::ReLU)
      .value("ReLUHalf", HugeCTR::Layer_t::ReLUHalf)
      .value("Sigmoid", HugeCTR::Layer_t::Sigmoid)
      .value("Slice", HugeCTR::Layer_t::Slice)
      .value("WeightMultiply", HugeCTR::Layer_t::WeightMultiply)
      .value("FmOrder2", HugeCTR::Layer_t::FmOrder2)
      .value("Add", HugeCTR::Layer_t::Add)
      .value("ReduceSum", HugeCTR::Layer_t::ReduceSum)
      .value("MultiCross", HugeCTR::Layer_t::MultiCross)
      .value("Cast", HugeCTR::Layer_t::Cast)
      .value("DotProduct", HugeCTR::Layer_t::DotProduct)
      .export_values();
  pybind11::class_<HugeCTR::DataReaderSparseParam>(m, "DataReaderSparseParam")
      .def(pybind11::init<HugeCTR::DataReaderSparse_t, int, int, int>(), pybind11::arg("type"),
           pybind11::arg("max_feature_num"), pybind11::arg("max_nnz"), pybind11::arg("slot_num"));
  pybind11::class_<HugeCTR::NameID>(m, "NameId")
      .def(pybind11::init<std::string, unsigned int>(), pybind11::arg("file_name"),
           pybind11::arg("id"));
  pybind11::enum_<HugeCTR::LrPolicy_t>(m, "LrPolicy_t")
      .value("fixed", HugeCTR::LrPolicy_t::fixed)
      .export_values();
  pybind11::enum_<HugeCTR::Optimizer_t>(m, "Optimizer_t")
      .value("Adam", HugeCTR::Optimizer_t::Adam)
      .value("MomentumSGD", HugeCTR::Optimizer_t::MomentumSGD)
      .value("Nesterov", HugeCTR::Optimizer_t::Nesterov)
      .value("SGD", HugeCTR::Optimizer_t::SGD)
      .export_values();
  pybind11::enum_<HugeCTR::Update_t>(m, "Update_t")
      .value("Local", HugeCTR::Update_t::Local)
      .value("Global", HugeCTR::Update_t::Global)
      .value("LazyGlobal", HugeCTR::Update_t::LazyGlobal)
      .export_values();
  pybind11::enum_<HugeCTR::Regularizer_t>(m, "Regularizer_t")
      .value("L1", HugeCTR::Regularizer_t::L1)
      .value("L2", HugeCTR::Regularizer_t::L2)
      .export_values();
}

}  // namespace python_lib

}  // namespace HugeCTR
