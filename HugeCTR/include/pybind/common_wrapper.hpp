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
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <collectives/all_reduce_comm.hpp>
#include <common.hpp>
#include <device_map.hpp>
#include <io/filesystem.hpp>
#include <metrics.hpp>

namespace HugeCTR {

namespace python_lib {

void CommonPybind(pybind11::module& m) {
  m.attr("__version__") = std::to_string(HUGECTR_VERSION_MAJOR) + "." +
                          std::to_string(HUGECTR_VERSION_MINOR) + "." +
                          std::to_string(HUGECTR_VERSION_PATCH);
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
      .value("CudaDriverError", HugeCTR::Error_t::CudaDriverError)
      .value("CudaRuntimeError", HugeCTR::Error_t::CudaRuntimeError)
      .value("NcclError", HugeCTR::Error_t::NcclError)
      .value("DataCheckError", HugeCTR::Error_t::DataCheckError)
      .value("UnspecificError", HugeCTR::Error_t::UnspecificError)
      .value("EndOfFile", HugeCTR::Error_t::EndOfFile)
      .export_values();
  pybind11::enum_<HugeCTR::Check_t>(m, "Check_t")
      .value("Sum", HugeCTR::Check_t::Sum)
      .value("Non", HugeCTR::Check_t::None)
      .export_values();
  pybind11::enum_<HugeCTR::DataReaderType_t>(m, "DataReaderType_t")
      .value("Norm", HugeCTR::DataReaderType_t::Norm)
      .value("Raw", HugeCTR::DataReaderType_t::Raw)
      .value("Parquet", HugeCTR::DataReaderType_t::Parquet)
      .value("RawAsync", HugeCTR::DataReaderType_t::RawAsync)
      .export_values();
  pybind11::enum_<HugeCTR::FileSystemType_t>(m, "FileSystemType_t")
      .value("Local", HugeCTR::FileSystemType_t::Local)
      .value("HDFS", HugeCTR::FileSystemType_t::HDFS)
      .value("S3", HugeCTR::FileSystemType_t::S3)
      .value("GCS", HugeCTR::FileSystemType_t::GCS)
      .value("Other", HugeCTR::FileSystemType_t::Other)
      .export_values();
  pybind11::enum_<HugeCTR::SourceType_t>(m, "SourceType_t")
      .value("FileList", HugeCTR::SourceType_t::FileList)
      .value("Mmap", HugeCTR::SourceType_t::Mmap)
      .value("Parquet", HugeCTR::SourceType_t::Parquet)
      .export_values();
  pybind11::enum_<HugeCTR::TrainPSType_t>(m, "TrainPSType_t")
      .value("Staged", HugeCTR::TrainPSType_t::Staged)
      .value("Cached", HugeCTR::TrainPSType_t::Cached)
      .export_values();
  pybind11::enum_<HugeCTR::Embedding_t>(m, "Embedding_t")
      .value("DistributedSlotSparseEmbeddingHash",
             HugeCTR::Embedding_t::DistributedSlotSparseEmbeddingHash)
      .value("LocalizedSlotSparseEmbeddingHash",
             HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingHash)
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
      .value("LayerNorm", HugeCTR::Layer_t::LayerNorm)
      .value("BinaryCrossEntropyLoss", HugeCTR::Layer_t::BinaryCrossEntropyLoss)
      .value("Reshape", HugeCTR::Layer_t::Reshape)
      .value("Select", HugeCTR::Layer_t::Select)
      .value("Concat", HugeCTR::Layer_t::Concat)
      .value("CrossEntropyLoss", HugeCTR::Layer_t::CrossEntropyLoss)
      .value("Dropout", HugeCTR::Layer_t::Dropout)
      .value("ElementwiseMultiply", HugeCTR::Layer_t::ElementwiseMultiply)
      .value("ELU", HugeCTR::Layer_t::ELU)
      .value("InnerProduct", HugeCTR::Layer_t::InnerProduct)
      .value("MLP", HugeCTR::Layer_t::MLP)
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
      .value("Softmax", HugeCTR::Layer_t::Softmax)
      .value("Gather", HugeCTR::Layer_t::Gather)
      .value("PReLU_Dice", HugeCTR::Layer_t::PReLU_Dice)
      .value("GRU", HugeCTR::Layer_t::GRU)
      .value("MatrixMultiply", HugeCTR::Layer_t::MatrixMultiply)
      .value("MultiHeadAttention", HugeCTR::Layer_t::MultiHeadAttention)
      .value("Scale", HugeCTR::Layer_t::Scale)
      .value("FusedReshapeConcat", HugeCTR::Layer_t::FusedReshapeConcat)
      .value("FusedReshapeConcatGeneral", HugeCTR::Layer_t::FusedReshapeConcatGeneral)
      .value("Sub", HugeCTR::Layer_t::Sub)
      .value("ReduceMean", HugeCTR::Layer_t::ReduceMean)
      .value("MultiCross", HugeCTR::Layer_t::MultiCross)
      .value("Cast", HugeCTR::Layer_t::Cast)
      .value("SequenceMask", HugeCTR::Layer_t::SequenceMask)
      .export_values();
  pybind11::class_<HugeCTR::DataReaderSparseParam>(m, "DataReaderSparseParam")
      .def(pybind11::init<const std::string&, const std::vector<int>&, bool, int>(),
           pybind11::arg("top_name"), pybind11::arg("nnz_per_slot"),
           pybind11::arg("is_fixed_length"), pybind11::arg("slot_num"))
      .def(pybind11::init<const std::string&, const int, bool, int>(), pybind11::arg("top_name"),
           pybind11::arg("nnz_per_slot"), pybind11::arg("is_fixed_length"),
           pybind11::arg("slot_num"));
  pybind11::enum_<HugeCTR::Alignment_t>(m, "Alignment_t")
      .value("Auto", HugeCTR::Alignment_t::Auto)
      .value("Non", HugeCTR::Alignment_t::None)
      .export_values();
  pybind11::class_<HugeCTR::AsyncParam>(m, "AsyncParam")
      .def(pybind11::init<int, int, int, int, int, bool, Alignment_t, bool, bool>(),
           pybind11::arg("num_threads"), pybind11::arg("num_batches_per_thread"),
           pybind11::arg("max_num_requests_per_thread") = 0, pybind11::arg("io_depth") = 0,
           pybind11::arg("io_alignment") = 0, pybind11::arg("shuffle"),
           pybind11::arg("aligned_type") = Alignment_t::None,
           pybind11::arg("multi_hot_reader") = true, pybind11::arg("is_dense_float") = true);
  pybind11::enum_<HugeCTR::LrPolicy_t>(m, "LrPolicy_t")
      .value("fixed", HugeCTR::LrPolicy_t::fixed)
      .export_values();
  pybind11::enum_<HugeCTR::Optimizer_t>(m, "Optimizer_t")
      .value("Ftrl", HugeCTR::Optimizer_t::Ftrl)
      .value("Adam", HugeCTR::Optimizer_t::Adam)
      .value("RMSProp", HugeCTR::Optimizer_t::RMSProp)
      .value("AdaGrad", HugeCTR::Optimizer_t::AdaGrad)
      .value("MomentumSGD", HugeCTR::Optimizer_t::MomentumSGD)
      .value("Nesterov", HugeCTR::Optimizer_t::Nesterov)
      .value("SGD", HugeCTR::Optimizer_t::SGD)
      .export_values();
  pybind11::enum_<HugeCTR::Update_t>(m, "Update_t")
      .value("Local", HugeCTR::Update_t::Local)
      .value("Global", HugeCTR::Update_t::Global)
      .value("LazyGlobal", HugeCTR::Update_t::LazyGlobal)
      .export_values();
  pybind11::enum_<HugeCTR::Activation_t>(m, "Activation_t")
      .value("Relu", HugeCTR::Activation_t::Relu)
      .value("Non", HugeCTR::Activation_t::None)
      .export_values();
  pybind11::enum_<HugeCTR::FcPosition_t>(m, "FcPosition_t")
      .value("Non", HugeCTR::FcPosition_t::None)
      .value("Head", HugeCTR::FcPosition_t::Head)
      .value("Body", HugeCTR::FcPosition_t::Body)
      .value("Tail", HugeCTR::FcPosition_t::Tail)
      .value("Isolated", HugeCTR::FcPosition_t::Isolated)
      .export_values();
  pybind11::enum_<HugeCTR::Regularizer_t>(m, "Regularizer_t")
      .value("L1", HugeCTR::Regularizer_t::L1)
      .value("L2", HugeCTR::Regularizer_t::L2)
      .export_values();
  pybind11::enum_<HugeCTR::metrics::RawType>(m, "MetricsRawType")
      .value("Loss", HugeCTR::metrics::RawType::Loss)
      .value("Pred", HugeCTR::metrics::RawType::Pred)
      .value("Label", HugeCTR::metrics::RawType::Label)
      .export_values();
  pybind11::enum_<HugeCTR::metrics::Type>(m, "MetricsType")
      .value("AUC", HugeCTR::metrics::Type::AUC)
      .value("AverageLoss", HugeCTR::metrics::Type::AverageLoss)
      .value("HitRate", HugeCTR::metrics::Type::HitRate)
      .value("NDCG", HugeCTR::metrics::Type::NDCG)
      .value("SMAPE", HugeCTR::metrics::Type::SMAPE)
      .export_values();
  pybind11::enum_<HugeCTR::DeviceMap::Layout>(m, "DeviceLayout")
      .value("LocalFirst", HugeCTR::DeviceMap::Layout::LOCAL_FIRST)
      .value("NodeFirst", HugeCTR::DeviceMap::Layout::NODE_FIRST)
      .export_values();
  pybind11::enum_<HugeCTR::AllReduceAlgo>(m, "AllReduceAlgo")
      .value("OneShot", HugeCTR::AllReduceAlgo::ONESHOT)
      .value("NCCL", HugeCTR::AllReduceAlgo::NCCL)
      .export_values();
  pybind11::enum_<HugeCTR::Distribution_t>(m, "Distribution_t")
      .value("Uniform", HugeCTR::Distribution_t::Uniform)
      .value("PowerLaw", HugeCTR::Distribution_t::PowerLaw)
      .export_values();
  pybind11::enum_<HugeCTR::PowerLaw_t>(m, "PowerLaw_t")
      .value("Long", HugeCTR::PowerLaw_t::Long)
      .value("Medium", HugeCTR::PowerLaw_t::Medium)
      .value("Short", HugeCTR::PowerLaw_t::Short)
      .value("Specific", HugeCTR::PowerLaw_t::Specific)
      .export_values();
  pybind11::enum_<HugeCTR::Tensor_t>(m, "Tensor_t")
      .value("Train", HugeCTR::Tensor_t::Train)
      .value("Evaluate", HugeCTR::Tensor_t::Evaluate)
      .export_values();
}

}  // namespace python_lib

}  // namespace HugeCTR
