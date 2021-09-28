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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <HugeCTR/include/collectives/all_reduce_comm.hpp>
#include <HugeCTR/include/common.hpp>
#include <HugeCTR/include/device_map.hpp>
#include <HugeCTR/include/embeddings/hybrid_embedding/utils.hpp>
#include <HugeCTR/include/metrics.hpp>

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
      .value("CudaError", HugeCTR::Error_t::CudaError)
      .value("NcclError", HugeCTR::Error_t::NcclError)
      .value("DataCheckError", HugeCTR::Error_t::DataCheckError)
      .value("UnspecificError", HugeCTR::Error_t::UnspecificError)
      .value("EndOfFile", HugeCTR::Error_t::EndOfFile)
      .export_values();
  pybind11::enum_<HugeCTR::Check_t>(m, "Check_t")
      .value("Sum", HugeCTR::Check_t::Sum)
      .value("Non", HugeCTR::Check_t::None)
      .export_values();
  pybind11::enum_<HugeCTR::DataReaderSparse_t>(m, "DataReaderSparse_t")
      .value("Distributed", HugeCTR::DataReaderSparse_t::Distributed)
      .value("Localized", HugeCTR::DataReaderSparse_t::Localized)
      .export_values();
  pybind11::enum_<HugeCTR::DataReaderType_t>(m, "DataReaderType_t")
      .value("Norm", HugeCTR::DataReaderType_t::Norm)
      .value("Raw", HugeCTR::DataReaderType_t::Raw)
      .value("Parquet", HugeCTR::DataReaderType_t::Parquet)
      .value("RawAsync", HugeCTR::DataReaderType_t::RawAsync)
      .export_values();
  pybind11::enum_<HugeCTR::SourceType_t>(m, "SourceType_t")
      .value("FileList", HugeCTR::SourceType_t::FileList)
      .value("Mmap", HugeCTR::SourceType_t::Mmap)
      .value("Parquet", HugeCTR::SourceType_t::Parquet)
      .export_values();
  pybind11::enum_<HugeCTR::Embedding_t>(m, "Embedding_t")
      .value("DistributedSlotSparseEmbeddingHash",
             HugeCTR::Embedding_t::DistributedSlotSparseEmbeddingHash)
      .value("LocalizedSlotSparseEmbeddingHash",
             HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingHash)
      .value("LocalizedSlotSparseEmbeddingOneHot",
             HugeCTR::Embedding_t::LocalizedSlotSparseEmbeddingOneHot)
      .value("HybridSparseEmbedding", HugeCTR::Embedding_t::HybridSparseEmbedding)
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
      .value("ElementwiseMultiply", HugeCTR::Layer_t::ElementwiseMultiply)
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
      .value("Softmax", HugeCTR::Layer_t::Softmax)
      .value("Gather", HugeCTR::Layer_t::Gather)
      .value("PReLU_Dice", HugeCTR::Layer_t::PReLU_Dice)
      .value("GRU", HugeCTR::Layer_t::GRU)
      .value("Scale", HugeCTR::Layer_t::Scale)
      .value("FusedReshapeConcat", HugeCTR::Layer_t::FusedReshapeConcat)
      .value("FusedReshapeConcatGeneral", HugeCTR::Layer_t::FusedReshapeConcatGeneral)
      .value("Sub", HugeCTR::Layer_t::Sub)
      .value("ReduceMean", HugeCTR::Layer_t::ReduceMean)
      .value("MultiCross", HugeCTR::Layer_t::MultiCross)
      .value("Cast", HugeCTR::Layer_t::Cast)
      .value("DotProduct", HugeCTR::Layer_t::DotProduct)
      .export_values();
  pybind11::class_<HugeCTR::DataReaderSparseParam>(m, "DataReaderSparseParam")
      .def(pybind11::init<const std::string&, const std::vector<int>&, bool, int>(),
           pybind11::arg("top_name"), pybind11::arg("nnz_per_slot"),
           pybind11::arg("is_fixed_length"), pybind11::arg("slot_num"))
      .def(pybind11::init<const std::string&, const int, bool, int>(), pybind11::arg("top_name"),
           pybind11::arg("nnz_per_slot"), pybind11::arg("is_fixed_length"),
           pybind11::arg("slot_num"));
  pybind11::class_<HugeCTR::AsyncParam>(m, "AsyncParam")
      .def(pybind11::init<int, int, int, int, int, bool, Alignment_t>(),
           pybind11::arg("num_threads"), pybind11::arg("num_batches_per_thread"),
           pybind11::arg("io_block_size"), pybind11::arg("io_depth"), pybind11::arg("io_alignment"),
           pybind11::arg("shuffle"), pybind11::arg("aligned_type"));
  pybind11::class_<HugeCTR::HybridEmbeddingParam>(m, "HybridEmbeddingParam")
      .def(pybind11::init<size_t, int64_t, double, double, double, double, bool, bool,
                          hybrid_embedding::CommunicationType,
                          hybrid_embedding::HybridEmbeddingType>(),
           pybind11::arg("max_num_frequent_categories"),
           pybind11::arg("max_num_infrequent_samples"), pybind11::arg("p_dup_max"),
           pybind11::arg("max_all_reduce_bandwidth"), pybind11::arg("max_all_to_all_bandwidth"),
           pybind11::arg("efficiency_bandwidth_ratio"),
           pybind11::arg("use_train_precompute_indices"), pybind11::arg("use_eval_precompute_indices"),
           pybind11::arg("communication_type"), pybind11::arg("hybrid_embedding_type"));
  pybind11::enum_<HugeCTR::LrPolicy_t>(m, "LrPolicy_t")
      .value("fixed", HugeCTR::LrPolicy_t::fixed)
      .export_values();
  pybind11::enum_<HugeCTR::Optimizer_t>(m, "Optimizer_t")
      .value("Adam", HugeCTR::Optimizer_t::Adam)
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
  pybind11::enum_<HugeCTR::Alignment_t>(m, "Alignment_t")
      .value("Auto", HugeCTR::Alignment_t::Auto)
      .value("Non", HugeCTR::Alignment_t::None)
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
      .export_values();
  pybind11::enum_<HugeCTR::DeviceMap::Layout>(m, "DeviceLayout")
      .value("LocalFirst", HugeCTR::DeviceMap::Layout::LOCAL_FIRST)
      .value("NodeFirst", HugeCTR::DeviceMap::Layout::NODE_FIRST)
      .export_values();
  pybind11::enum_<HugeCTR::AllReduceAlgo>(m, "AllReduceAlgo")
      .value("OneShot", HugeCTR::AllReduceAlgo::ONESHOT)
      .value("NCCL", HugeCTR::AllReduceAlgo::NCCL)
      .export_values();
  pybind11::enum_<HugeCTR::hybrid_embedding::HybridEmbeddingType>(m, "HybridEmbeddingType")
      .value("Distributed", HugeCTR::hybrid_embedding::HybridEmbeddingType::Distributed)
      .export_values();
  pybind11::enum_<HugeCTR::hybrid_embedding::CommunicationType>(m, "CommunicationType")
      .value("IB_NVLink_Hier", HugeCTR::hybrid_embedding::CommunicationType::IB_NVLink_Hier)
      .value("IB_NVLink", HugeCTR::hybrid_embedding::CommunicationType::IB_NVLink)
      .value("NVLink_SingleNode", HugeCTR::hybrid_embedding::CommunicationType::NVLink_SingleNode)
      .export_values();
}

}  // namespace python_lib

}  // namespace HugeCTR
