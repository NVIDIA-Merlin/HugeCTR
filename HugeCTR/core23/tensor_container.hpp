#pragma once

#include <algorithm>
#include <base/debug/logger.hpp>
#include <core23/allocator_factory.hpp>
#include <core23/low_level_primitives.hpp>
#include <core23/shape.hpp>
#include <core23/tensor.hpp>
#include <memory>
#include <vector>

namespace HugeCTR {
namespace core23 {

template <typename BuiltInType, int64_t TensorDims, int64_t ContainerDims>
class TensorContainer : public TensorView<Tensor, ContainerDims> {
 public:
  using Base = TensorView<Tensor, ContainerDims>;
  using TargetTensorView = TensorView<BuiltInType, TensorDims>;
  using FlattenedTensorView = TensorView<BuiltInType, 1>;
  using View = TensorView<TargetTensorView, ContainerDims>;

  TensorContainer(void* d_workspace, std::vector<Tensor>&& tensors, Shape shape)
      : Base(tensors.data(), shape.data()),
        tensors_(std::move(tensors)),
        shape_(shape),
        tensor_view_ptrs_(static_cast<TargetTensorView*>(d_workspace)),
        viewed_(false) {
    initialize();
  }
  TensorContainer(std::vector<Tensor>&& tensors, Shape shape)
      : TensorContainer(nullptr, std::move(tensors), shape) {}
  TensorContainer(void* d_workspace, const std::vector<Tensor>& tensors, Shape shape)
      : Base(nullptr, shape.data()),
        tensors_(tensors),
        shape_(shape),
        tensor_view_ptrs_(static_cast<TargetTensorView*>(d_workspace)),
        viewed_(false) {
    Base::data_ = tensors_.data();
    initialize();
  }
  TensorContainer(const std::vector<Tensor>& tensors, Shape shape)
      : TensorContainer(nullptr, tensors, shape) {}
  TensorContainer(const TensorContainer& other)
      : TensorContainer(other.tensor_view_ptrs_, other.tensors_, other.shape_) {
    viewed_ = other.viewed_;
  }
  TensorContainer(TensorContainer&& other) = delete;
  TensorContainer& operator=(const TensorContainer& other) = delete;
  TensorContainer& operator=(TensorContainer&& other) = delete;

  ~TensorContainer() {
    if (allocator_) {
      allocator_->deallocate(tensor_view_ptrs_);
    }
  }

  FlattenedTensorView flatten() const {
    std::vector<Tensor> tensors = tensors_;
    std::sort(tensors.begin(), tensors.end(),
              [](const Tensor& lhs, const Tensor& rhs) { return lhs.data() < rhs.data(); });
    uint8_t* start_addr = nullptr;
    uint8_t* next_addr = nullptr;
    for (auto& t : tensors) {
      uint8_t* ptr = static_cast<uint8_t*>(t.data());
      if (next_addr) {
        HCTR_THROW_IF((next_addr != ptr) &&
                          !(next_addr < ptr && next_addr + t.my_params().alignment() >= ptr),
                      HugeCTR::Error_t::IllegalCall,
                      "Tensors cannot be flatten because their data are not contiguous enough");
      } else {
        start_addr = ptr;
      }
      next_addr = ptr + t.num_bytes();
    }
    int64_t num_bytes = reinterpret_cast<int64_t>(next_addr - start_addr);
    HCTR_THROW_IF(
        num_bytes % sizeof(BuiltInType), HugeCTR::Error_t::IllegalCall,
        "The number of bytes in total is not consistent with the BuiltInType of the Tensors");
    int64_t num_elements = num_bytes / sizeof(BuiltInType);
    return FlattenedTensorView(reinterpret_cast<BuiltInType*>(start_addr),
                               Shape({num_elements}).data());
  }

  View view() const {
    if (viewed_ == false) {
      std::vector<TargetTensorView> host_tensor_views;
      std::transform(tensors_.begin(), tensors_.end(), std::back_inserter(host_tensor_views),
                     [](const Tensor& tensor) { return tensor.view<BuiltInType, TensorDims>(); });
      int64_t size = sizeof(TargetTensorView) * shape_.size();
      if (allocator_) {
        tensor_view_ptrs_ = static_cast<TargetTensorView*>(allocator_->allocate(size));
      }
      auto dst_device = tensors_.begin()->device();
      auto src_device = Device(DeviceType::CPU);
      copy_sync(tensor_view_ptrs_, host_tensor_views.data(), size, dst_device, src_device);
      viewed_ = true;
    }
    return View(tensor_view_ptrs_, shape_.data());
  }

  Tensor at(int64_t t) const { return tensors_.at(t); }

 private:
  std::unique_ptr<Allocator> create_allocator() {
    AllocatorParams allocator_params;
    auto device = tensors_.begin()->device();
    auto data_type = tensors_.begin()->data_type();
    for (auto it = tensors_.begin()++; it != tensors_.end(); ++it) {
      HCTR_THROW_IF(data_type != it->data_type(), HugeCTR::Error_t::IllegalCall,
                    "Tensors with different data types cannot be included in the same "
                    "TensorContainer");
    }
    return GetAllocator(allocator_params, device);
  }

  void initialize() {
    HCTR_THROW_IF(tensors_.empty(), HugeCTR::Error_t::IllegalCall,
                  "A TensorContainer cannot be empty");
    HCTR_THROW_IF(ContainerDims != shape_.dims(), HugeCTR::Error_t::IllegalCall,
                  "ContainerDims is inconsistent with the specified Shape's");
    HCTR_THROW_IF(static_cast<int64_t>(tensors_.size()) != shape_.size(),
                  HugeCTR::Error_t::IllegalCall,
                  "The number of Tensors is inconsistent with the specified Shape");

    if (tensor_view_ptrs_ == nullptr) {
      allocator_ = create_allocator();
    }
  }

  std::vector<Tensor> tensors_;
  Shape shape_;
  mutable TargetTensorView* tensor_view_ptrs_;
  mutable bool viewed_;

  std::unique_ptr<Allocator> allocator_;
};

}  // namespace core23
}  // namespace HugeCTR