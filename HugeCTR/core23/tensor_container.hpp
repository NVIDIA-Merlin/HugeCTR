#pragma once

#include <algorithm>
#include <core23/allocator_factory.hpp>
#include <core23/logger.hpp>
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
        flattened_start_addr_(nullptr),
        flattened_num_elements_(0),
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
        flattened_start_addr_(nullptr),
        flattened_num_elements_(0),
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
  TensorContainer& operator=(const TensorContainer& other) {
    if (this != &other) {
      this->data_ = const_cast<decltype(this->data_)>(other.tensors_.data());
      for (int64_t dim = 0; dim < ContainerDims; dim++) {
        this->shape_[dim] = other.shape_[dim];
        this->strides_[dim] = other.strides_[dim];
        this->offsets_[dim] = other.offsets_[dim];
      }
      tensors_ = other.tensors_;
      shape_ = other.shape_;
      if (tensor_view_ptrs_ && allocator_) {
        allocator_->deallocate(tensor_view_ptrs_);
      }
      tensor_view_ptrs_ = other.tensor_view_ptrs_;
      flattened_start_addr_ = other.flattened_start_addr_;
      flattened_num_elements_ = other.flattened_num_elements_;
      viewed_ = false;
      allocator_ = create_allocator();
    }
    return *this;
  }
  TensorContainer(TensorContainer&& other) = delete;
  TensorContainer& operator=(TensorContainer&& other) = delete;
  // TODO: we should remove this constructor which is a kind of temporary WAR
  TensorContainer() : Base(nullptr, nullptr), tensor_view_ptrs_(nullptr), viewed_(false) {}

  ~TensorContainer() {
    if (tensor_view_ptrs_ && allocator_) {
      allocator_->deallocate(tensor_view_ptrs_);
    }
  }

  FlattenedTensorView flatten() const {
    if (!flattened_start_addr_) {
      HCTR_THROW_IF(tensors_.empty() || Base::data_ == nullptr, HugeCTR::Error_t::IllegalCall,
                    "This empty TensorContainer cannot be flattened");

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
      flattened_start_addr_ = reinterpret_cast<BuiltInType*>(start_addr);
      flattened_num_elements_ = num_bytes / sizeof(BuiltInType);
    }
    return FlattenedTensorView(flattened_start_addr_, Shape({flattened_num_elements_}).data());
  }

  View view() const {
    HCTR_THROW_IF(tensors_.empty() || Base::data_ == nullptr, HugeCTR::Error_t::IllegalCall,
                  "This empty TensorContainer cannot be flattened");

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

  BuiltInType* data() const { return flatten().data(); }

  int64_t num_elements() const { return flatten().size(0); }

  int64_t num_bytes() const { return num_elements() * sizeof(BuiltInType); }

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
    if (!tensors_.empty()) {
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
  }

  std::vector<Tensor> tensors_;
  Shape shape_;
  mutable TargetTensorView* tensor_view_ptrs_;
  mutable BuiltInType* flattened_start_addr_;
  mutable int64_t flattened_num_elements_;
  mutable bool viewed_;

  std::unique_ptr<Allocator> allocator_;
};

}  // namespace core23
}  // namespace HugeCTR
