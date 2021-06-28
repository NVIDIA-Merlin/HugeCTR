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

#include "parameters/param_interface.h"
#include "common.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/variant_op_registry.h"

namespace SparseOperationKit {

void ParamInterface::set_user(std::shared_ptr<EmbeddingLayer>& embedding) {
    // It is not compulsory for the subclass to override this function.
    throw std::runtime_error(ErrorBase + "Not implemented.");
}

std::shared_ptr<Tensor>& ParamInterface::get_tensor(const size_t local_replica_id) {
    return get_embedding_table_tensor(local_replica_id);
}

class ParamVariantWrapper {
public:
    ParamVariantWrapper() : param_(nullptr) {}
    explicit ParamVariantWrapper(const std::shared_ptr<ParamInterface> param) : param_(param) {}
    ParamVariantWrapper(const ParamVariantWrapper& other): param_(other.param_) {}
    ParamVariantWrapper& operator=(ParamVariantWrapper&& other) {
        if (&other == this) return *this;
        param_ = other.param_;
        return *this;
    }
    ParamVariantWrapper& operator=(const ParamVariantWrapper& other) = delete;

    std::shared_ptr<ParamInterface> get() const { return param_; }

    ~ParamVariantWrapper() = default;
    tensorflow::string TypeName() const { return "EmbeddingPlugin::ParamVariantWrapper"; }
    void Encode(tensorflow::VariantTensorData* data) const {
        LOG(ERROR) << "The Encode() method is not implemented for "
                      "ParamVariantWrapper objects.";
    }
    bool Decode(const tensorflow::VariantTensorData& data) {
        LOG(ERROR) << "The Decode() method is not implemented for "
                      "ParamVariantWrapper objects.";
        return false;
    }

private:
    std::shared_ptr<ParamInterface> param_;
};


void GetParamFromVariantTensor(const tensorflow::Tensor* tensor,
                               std::shared_ptr<ParamInterface>& out_param) {
    if (!(tensor->dtype() == tensorflow::DT_VARIANT && 
          tensorflow::TensorShapeUtils::IsScalar(tensor->shape()))) {
        throw std::runtime_error(ErrorBase + "Param tensor must be a scalar of dtype DT_VARIANT.");
    }
    const tensorflow::Variant& variant = tensor->scalar<tensorflow::Variant>()();
    const ParamVariantWrapper* wrapper = variant.get<ParamVariantWrapper>();
    if (nullptr == wrapper) throw std::runtime_error(ErrorBase + "Tensor must be a EmbeddingPlugin::Param object.");
    out_param = wrapper->get();
    if (!out_param) throw std::runtime_error(ErrorBase + "read empty param pointer.");
}

void StoreParamInVariantTensor(const std::shared_ptr<ParamInterface>& param, 
                               tensorflow::Tensor* tensor) {
    if (!(tensor->dtype() == tensorflow::DT_VARIANT &&
          tensorflow::TensorShapeUtils::IsScalar(tensor->shape()))) {
        throw std::runtime_error(ErrorBase + "Param tensor must be a scalar of dtype DT_VARIANT.");
    }
    tensor->scalar<tensorflow::Variant>()() = ParamVariantWrapper(param);
}


} // namespace SparseOperationKit