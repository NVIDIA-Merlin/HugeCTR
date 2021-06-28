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

#ifndef OPERATION_H
#define OPERATION_H

#include "tensor_buffer/general_buffer2.hpp"
#include "operation/op_context.h"
#include "operation/construction_context.h"
#include <vector>
#include <memory>

namespace SparseOperationKit {
/*
* This class is the interface to represent an operation 
* used inside embedding layer.
*/
class Operation {
protected:
    template <typename T>
    using Tensor2 = HugeCTR::Tensor2<T>;
    template <typename T>
    using Tensors2 = HugeCTR::Tensors2<T>;

    ConstructionContext_t base_context() const;
public:
    explicit Operation(ConstructionContext_t context);
    virtual ~Operation() {}
    void AllocateForwardSpaces(size_t const global_batch_size);
    virtual void allocate_forward_spaces(size_t const global_batch_size) = 0;
    void AllocateBackwardSpaces(size_t const global_batch_size);
    virtual void allocate_backward_spaces(size_t const global_batch_size) = 0;
    void Forward(const Context_t &replica_context, const bool training);
    virtual void forward(const Context_t &replica_context, const bool training) = 0;
    void Backward(const Context_t &replica_context);
    virtual void backward(const Context_t &replica_context) = 0;

    void set_next(std::shared_ptr<Operation> operation);
private:
    std::shared_ptr<Operation> next_op_ = nullptr;
    ConstructionContext_t base_context_;
};

using Dispatcher = Operation;

} // namespace SparseOperationKit

#endif // OPERATION_H