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

#ifndef EMBEDDING_LOOKUPER_H
#define EMBEDDING_LOOKUPER_H

#include "operation/operation.h"
#include "parameters/param_interface.h"

namespace SparseOperationKit {

class EmbeddingLookuper : public Operation {
public:
    EmbeddingLookuper(ConstructionContext_t construction_context, std::shared_ptr<ParamInterface> param);

    // help param to save parameters from GPU to file
    /**
    * by default, it return false, which means it does not help ParamInterface to save parameters to file.
    * if it want to help ParamInterface to save to file, then override this function and return true.
    */
    virtual bool save_params(const std::string filepath) const;

    // help Param to restore parameters from CPU tensor to GPU memory 
    virtual void restore_params(const std::shared_ptr<Tensor> &keys, 
                                const std::shared_ptr<Tensor> &embedding_values,
                                const size_t num_total_keys) = 0;

protected:
    std::shared_ptr<ParamInterface> param_;
};

} // namespace SparseOperationKit

#endif // EMBEDDING_LOOKUPER_H