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

#ifndef OPERATION_INTERFACE_H
#define OPERATION_INTERFACE_H

/*
* This file is the interface header for expanding operations.
*
* In this framework, an embedding layer is made of there key components,
* which are 'input_dispathcer', 'embedding_lookuper', 'output_dispatcher', 
* respectively.
*
* 1. Input_dispathcer will dispatch input data to each GPU, 
* it will convert the computation from data-parallel to model-parallel. 
* 2. Embedding_lookuper is responsible for local-gpu computation, and looking 
* up embedding vector, which is key->embedding_vector.
* 3. Output_dispatcher will dispatch output data to each GPU,
* it will convert the computation from model-parallel to data-parallel.
*
* To increase the flexibility of this framework, operations can be linked 
* to input_dispathcer or output_dispatcher. For example, the computation pipeline
* might look like:
*   1. input_dispathcer->embedding_lookuper->output_dispatcher
*   2. input_dispathcer->op1->op2->op3->embedding_lookuper->output_dispatcher->op4->op5
* The number of ops after input_dispathcer and output_dispatcher is not limited.
*
* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
* The following steps describes how to add new operations to this framework.
* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
* 
* ----------------------------------
*   input / output dispathcer
* ----------------------------------
* 1. create a cpp source file. For example, MyInputDispathcer.cc
* 2. "operation_interface.h" should be included in that souce file.
* 3. inherit from Dispatcher class, and override methods: 
*   allocate_forward_spaces, allocate_backward_spaces, forward, backward
*   for example: 
*   class MyInputDispathcer : public Dispatcher {
*   public:
*       explicit MyInputDispathcer(ConstructionContext_t context) : Dispatcher(context) {}
*       void allocate_forward_spaces(const size_t global_batch_size) override {
*           // reserve spaces used in forward propagation.
*           // this function will be called only once, so that forward spaces used by all local GPUs
*           // should be reserved.
*       }
*       void allocate_backward_spaces(const size_t global_batch_size) override {
*           // reserve spaces used in backward propagation.
*           // this function will be called only once, so that backward spaces used by all local GPUs
*           // should be reserved.
*       }
*       void forward(Context_t replica_context, const bool training) override {
*           // do forward propagation.    
*           // this function will be called by multiple CPU-threads if there are multiple GPUs.
*       }
*       void backward(Context_t replica_context) override {
*           // do backward propagation.
*           // this function will be called by multiple CPU-threads if there are multiple GPUs.
*       }
*   };
*    
* 4. register this dispatcher by calling 'REGISTER_INPUT_DISPATCHER_BUILDER' macro in the cpp file.
*   for example: REGISTER_INPUT_DISPATCHER_BUILDER("MyInputDispathcer", MyInputDispathcer);
* 5. "MyInputDispathcer" will be used to find this dispatcher in python script.
*
* ----------------------------------
*        embedding_lookuper
* ----------------------------------
* 1. create a cpp source file. For example, MyLookuper.cc
* 2. "operation_interface.h" should be included in that souce file.
* 3. inherit from EmbeddingLookuper class, and override methods: 
*   allocate_forward_spaces, allocate_backward_spaces, forward, backward, 
*   load_tensors_to_memory
*   for example: 
*   class MyLookuper : public EmbeddingLookuper {
*   public:
*       MyLookuper(ConstructionContext_t context, std::shared_ptr<ParamInterface> param) 
            : Dispatcher(context, param) {}
*       void allocate_forward_spaces(const size_t global_batch_size) override {
*           // reserve spaces used in forward propagation.
*           // this function will be called only once, so that forward spaces used by all local GPUs
*           // should be reserved.
*       }
*       void allocate_backward_spaces(const size_t global_batch_size) override {
*           // reserve spaces used in backward propagation.
*           // this function will be called only once, so that backward spaces used by all local GPUs
*           // should be reserved.
*       }
*       void forward(Context_t replica_context, const bool training) override {
*           // do forward propagation.    
*           // this function will be called by multiple CPU-threads if there are multiple GPUs.
*       }
*       void backward(Context_t replica_context) override {
*           // do backward propagation.
*           // this function will be called by multiple CPU-threads if there are multiple GPUs.
*       }
*       void load_tensors_to_memory(const std::vector<std::shared_ptr<Tensor>>& tensors) override {
*           // load tensors to initialize trainable parameters.
*           // the input tensors is a vector of tensor whose shape is [dim, embedding_vec_size],
*           // and all the the sum of dim is <= max_vocabulary_size_in_total.
*           // this function will be called by one CPU-threads.
*       }
*   };
*    
* 4. register this embedding_lookuper by calling 'REGISTER_EMB_LOOKUPER_BUILDER' macro in the cpp file.
*   for example: REGISTER_EMB_LOOKUPER_BUILDER("MyLookuper", MyLookuper);
* 5. "MyLookuper" will be used to find this embedding_lookuper in python script.
*
* ----------------------------------
*           operation
* ----------------------------------
* 1. create a cpp source file. For example, MyOperation.cc
* 2. "operation_interface.h" should be included in that souce file.
* 3. inherit from Operation class, and override methods: 
*   allocate_forward_spaces, allocate_backward_spaces, forward, backward
*   for example: 
*   class MyOperation : public Operation {
*   public:
*       explicit MyOperation(ConstructionContext_t context) : Operation(context) {}
*       void allocate_forward_spaces(const size_t global_batch_size) override {
*           // reserve spaces used in forward propagation.
*           // this function will be called only once, so that forward spaces used by all local GPUs
*           // should be reserved.
*       }
*       void allocate_backward_spaces(const size_t global_batch_size) override {
*           // reserve spaces used in backward propagation.
*           // this function will be called only once, so that backward spaces used by all local GPUs
*           // should be reserved.
*       }
*       void forward(Context_t replica_context, const bool training) override {
*           // do forward propagation.    
*           // this function will be called by multiple CPU-threads if there are multiple GPUs.
*       }
*       void backward(Context_t replica_context) override {
*           // do backward propagation.
*           // this function will be called by multiple CPU-threads if there are multiple GPUs.
*       }
*   };
*    
* 4. register this dispatcher by calling 'REGISTER_OPERATION_BUILDER' macro in the cpp file.
*   for example: REGISTER_OPERATION_BUILDER("MyOperation", MyOperation);
* 5. "MyOperation" will be used to find this dispatcher in python script.
*/

#include "operation/operation.h"
#include "operation/operation_helper.h"

#endif // OPERATION_INTERFACE_H