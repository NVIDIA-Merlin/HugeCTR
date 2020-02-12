/* Copyright 2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#pragma once

#include <assert.h>
#include "../FasterComm.h"
#include "FasterGossipCommTraits.h"
#include "gossip/include/plan_parser.hpp"
#include "../../nv_util.h"

namespace FasterGossipComm{

template<typename data_t_, typename GossipCommTraits>
class FasterGossipComm;

template<typename data_t_>
class FasterGossipComm<data_t_, FasterGossipCommAll2AllTraits<data_t_>> : public FasterComm::FasterComm{
public:
    using GossipCommTraits = FasterGossipCommAll2AllTraits<data_t_>;
    using data_t = typename GossipCommTraits::data_t;
    using param_t = typename GossipCommTraits::param_t;
    using context_t = typename GossipCommTraits::context_t;
    using executor_t = typename GossipCommTraits::executor_t;
    using transfer_plan_t = typename GossipCommTraits::transfer_plan_t;
    using transfer_plan_util_t = typename GossipCommTraits::transfer_plan_util_t;

    FasterGossipComm(const std::string& plan_file, const std::vector<gossip::gpu_id_t>& GPU_list) : 
                     GPU_list_(GPU_list), buf_(GPU_list_.size()), src_len_(GPU_list_.size()),
                     dst_len_(GPU_list_.size()), buf_len_(GPU_list_.size()){

        // Parsing user input transfer plan file
        transfer_plan_ = new transfer_plan_t( parse_plan(plan_file.c_str()) );
        // Verify the transfer plan for the communication pattern
        transfer_plan_util_t::verify_plan(*transfer_plan_);
        // Number of GPUs must consist from transfer plan and gpu_list
        assert( ( transfer_plan_->num_gpus() == GPU_list_.size() ) && 
                "The # of GPU is inconsist between transfer plan and GPU list!\n");
        
        num_gpu_ = transfer_plan_->num_gpus();

        // Assert that transfer plan is valid
        assert( transfer_plan_->valid() && "The transfer plan is not valid!\n");
    
        // Create context
        context_ = new context_t(GPU_list_);
        // Create communication executor
        executor_ = new executor_t(*context_, *transfer_plan_);

    }

    ~FasterGossipComm(){
        delete transfer_plan_;
        delete context_;
        delete executor_;
    }

    void Initialize(const std::vector<data_t *>& src,
                    const std::vector<data_t *>& dst,
                    const std::vector<std::vector<size_t>>& table){

        // Device restorer
        nv::CudaDeviceRestorer dev_restorer;

        // Set user specific src and dst GPU buffers and partition table
        parameters_.set(src, dst, table);

        // Calculate send/recv buffer length on each GPUs
        for(gossip::gpu_id_t i = 0; i < num_gpu_; i++){
            src_len_[i] = 0;
            dst_len_[i] = 0;
            for( gossip::gpu_id_t j = 0; j < num_gpu_; j++){
                src_len_[i] += parameters_.table_[i][j];
                dst_len_[i] += parameters_.table_[j][i];
            }
        }

        // Calculate temp buffers required on each GPU for communication 
        std::vector<size_t> bufs_lens_calc = executor_->calcBufferLengths(parameters_.table_);

        // Temp buffer length
        buf_len_ = bufs_lens_calc;

        // Allocate local GPU temp buffers for communication
        for(gossip::gpu_id_t i = 0; i < num_gpu_; i++){
            // Allocate temp buffers on each GPU
            CUDA_CHECK( cudaSetDevice( context_->get_device_id(i) ) );
            CUDA_CHECK( cudaMalloc( &buf_[i], sizeof(data_t) * bufs_lens_calc[i] ) );
        }
    }

    void execAsync(){
        executor_ -> execAsync(parameters_.src_, src_len_, parameters_.dst_, dst_len_, buf_, buf_len_, parameters_.table_);
    }

    void sync(){
        executor_ -> sync();
    }

    void reset(){

        // Device restorer
        nv::CudaDeviceRestorer dev_restorer;

        // Free local GPU temp buffers for communication
        for(gossip::gpu_id_t i = 0; i < num_gpu_; i++){
            // Free temp buffers on each GPU
            CUDA_CHECK( cudaSetDevice( context_->get_device_id(i) ) );
            CUDA_CHECK( cudaFree( buf_[i] ) );
        }
    
    }


private:
    transfer_plan_t* transfer_plan_;
    context_t* context_;
    std::vector<gossip::gpu_id_t> GPU_list_;
    gossip::gpu_id_t num_gpu_;
    executor_t* executor_;
    param_t parameters_;
    std::vector<data_t *> buf_;
    std::vector<size_t> src_len_;
    std::vector<size_t> dst_len_;
    std::vector<size_t> buf_len_;

};

} // namespace