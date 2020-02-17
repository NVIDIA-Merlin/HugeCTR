/* Copyright 2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#pragma once

#include "gossip/include/gossip.cuh"

namespace FasterGossipComm{

// Common Macros define for Gossip communications 
enum class CommType{All2All}; // Define Macros for communication type

// Traits for All2All communication using GOSSIP library
template<typename data_t_>
class FasterGossipCommAll2AllTraits{
public:

    enum {
        TYPE = CommType::All2All // 0 stands for 
    };

    using data_t = data_t_;

    // The parameters needed by the GOSSIP All2All communication
    struct FasterGossipCommAll2AllParam
    {
        // Src buffers on each GPU
        std::vector<data_t *> src_;
        // Dest buffers on each GPU
        std::vector<data_t *> dst_;
        // Partition table: How many elements each GPU send to other GPUs
        std::vector<std::vector<size_t>> table_;
        // Temp buffers on each GPU
        //std::vector<data_t *> buf_;
        // Path to JSON file for transfer plan
        //std::string plan_file_;
        // GPU list
        // std::vector<gpu_id_t> GPU_list_;

        void set(const std::vector<data_t *>& src, 
                 const std::vector<data_t *>& dst,
                 const std::vector<std::vector<size_t>>& table
                ){
            src_ = src;
            dst_ = dst;
            table_ = table;
        }

    };

    using param_t = FasterGossipCommAll2AllParam;
    using context_t = gossip::context_t;
    using executor_t = gossip::all2all_async_t;
    using transfer_plan_t = gossip::transfer_plan_t;
    using transfer_plan_util_t = gossip::all2all;
};

} // namespace