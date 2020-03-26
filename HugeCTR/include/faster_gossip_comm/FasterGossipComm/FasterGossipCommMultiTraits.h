/* Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */


#pragma once

#include "FasterGossipComm.h"

namespace FasterGossipCommMulti{

// Traits for All2All multi-node communication using GOSSIP library
template<typename data_t_>
class FasterGossipCommMultiAll2AllTraits{
public:
    using FasterGossipCommTrait = FasterGossipComm::FasterGossipCommAll2AllTraits<data_t_>;
    using FasterGossipComm = FasterGossipComm::FasterGossipComm<data_t_, FasterGossipCommTrait>;
    using gpu_id_t = gossip::gpu_id_t;
    using transfer_plan_t = gossip::transfer_plan_t;

}; // class

}// namespace
