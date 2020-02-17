/* Copyright 2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#pragma once

namespace FasterComm{

/* General Communication library interface */

class FasterComm{
public:

    virtual void execAsync() = 0;

    virtual void sync() = 0;

    virtual ~FasterComm(){}

};

}