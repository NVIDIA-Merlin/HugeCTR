/*
* Copyright (c) 2020, NVIDIA CORPORATION.
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

#ifndef THREADING_UTILS_H
#define THREADING_UTILS_H

#include <mutex>
#include <condition_variable>
#include <memory>
#include <vector>
#include <functional>
#include <chrono>

namespace HugeCTR {
namespace Version2 {

/**
Used to synchronize multi threads to the same point.
*/
class Barrier {
public:
    Barrier(const Barrier&) = delete;
    Barrier(Barrier&&) = delete;
    Barrier& operator=(const Barrier&) = delete;
    Barrier& operator=(Barrier&&) = delete;
    ~Barrier() = default;

    explicit Barrier(unsigned int thread_count)
    : mMutex_(), mCond_(), mThreshould_(thread_count), mCount_(thread_count), mGeneration_(0)
    {}

    void wait() {
        std::unique_lock<std::mutex> lock(mMutex_);
        auto lGen = mGeneration_;
        if (!--mCount_) {
            mGeneration_++;
            mCount_ = mThreshould_;
            mCond_.notify_all();
        } else {
            // mCond_.wait(lock, [this, lGen](){ return lGen != mGeneration_; });
            mCond_.wait_for(lock, mTimeThreshold_, [this, lGen](){ return lGen != mGeneration_; });
        }
    }

private:
    std::mutex mMutex_;
    std::condition_variable mCond_;
    const unsigned int mThreshould_;
    volatile unsigned int mCount_;
    volatile unsigned int mGeneration_;
    const std::chrono::seconds mTimeThreshold_{10};
};

/**
Reuseable once flag in different iteration.
#TODO: Important: In different iteration, the steps for Reuseable once flag should be
[1]reuseable_once_flag.reset() -> [2]barrrier.wait() -> [3]reuseable_once_flag.get_once_flag()
[1] is to make sure at least one once_flag is available
[2] is the synchronize all threads participated in the same function. 
Otherwise, one threads may be doing [3], while others threads are doing [1] to reset again, which 
results in "multiple execution rather than once" in one single iteration.  
[3] is used to get on valid once_flag.

*/
class ReuseableOnceFlag {
public:
    ReuseableOnceFlag(const ReuseableOnceFlag&) = delete;
    ReuseableOnceFlag(ReuseableOnceFlag&&) = delete;
    ReuseableOnceFlag& operator=(const ReuseableOnceFlag&) = delete;
    ReuseableOnceFlag& operator=(ReuseableOnceFlag&&) = delete;

    ReuseableOnceFlag()
    :mMutex_(), once_flags_(2), current_(0) 
    {
        once_flags_[0].reset(new std::once_flag());
        once_flags_[1].reset(new std::once_flag());
    }
    ~ReuseableOnceFlag() = default;

    std::shared_ptr<std::once_flag> get_once_flag() {
        return once_flags_[current_];
    }

    void reset() {
        std::unique_lock<std::mutex> lock(mMutex_, std::try_to_lock);
        if (lock){
            current_ ^= 1;
            once_flags_[current_].reset(new std::once_flag());
        }
    }

private:
    std::mutex mMutex_;
    std::vector<std::shared_ptr<std::once_flag>> once_flags_;
    unsigned int current_; // 
};


/*blocking all threads and reuseable call_once*/
class BlockingCallOnce {
public:
    BlockingCallOnce() = delete;
    BlockingCallOnce(const BlockingCallOnce&) = delete;
    BlockingCallOnce(BlockingCallOnce&&) = delete;
    BlockingCallOnce& operator=(const BlockingCallOnce&) = delete;
    BlockingCallOnce& operator=(BlockingCallOnce&&) = delete;
    ~BlockingCallOnce() = default;

    explicit BlockingCallOnce(unsigned int threads_count)
    : mu_(), cond_(), threads_count_(threads_count), count_(threads_count), generation_(0) 
    {}

    template <typename Callable, typename... Args>
    void operator()(Callable&& func, Args&&... args) {
        std::unique_lock<std::mutex> lock(mu_);
        auto local_gen = generation_;
        if (!--count_) {
            generation_++;
            count_ = threads_count_;

            /*call once in this generation*/
            auto bound_functor = std::bind(func, args...);
            once_callable_ = &bound_functor;
            once_call_ = &BlockingCallOnce::once_call_impl_<decltype(bound_functor)>;
            
            try {
                (this->*once_call_)();
                cond_.notify_all();
            } catch (...) {
                excp_ptr = std::current_exception();
                cond_.notify_all(); // TODO: Need anthoer mutex??
                std::rethrow_exception(excp_ptr);
            }
        } else {
            cond_.wait_for(lock, time_threshold_, [this, local_gen](){ return local_gen != generation_; });
            if (excp_ptr) { std::rethrow_exception(excp_ptr); }
            if (local_gen == generation_) { throw std::runtime_error("Blocking threads time out."); }
        }
    }

private:
    std::mutex mu_;
    std::condition_variable cond_;
    const unsigned int threads_count_;
    volatile unsigned int count_;
    volatile unsigned int generation_;
    std::exception_ptr excp_ptr = nullptr;
    const std::chrono::seconds time_threshold_{10};

private:
    void* once_callable_;
    void (BlockingCallOnce::*once_call_)();
    template <typename Callable>
    void once_call_impl_(){
        (*(Callable*)once_callable_)();
    }
};



} // namespace Version2
} // namespace HugeCTR

#endif // THREADING_UTILS_H