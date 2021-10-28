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

#include "resources/event_manager.h"
#include "common.h"
#include <chrono>

namespace SparseOperationKit {

EventManager::EventManager()
: should_stop_(false), mu_(), 
porter_thread_(&EventManager::porter_function, this) {
}

EventManager::~EventManager() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        should_stop_ = true;
    }
    porter_thread_.join();
}

std::unique_ptr<EventManager> EventManager::create() {
    return std::unique_ptr<EventManager>(new EventManager());
}

std::shared_ptr<Event> EventManager::get_event() {
    std::unique_lock<std::mutex> lock(mu_);
    std::shared_ptr<Event> event{nullptr};
    if (unused_events_.empty()) {
        event = Event::create();
    } else {
        event = unused_events_.front();
        unused_events_.pop();
    }
    inuse_events_.push(event);
    lock.unlock();
    return event;
}

void EventManager::sync_two_streams(cudaStream_t& root_stream, 
                                    cudaStream_t& sub_stream) {
    /*--root_stream->event->sub_stream--*/
    std::shared_ptr<Event> event = get_event();
    event->Record(root_stream);
    event->TillReady(sub_stream);
}

void EventManager::porter_function() {
    while (true) {
        try {
            std::unique_lock<std::mutex> lock(mu_);
            if (should_stop_) break;
            if (!inuse_events_.empty()) {
                auto& event = inuse_events_.front();
                if (!event->IsInUse()) {
                    inuse_events_.pop();
                    lock.unlock();
                    event->Reset();
                    lock.lock();
                    unused_events_.push(event);
                }
            }
            lock.unlock();
            // FIXME: if heavy cpu contention, let this thread sleep longer.
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } catch (const std::exception& error) {
            throw std::runtime_error(ErrorBase + 
                " EventManager stopped due to: " + error.what());
        } // try-catch
    } // while-block
}

} // namespace SparseOperationKit