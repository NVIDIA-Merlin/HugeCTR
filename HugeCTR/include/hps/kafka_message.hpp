/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#pragma once

#include <rdkafka.h>

#include <condition_variable>
#include <hps/message.hpp>
#include <thread>

namespace HugeCTR {

struct KafkaMessageSinkParams : public MessageSinkParams {
  std::string brokers = "localhost";  // The host-address of the Kafka broker to use.
  size_t num_partitions =
      8;  // Groups data into N partitions; this is equivalent to the number of key groups to use.
          // We use consistent partitioning to distributed updates to these.
  size_t send_buffer_size =
      256 * 1024;  // The maximum message size to send. Hence, this value should must be in [16 +
                   // value_size, message.max.bytes of the broker - 1024].
  size_t num_send_buffers = 1024;  // Maximum number of send buffers.
  bool await_connection =
      false;  // Awaits a handshake with the broker by attempting to queue an empty message.
};

/**
 * \p MessageSink implementation for Kafka message queues.
 *
 * @tparam Key Data-type to be used for keys in this message queue.
 */
template <typename Key>
class KafkaMessageSink final : public MessageSink<Key, KafkaMessageSinkParams> {
 public:
  using Base = MessageSink<Key, KafkaMessageSinkParams>;

  HCTR_DISALLOW_COPY_AND_MOVE(KafkaMessageSink);

  KafkaMessageSink() = delete;

  /**
   * Construct a new \p KafkaMessageSink object.
   */
  KafkaMessageSink(const KafkaMessageSinkParams& params);

  virtual ~KafkaMessageSink();

  virtual void post(const std::string& tag, size_t num_pairs, const Key* keys, const char* values,
                    uint32_t value_size) override;

  virtual void flush() override;

 protected:
  /**
   * Internally called to find/create Kafka topics.
   *
   * @param tag The name of the the topic corresponding to the supplied tag.
   *
   * @return Pointer to the Kafka topic.
   */
  rd_kafka_topic_t* resolve_topic(const std::string& tag);

  /**
   * Internally called to get a send buffer.
   *
   * @param value_size Size of each value. This will be used to fill the header and check for some
   * basic errors.
   * @return Pointer to send buffer.
   */
  char* acquire_send_buffer(uint32_t value_size);

  /**
   * Internally called to
   *
   * @param topic Kafka topic to which to produce.
   * @param send_buffer Pointer to send buffer.
   * @param payload_length Valid part of the payload.
   * @param key_group Used by the partitioner to
   */
  void blocking_produce(rd_kafka_topic_t* topic, char* send_buffer, size_t payload_length,
                        size_t key_group);

 protected:
  rd_kafka_t* rk_;

  std::chrono::milliseconds queue_full_backoff_delay_ = std::chrono::milliseconds(50);

  std::unordered_map<std::string, rd_kafka_topic_t*> topics_;

  // Preallocated buffers to speed up sending.
  std::vector<char> send_buffer_memory_;
  std::vector<char*> send_buffers_;
  mutable std::mutex send_buffer_barrier_;
  mutable std::condition_variable send_buffer_semaphore_;

 private:
  // Background thread.
  bool terminate_ = false;
  std::thread event_handler_;
  void run();

  size_t num_events_served_ = 0;

  /**
   * Invoked upon raising error.
   *
   * @param err_val Error code.
   * @param reason Error description.
   */
  void on_error(rd_kafka_resp_err_t err, const char* reason);

  /**
   * Invoked upon delivery of a message.
   *
   * @param msg Delivered message.
   */
  void on_delivered(const rd_kafka_message_t& msg);

  size_t num_delivered_success_ = 0;
  size_t num_delivered_failure_ = 0;
};

/**
 * \p MessageSource implementation for Kafka message queues.
 *
 * @tparam Key Data-type to be used for keys in this message queue.
 */
template <typename Key>
class KafkaMessageSource final : public MessageSource<Key> {
 public:
  using Base = MessageSource<Key>;

  HCTR_DISALLOW_COPY_AND_MOVE(KafkaMessageSource);

  /**
   * Construct a new KafkaMessageSource object.
   *
   * @param brokers The host-address of the Kafka broker to use.
   * @param consumer_group_id Consumer group ID to use for this message source
   * @param tag_filters Regular expressions to limit the scope of tags that can be seen.
   * @param metadata_refresh_interval_ms Refresh metadata information from server very x ms.
   * @param receive_buffer_size Size of a receive buffer. This should be identical to the broker's
   * \p send_buffer_size .
   * @param poll_timeout_ms Timeout for downloading messsages in milliseconds.
   * @param max_batch_size Maximum number of key/values that can accumulate before invoking
   * callback.
   * @param failure_backoff_ms In case something bad happend, wait this number of milliseconds.
   * @param max_commit_interval Regardless of the amount of values that are available, after this
   * many messages have been decoded, invoke the callback and commit.
   */
  KafkaMessageSource(const std::string& brokers = "127.0.0.1:9092",
                     const std::string& consumer_group_id = "",
                     const std::vector<std::string>& tag_filters = {"^hps_.+$"},
                     size_t metadata_refresh_interval_ms = 30'000,
                     size_t receive_buffer_size = 256 * 1024, size_t poll_timeout_ms = 500,
                     size_t max_batch_size = 8 * 1024, size_t failure_backoff_ms = 50,
                     size_t max_commit_interval = 32);

  virtual ~KafkaMessageSource();

  size_t num_keys_delivered() const { return num_keys_delivered_; }
  size_t num_keys_committed() const { return num_keys_committed_; }
  size_t num_messages_committed() const { return num_messages_committed_; }

  virtual void engage(std::function<HCTR_MESSAGE_SOURCE_CALLBACK> callback) override;

 protected:
  rd_kafka_t* rk_;

  const std::vector<std::string> tag_filters_;

  // Background thread.
  const size_t poll_timeout_ms_;
  const size_t max_batch_size_;
  const std::chrono::milliseconds failure_backoff_ms_;
  const size_t max_commit_interval_;

 private:
  bool terminate_ = false;
  std::thread event_handler_;
  size_t num_keys_delivered_ = 0;
  size_t num_keys_committed_ = 0;
  size_t num_messages_committed_ = 0;

  void resubscribe();
  void run(std::function<HCTR_MESSAGE_SOURCE_CALLBACK> callback);

  /**
   * Invoked upon raising error.
   *
   * @param err_val Error code.
   * @param reason Error description.
   */
  void on_error(rd_kafka_resp_err_t err, const char* reason);
};

}  // namespace HugeCTR
