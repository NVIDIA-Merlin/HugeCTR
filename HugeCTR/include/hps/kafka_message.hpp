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
#pragma once

#include <librdkafka/rdkafka.h>

#include <hps/message.hpp>
#include <thread>

namespace HugeCTR {

/**
 * Helper class for Kafka implementations to ensure proper cleanup.
 */
class KafkaLifetimeService final {
 private:
  DISALLOW_COPY_AND_MOVE(KafkaLifetimeService);

  KafkaLifetimeService();

 public:
  virtual ~KafkaLifetimeService();

  /**
   * Instance access to \p KafkaLifetimeService .
   *
   * @return Singleton instance of KafkaLifetimeService.
   */
  static const std::shared_ptr<KafkaLifetimeService>& get();
};

/**
 * \p MessageSink implementation for Kafka message queues.
 *
 * @tparam TKey Data-type to be used for keys in this message queue.
 */
template <typename TKey>
class KafkaMessageSink final : public MessageSink<TKey> {
 public:
  DISALLOW_COPY_AND_MOVE(KafkaMessageSink);
  using TBase = MessageSink<TKey>;

  KafkaMessageSink() = delete;

  /**
   * Construct a new \p KafkaMessageSink object.
   *
   * @param brokers The host-address of the Kafka broker to use.
   */
  KafkaMessageSink(const std::string& brokers);

  virtual ~KafkaMessageSink();

  void post(const std::string& tag, const TKey& key, const char* value, size_t value_size) override;

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
   * Invoked upon delivery of a message.
   *
   * @param kafka Pointer to Kafka handle.
   * @param message Pointer to the message.
   * @param context Pointer to context (this object).
   */
  static void on_delivered(rd_kafka_t* kafka, const rd_kafka_message_t* message, void* context);

 protected:
  std::shared_ptr<KafkaLifetimeService> lifetime_service_ = KafkaLifetimeService::get();

  rd_kafka_t* kafka_;
  std::chrono::milliseconds queue_full_backoff_delay_ = std::chrono::milliseconds(50);

  std::unordered_map<std::string, rd_kafka_topic_t*> topics_;
  size_t num_posted_ = 0;
  size_t num_delivered_success_ = 0;
  size_t num_delivered_failure_ = 0;
  size_t num_events_served_ = 0;

  // Background thread.
  bool terminate_ = false;
  std::thread event_handler_;
  void run();
};

/**
 * \p MessageSource implementation for Kafka message queues.
 *
 * @tparam TKey Data-type to be used for keys in this message queue.
 */
template <typename TKey>
class KafkaMessageSource final : public MessageSource<TKey> {
 public:
  DISALLOW_COPY_AND_MOVE(KafkaMessageSource);
  using TBase = MessageSource<TKey>;

  /**
   * Construct a new KafkaMessageSource object.
   *
   * @param brokers The host-address of the Kafka broker to use.
   * @param consumer_group_id Consumer group ID to use for this message source
   * @param tag_filters Regular expressions to limit the scope of tags that can be seen.
   */
  KafkaMessageSource(const std::string& brokers = "127.0.0.1:9092",
                     const std::string& consumer_group_id = "",
                     const std::vector<std::string>& tag_filters = {"^.*$"},
                     size_t poll_timeout_ms = 500, size_t max_receive_buffer_size = 2000,
                     size_t max_batch_size = 1000, size_t failure_backoff_ms = 50);

  virtual ~KafkaMessageSource();

  size_t num_messages_processed() const;

  void enable(std::function<HCTR_MESSAGE_SOURCE_CALLBACK> callback) override;

 protected:
  std::shared_ptr<KafkaLifetimeService> lifetime_service_ = KafkaLifetimeService::get();
  rd_kafka_t* kafka_;
  const std::vector<std::string> tag_filters_;

  // Background thread.
  const int poll_timeout_ms_;
  const size_t max_receive_buffer_size_;
  const size_t max_batch_size_;
  const std::chrono::milliseconds failure_backoff_ms_;

  bool terminate_ = false;
  std::thread event_handler_;
  size_t num_messages_processed_ = 0;
  void subscribe();
  void run(std::function<HCTR_MESSAGE_SOURCE_CALLBACK> callback);
};

}  // namespace HugeCTR