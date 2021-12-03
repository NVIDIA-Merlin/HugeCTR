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

#include <cstring>
#include <inference/kafka_message.hpp>
#include <vector>

namespace HugeCTR {

KafkaLifetimeService::KafkaLifetimeService() {
  HCTR_LOG(DEBUG, WORLD, "Creating Kafka lifetime service.\n");
}

KafkaLifetimeService::~KafkaLifetimeService() {
  HCTR_LOG(DEBUG, WORLD, "Destroying Kafka lifetime service.\n");
  HCTR_CHECK_HINT(!rd_kafka_wait_destroyed(-1),
                  "Kafka error. Objects were not destructed properly!\n");
}

const std::shared_ptr<KafkaLifetimeService>& KafkaLifetimeService::get() {
  static std::shared_ptr<KafkaLifetimeService> lifetime_service;
  if (!lifetime_service) {
    lifetime_service.reset(new KafkaLifetimeService);
  }
  return lifetime_service;
}

#define HCTR_KAFKA_CHECK(expr)                                                      \
  do {                                                                              \
    const auto& resp = (expr);                                                      \
    HCTR_CHECK_HINT(resp == RD_KAFKA_RESP_ERR_NO_ERROR, "Kafka %s error: \"%s\"\n", \
                    rd_kafka_err2name(resp), rd_kafka_err2str(resp));               \
  } while (0)

#define HCTR_KAFKA_CONF_SET(conf, key, value)                                                    \
  do {                                                                                           \
    char error[512];                                                                             \
    const rd_kafka_conf_res_t res = rd_kafka_conf_set(conf, key, value, error, sizeof(error));   \
    HCTR_CHECK_HINT(res == RD_KAFKA_CONF_OK,                                                     \
                    "Kafka configuration \"%s\" = \"%s\". Error: \"%s\".\n", key, value, error); \
  } while (0)

template <typename TKey>
KafkaMessageSink<TKey>::KafkaMessageSink(const std::string& brokers) : TBase() {
  // Configure Kafka.
  rd_kafka_conf_t* conf = rd_kafka_conf_new();

  rd_kafka_conf_set_opaque(conf, this);
  HCTR_KAFKA_CONF_SET(conf, "bootstrap.servers", brokers.c_str());
  HCTR_KAFKA_CONF_SET(conf, "enable.idempotence", "true");
  HCTR_KAFKA_CONF_SET(conf, "queue.buffering.max.messages", "100000");
  HCTR_KAFKA_CONF_SET(conf, "queue.buffering.max.kbytes", "1048576");
  HCTR_KAFKA_CONF_SET(conf, "linger.ms", "50");
  HCTR_KAFKA_CONF_SET(conf, "batch.num.messages", "10000");
  HCTR_KAFKA_CONF_SET(conf, "batch.size", "16777216");
  rd_kafka_conf_set_dr_msg_cb(conf, on_delivered);

  // Create actual producer context.
  char error[512];
  kafka_ = rd_kafka_new(RD_KAFKA_PRODUCER, conf, error, sizeof(error));
  if (!kafka_) {
    rd_kafka_conf_destroy(conf);
    HCTR_DIE("Creating Kafka producer failed. Reason: \"%s\".\n", error);
  }
  // If we reach here, the ownership for "conf" is transferred to Kafka.

  // Startup background event processing thread.
  event_handler_ = std::thread(&KafkaMessageSink<TKey>::run, this);
}

template <typename TKey>
KafkaMessageSink<TKey>::~KafkaMessageSink() {
  // Stop background event processing.
  terminate_ = true;
  event_handler_.join();

  // If any events are left, process them now.
  HCTR_KAFKA_CHECK(rd_kafka_flush(kafka_, -1));

  // Destroy Kafka context.
  for (const auto pair : topics_) {
    rd_kafka_topic_destroy(pair.second);
  }
  topics_.clear();
  rd_kafka_destroy(kafka_);
  kafka_ = nullptr;
}

template <typename TKey>
void KafkaMessageSink<TKey>::post(const std::string& tag, const TKey& key, const char* value,
                                  const size_t value_size) {
  // Get topic, or create if it doesn't exist yet.
  rd_kafka_topic_t* topic = resolve_topic(tag);

  // Produce message.
  rd_kafka_resp_err_t error;
  do {
    if (!rd_kafka_produce(topic, RD_KAFKA_PARTITION_UA,
                          RD_KAFKA_MSG_F_BLOCK | RD_KAFKA_MSG_F_COPY | RD_KAFKA_MSG_F_PARTITION,
                          const_cast<char*>(value), value_size, &key, sizeof(TKey), this)) {
      error = RD_KAFKA_RESP_ERR_NO_ERROR;
      break;
    }

    error = rd_kafka_last_error();
    if (error != RD_KAFKA_RESP_ERR__QUEUE_FULL) {
      break;
    }

    HCTR_LOG(INFO, WORLD, "Kafka producer queue is full. Backing off...\n");
    std::this_thread::sleep_for(queue_full_backoff_delay_);
  } while (true);
  HCTR_KAFKA_CHECK(error);
  num_posted_++;
}

template <typename TKey>
rd_kafka_topic_t* KafkaMessageSink<TKey>::resolve_topic(const std::string& tag) {
  const auto topics_it = topics_.find(tag);
  if (topics_it != topics_.end()) {
    return topics_it->second;
  } else {
    // Configure topic.
    const auto conf = rd_kafka_topic_conf_new();
    rd_kafka_topic_conf_set_opaque(conf, this);

    // Create topic.
    HCTR_LOG(INFO, WORLD, "Creating new Kafka topic \"%s\".\n", tag.c_str());
    const auto topic = rd_kafka_topic_new(kafka_, tag.c_str(), conf);
    if (!topic) {
      rd_kafka_topic_conf_destroy(conf);
      HCTR_KAFKA_CHECK(rd_kafka_last_error());
    }

    // Store to speedup next usage.
    topics_.emplace(tag, topic);
    return topic;
  }
}

template <typename TKey>
void KafkaMessageSink<TKey>::on_delivered(rd_kafka_t* kafka, const rd_kafka_message_t* message,
                                          void* context) {
  const auto ctx = static_cast<KafkaMessageSink<TKey>*>(context);
  if (message->err == RD_KAFKA_RESP_ERR_NO_ERROR) {
    ctx->num_delivered_success_++;
  } else {
    ctx->num_delivered_failure_++;
  }
}

template <typename TKey>
void KafkaMessageSink<TKey>::run() {
  while (!terminate_) {
    num_events_served_ += rd_kafka_poll(kafka_, 1000);
  }
}

template class KafkaMessageSink<unsigned int>;
template class KafkaMessageSink<long long>;

template <typename TKey>
KafkaMessageSource<TKey>::KafkaMessageSource(const std::string& brokers,
                                             const std::string& consumer_group_id,
                                             const std::vector<std::string>& tag_filters,
                                             const size_t poll_timeout_ms,
                                             const size_t max_receive_buffer_size,
                                             const size_t max_batch_size,
                                             const size_t failure_backoff_ms)
    : TBase(),
      tag_filters_(tag_filters),
      poll_timeout_ms_(hctr_safe_cast<int>(poll_timeout_ms)),
      max_receive_buffer_size_(max_receive_buffer_size),
      max_batch_size_(max_batch_size),
      failure_backoff_ms_(failure_backoff_ms) {
  // Configure Kafka.
  rd_kafka_conf_t* conf = rd_kafka_conf_new();

  HCTR_KAFKA_CONF_SET(conf, "bootstrap.servers", brokers.c_str());
  HCTR_KAFKA_CONF_SET(conf, "allow.auto.create.topics", "true");
  HCTR_KAFKA_CONF_SET(conf, "group.id", consumer_group_id.c_str());
  HCTR_KAFKA_CONF_SET(conf, "enable.auto.commit", "false");
  HCTR_KAFKA_CONF_SET(conf, "enable.partition.eof", "false");
  HCTR_KAFKA_CONF_SET(conf, "topic.metadata.refresh.interval.ms", "60000");

  // Create actual consumer context.
  char error[512];
  kafka_ = rd_kafka_new(RD_KAFKA_CONSUMER, conf, error, sizeof(error));
  if (!kafka_) {
    rd_kafka_conf_destroy(conf);
    HCTR_DIE("Creating Kafka consumer \"%s\" failed. Reason: \"%s\".\n", consumer_group_id.c_str(),
             error);
  }
  // If we reach here, the ownership for "conf" is transferred to Kafka.

  // Redirect all topics to main queue.
  HCTR_KAFKA_CHECK(rd_kafka_poll_set_consumer(kafka_));

  // Make sure that there is at least one valid subscription pattern.
  HCTR_CHECK_HINT(!tag_filters_.empty(),
                  "Must provide at least subscription topic filter for Kafka.");
}

template <typename TKey>
KafkaMessageSource<TKey>::~KafkaMessageSource() {
  // Stop processing events.
  terminate_ = true;
  event_handler_.join();

  // Destroy consumer.
  HCTR_KAFKA_CHECK(rd_kafka_consumer_close(kafka_));
  rd_kafka_destroy(kafka_);
  kafka_ = nullptr;
}

template <typename TKey>
size_t KafkaMessageSource<TKey>::num_messages_processed() const {
  return num_messages_processed_;
}

template <typename TKey>
void KafkaMessageSource<TKey>::enable(std::function<HCTR_MESSAGE_SOURCE_CALLBACK> callback) {
  // Stop processing events (if already doing so).
  terminate_ = true;
  if (event_handler_.joinable()) {
    event_handler_.join();
  }

  // Start new thread with updated function pointer.
  terminate_ = false;
  event_handler_ = std::thread(&KafkaMessageSource<TKey>::run, this, std::move(callback));
}

struct KafkaTopicPartitionListDeleter {
  void operator()(rd_kafka_topic_partition_list_t* p) { rd_kafka_topic_partition_list_destroy(p); }
};

template <typename TKey>
void KafkaMessageSource<TKey>::subscribe() {
  // Get rid of previous subscription (if exits).
  HCTR_KAFKA_CHECK(rd_kafka_unsubscribe(kafka_));

  std::unique_ptr<rd_kafka_topic_partition_list_t, KafkaTopicPartitionListDeleter> part_list(
      rd_kafka_topic_partition_list_new(hctr_safe_cast<int>(tag_filters_.size())));

  // Define of topic partitions for subscription.
  HCTR_LOG(INFO, WORLD, "Attempting to (re-)subscribe to Kafka topics <\n");
  for (const std::string& tag_filter : tag_filters_) {
    HCTR_LOG(INFO, WORLD, "%s\n", tag_filter.c_str());
    rd_kafka_topic_partition_list_add(part_list.get(), ("^" + tag_filter).c_str(),
                                      RD_KAFKA_PARTITION_UA);
  }
  HCTR_LOG(INFO, WORLD, ">\n");

  // Enable subscription.
  // rd_kafka_topic_partition_list_sort(part_list.get(), nullptr, nullptr);
  HCTR_KAFKA_CHECK(rd_kafka_subscribe(kafka_, part_list.get()));
}

struct KafkaMessageDeleter {
  void operator()(rd_kafka_message_t* p) { rd_kafka_message_destroy(p); }
};

template <typename TKey>
struct KafkaReceiveBuffer {
  std::vector<TKey> keys;
  std::vector<char> values;
  size_t value_size;

  KafkaReceiveBuffer() = delete;
  KafkaReceiveBuffer(const size_t _value_size) : value_size{_value_size} {}
};

struct KafkaPartitionOffsetHash {
  template <class T, class U>
  std::size_t operator()(const std::pair<T, U>& v) const {
    return std::hash<T>()(v.first) ^ std::hash<U>()(v.second);
  }
};

template <typename TKey>
void KafkaMessageSource<TKey>::run(std::function<HCTR_MESSAGE_SOURCE_CALLBACK> callback) {
  // Attempt to subscribe to topics.
  subscribe();

  // Allocate buffer for the messages.
  std::unordered_map<std::string, KafkaReceiveBuffer<TKey>> receive_buffers;
  std::unordered_map<std::pair<std::string, int32_t>, size_t, KafkaPartitionOffsetHash> offsets;

  while (!terminate_) {
    // Append messages to the receive buffer until is is full, or we reached the end of the message
    // queue, of if there is an error.
    offsets.clear();
    receive_buffers.clear();
    for (size_t i = 0; i < max_receive_buffer_size_; i++) {
      std::unique_ptr<rd_kafka_message_t, KafkaMessageDeleter> msg(
          rd_kafka_consumer_poll(kafka_, poll_timeout_ms_));
      if (terminate_) {
        return;
      }

      // Poll timeout reached, or no data.
      if (!msg || msg->err == RD_KAFKA_RESP_ERR__PARTITION_EOF) {
        break;
      }

      // Print message if anything happened
      if (msg->err != RD_KAFKA_RESP_ERR_NO_ERROR) {
        HCTR_LOG(WARNING, WORLD, "Kafka event/error %d: \"%s\".\n", msg->err,
                 rd_kafka_message_errstr(msg.get()));

        // Break here, since message does not contain valid data.
        std::this_thread::sleep_for(failure_backoff_ms_);
        break;
      }

      const std::string topic = rd_kafka_topic_name(msg->rkt);

      if (msg->key_len != sizeof(TKey)) {
        HCTR_LOG(
            WARNING, WORLD,
            "Data corruption? Key-size of message retrieved via Kafka does not match expecation "
            "(%d != %d). Discarding!\n",
            msg->key_len, sizeof(TKey));
        break;
      }
      const TKey& key = *static_cast<TKey*>(msg->key);

      // Locate buffer and offsets.
      auto& buf = receive_buffers.try_emplace(topic, msg->len).first->second;
      const auto& off_key = std::make_pair(topic, msg->partition);
      offsets.try_emplace(off_key, msg->offset);

      // Check value size.
      if (msg->len != buf.value_size) {
        HCTR_LOG(
            WARNING, WORLD,
            "Data corruption? Value-size of message retrieved via Kafka does not match expecation "
            "(%d != %d). Discarding!\n",
            msg->len, buf.value_size);
        break;
      }

      /*
      HCTR_LOG(INFO, WORLD, "Consuming Kafka message. Topic: \"%s\", key: %d, offset: %d\n",
               topic.c_str(), key, msg->offset);
      */

      // Append message contents to the buffer.
      buf.keys.emplace_back(key);
      buf.values.insert(buf.values.end(), static_cast<char*>(msg->payload),
                        &static_cast<char*>(msg->payload)[msg->len]);
      // buffer.values.emplace_back(static_cast<char*>(msg->payload), msg->len);
      offsets[off_key] = msg->offset;

      // Reached batch size limit.
      if (buf.keys.size() >= max_batch_size_) {
        break;
      }
    }

    // Invoke callback for each topic in the buffer.
    for (const auto& receive_buffer : receive_buffers) {
      const std::string& topic = receive_buffer.first;
      const KafkaReceiveBuffer<TKey>& buf = receive_buffer.second;

      // Retry until receiver signals that the delivery was successful.
      while (
          !callback(topic, buf.keys.size(), buf.keys.data(), buf.values.data(), buf.value_size)) {
        if (terminate_) {
          return;
        }
        HCTR_LOG(WARNING, WORLD, "Unable to deliver %d key/value pairs from Kafka topic %s.\n",
                 buf.keys.size(), topic.c_str());
        std::this_thread::sleep_for(failure_backoff_ms_);
      }
      HCTR_LOG(DEBUG, WORLD, "Delivered %d key/value pairs from Kafka topic %s.\n", buf.keys.size(),
               topic.c_str());
      num_messages_processed_ += buf.keys.size();
    }

    // Commit all the offsets.
    if (!offsets.empty()) {
      std::unique_ptr<rd_kafka_topic_partition_list_t, KafkaTopicPartitionListDeleter> part_list(
          rd_kafka_topic_partition_list_new(hctr_safe_cast<int>(offsets.size())));

      for (const auto& off : offsets) {
        const std::string& topic = off.first.first;
        const int32_t partition = off.first.second;
        const int64_t offset = off.second + 1;

        auto part = rd_kafka_topic_partition_list_add(part_list.get(), topic.c_str(), partition);
        part->offset = offset;

        HCTR_LOG(DEBUG, WORLD, "Committing Kafka topic: %s, partition: %d, offset: %d\n",
                 topic.c_str(), partition, offset);
      }

      HCTR_KAFKA_CHECK(rd_kafka_commit(kafka_, part_list.get(), false));
    }
  }
}

#ifdef HCTR_KAFKA_CONF_SET
#undef HCTR_KAFKA_CONF_SET
#else
#error "HCTR_KAFKA_CONF_SET not defined?!"
#endif
#ifdef HCTR_KAFKA_CHECK
#undef HCTR_KAFKA_CHECK
#else
#error "HCTR_KAFKA_CHECK not defined?!"
#endif

template class KafkaMessageSource<unsigned int>;
template class KafkaMessageSource<long long>;

}  // namespace HugeCTR