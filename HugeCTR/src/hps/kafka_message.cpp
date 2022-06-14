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

#include <parallel_hashmap/phmap.h>

#include <cstring>
#include <hps/kafka_message.hpp>
#include <vector>

#define HCTR_USE_XXHASH
#ifdef HCTR_USE_XXHASH
#include <xxh3.h>
#define HCTR_KEY_GROUP_OF_KEY(KEY) (XXH3_64bits((KEY), sizeof(TKey)) % num_key_groups_)
#else
#define HCTR_KEY_GROUP_OF_KEY(KEY) (static_cast<size_t>(*KEY) % num_key_groups_)
#endif

// TODO: Remove me!
#pragma GCC diagnostic error "-Wconversion"

namespace HugeCTR {

#ifdef HCTR_KAFKA_CHECK
#error HCTR_KAFKA_CHECK already defined!
#else
#define HCTR_KAFKA_CHECK(EXPR)                                                    \
  do {                                                                            \
    const auto& resp = (EXPR);                                                    \
    HCTR_CHECK_HINT(resp == RD_KAFKA_RESP_ERR_NO_ERROR, "Kafka %s error: '%s'\n", \
                    rd_kafka_err2name(resp), rd_kafka_err2str(resp));             \
  } while (0)
#endif

/**
 * Helper class for Kafka implementations to ensure proper cleanup.
 */
class KafkaLifetimeService final {
 private:
  KafkaLifetimeService() { HCTR_LOG(DEBUG, WORLD, "Creating Kafka lifetime service.\n"); }
  DISALLOW_COPY_AND_MOVE(KafkaLifetimeService);

 public:
  virtual ~KafkaLifetimeService() {
    HCTR_LOG(DEBUG, WORLD, "Destroying Kafka lifetime service.\n");
    HCTR_CHECK_HINT(!rd_kafka_wait_destroyed(-1),
                    "Kafka error. Objects were not destructed properly!\n");
  }

  /**
   * Called in the constructor of each Kafka object.
   */
  static void init() {
    static std::unique_ptr<KafkaLifetimeService> instance;
    static std::once_flag once_flag;
    std::call_once(once_flag, []() { instance.reset(new KafkaLifetimeService()); });
  }
};

const size_t HCTR_KAFKA_ERROR_STRING_LENGTH = 512;

const uint32_t HCTR_KAFKA_VALUE_PREFIX =
    (uint32_t)('H') | ((uint32_t)('C') << 8) | ((uint32_t)('T') << 16) | ((uint32_t)('R') << 24);

void kafka_conf_set_and_check(rd_kafka_conf_t* const conf, const char* const key,
                              const char* const value) {
  // HCTR_LOG_S(DEBUG, WORLD) << key << " = " << value << std::endl;
  char error[HCTR_KAFKA_ERROR_STRING_LENGTH];
  const rd_kafka_conf_res_t res = rd_kafka_conf_set(conf, key, value, error, sizeof(error));
  HCTR_CHECK_HINT(res == RD_KAFKA_CONF_OK, "Kafka configuration '%s' = '%s'. Error: %s.\n", key,
                  value, error);
}

void kafka_conf_set_and_check(rd_kafka_conf_t* const conf, const char* const key,
                              const std::string& value) {
  kafka_conf_set_and_check(conf, key, value.c_str());
}
void kafka_conf_set_and_check(rd_kafka_conf_t* const conf, const char* const key,
                              const bool value) {
  kafka_conf_set_and_check(conf, key, value ? "true" : "false");
}

void kafka_conf_set_and_check(rd_kafka_conf_t* const conf, const char* const key, const int value) {
  kafka_conf_set_and_check(conf, key, std::to_string(value));
}

void kafka_conf_set_and_check(rd_kafka_conf_t* const conf, const char* const key,
                              const size_t value) {
  kafka_conf_set_and_check(conf, key, std::to_string(value));
}

int32_t kafka_key_group_partitioner(const rd_kafka_topic_t* const topic, const void* const key,
                                    const size_t key_length, const int32_t partition_count,
                                    void* const topic_opaque, void* const msg_opaque) {
  HCTR_CHECK(key_length == 0);
  // Convert key group into kafka partition.
  const size_t key_group = reinterpret_cast<size_t>(msg_opaque);
  return static_cast<int32_t>(key_group & 0x7fffffff) % partition_count;
}

struct KafkaTopicPartitionListDeleter {
  void operator()(rd_kafka_topic_partition_list_t* p) { rd_kafka_topic_partition_list_destroy(p); }
};

struct KafkaMessageDeleter {
  void operator()(rd_kafka_message_t* p) { rd_kafka_message_destroy(p); }
};

template <typename TKey>
KafkaMessageSink<TKey>::KafkaMessageSink(const std::string& brokers, const size_t num_key_groups,
                                         const size_t send_buffer_size,
                                         const size_t num_send_buffers, const bool await_connection)
    : TBase(),
      num_key_groups_(num_key_groups),
      send_buffer_size_(send_buffer_size),
      num_send_buffers_(num_send_buffers),
      send_buffer_memory_(num_send_buffers * send_buffer_size) {
  HCTR_CHECK(send_buffer_size >= 1024 && num_send_buffers > 0);

  // Create send buffers.
  HCTR_LOG_S(DEBUG, WORLD) << "Allocating Kafka send buffer (" << send_buffer_memory_.size()
                           << " bytes)..." << std::endl;
  send_buffers_.reserve(num_send_buffers_);
  for (auto it = send_buffer_memory_.begin(); it != send_buffer_memory_.end();
       it += send_buffer_size) {
    char* send_buffer = &(*it);
    *reinterpret_cast<uint32_t*>(send_buffer) = HCTR_KAFKA_VALUE_PREFIX;
    send_buffers_.push_back(send_buffer);
  }
  HCTR_CHECK(send_buffers_.size() == num_send_buffers);

  // Configure Kafka.
  rd_kafka_conf_t* conf = rd_kafka_conf_new();

  // Global parameters.
  kafka_conf_set_and_check(conf, "metadata.broker.list", brokers);
  kafka_conf_set_and_check(conf, "message.max.bytes",
                           send_buffer_size + 1024);  // Default: 1'000'000
  kafka_conf_set_and_check(conf, "receive.message.max.bytes",
                           128 * 1024 * 1024);  // Default: 100'000'000
  kafka_conf_set_and_check(conf, "topic.metadata.refresh.interval.ms", 60'000);  // Default: 300'000
  kafka_conf_set_and_check(conf, "topic.metadata.refresh.sparse", true);         // Default: true
  rd_kafka_conf_set_events(conf, RD_KAFKA_EVENT_NONE);
  rd_kafka_conf_set_error_cb(
      conf, [](rd_kafka_t* const rk, int err, const char* const reason, void* const opaque) {
        const auto ctx = static_cast<KafkaMessageSink<TKey>*>(opaque);
        ctx->on_error(static_cast<rd_kafka_resp_err_t>(err), reason);
      });
  kafka_conf_set_and_check(conf, "enable.random.seed", true);  // Default: true
  rd_kafka_conf_set_opaque(conf, this);

  // Producer specific parameters.
  kafka_conf_set_and_check(conf, "enable.idempotence", true);  // Default: false, DO NOT CHANGE!
  kafka_conf_set_and_check(conf, "queue.buffering.max.messages", 64 * 1024);  // Default: 100'000
  kafka_conf_set_and_check(conf, "queue.buffering.max.kbytes",
                           1 * 1024 * 1024);                       // Default: 1'048'576
  kafka_conf_set_and_check(conf, "queue.buffering.max.ms", 100);   // Default: 5
  kafka_conf_set_and_check(conf, "compression.codec", "none");     // Default: none
  kafka_conf_set_and_check(conf, "batch.num.messages", 8 * 1024);  // Default: 10'000
  kafka_conf_set_and_check(conf, "batch.size", 64 * 1024 * 1024);  // Default: 1'000'000
  rd_kafka_conf_set_dr_msg_cb(
      conf, [](rd_kafka_t* const rk, const rd_kafka_message_t* const msg, void* opaque) {
        const auto ctx = static_cast<KafkaMessageSink<TKey>*>(opaque);
        ctx->on_delivered(*msg);
      });

  // Producer default topic parameters.
  kafka_conf_set_and_check(conf, "partitioner",
                           "consistent");  // Default: consistent_random, DO NOT CHANGE!

  // Create actual producer context.
  char error[HCTR_KAFKA_ERROR_STRING_LENGTH];
  rk_ = rd_kafka_new(RD_KAFKA_PRODUCER, conf, error, sizeof(error));
  if (!rk_) {
    rd_kafka_conf_destroy(conf);
    HCTR_DIE("Creating Kafka producer failed. Reason: '%s'.\n", error);
  }
  // If we reach here, the ownership for "conf" is transferred to Kafka.
  KafkaLifetimeService::init();
  HCTR_LOG_S(DEBUG, WORLD) << "Kafka sink initialization complete!" << std::endl;

  // Startup background event processing thread.
  event_handler_ = std::thread(&KafkaMessageSink<TKey>::run, this);

  // Send a beacon.
  if (await_connection) {
    HCTR_LOG_S(DEBUG, WORLD) << "Sending a beacon to the Kafka broker..." << std::endl;
    post("__hps_beacon", 0, nullptr, nullptr, sizeof(size_t));
    flush();
    HCTR_LOG_S(DEBUG, WORLD) << "Beacon was received. Kafka broker connected." << std::endl;
  }
}

template <typename TKey>
KafkaMessageSink<TKey>::~KafkaMessageSink() {
  // Stop background event processing.
  terminate_ = true;
  event_handler_.join();

  // If any events are left, process them now.
  HCTR_KAFKA_CHECK(rd_kafka_flush(rk_, -1));

  // Destroy Kafka context.
  for (const auto pair : topics_) {
    rd_kafka_topic_destroy(pair.second);
  }
  topics_.clear();
  rd_kafka_destroy(rk_);
  rk_ = nullptr;
}

template <typename TKey>
void KafkaMessageSink<TKey>::post(const std::string& tag, size_t num_pairs, const TKey* const keys,
                                  const char* values, const uint32_t value_size) {
  // Make sure there enough space to store at least one key-value pair.
  const size_t key_value_size = sizeof(TKey) + value_size;
  HCTR_CHECK(sizeof(uint32_t) * 2 + key_value_size <= send_buffer_size_);

  // Get topic, or create if it doesn't exist yet.
  rd_kafka_topic_t* const topic = resolve_topic(tag);

  if (num_pairs == 0) {
    // Request send buffer to hold the payload.
    char* const payload = acquire_send_buffer(value_size);
    const size_t p_length = sizeof(uint32_t) * 2;

    // Add nothing. This is just a beacon.

    // Produce Kafka message.
    blocking_produce(topic, payload, p_length, 0);
  } else if (num_pairs == 1) {
    // Determine the key group.
    const size_t key_group = HCTR_KEY_GROUP_OF_KEY(keys);

    // Request send buffer to hold the payload.
    char* const payload = acquire_send_buffer(value_size);
    size_t p_length = sizeof(uint32_t) * 2;

    // Append key & value.
    *reinterpret_cast<TKey*>(&payload[p_length]) = *keys;
    p_length += sizeof(TKey);
    memcpy(&payload[p_length], values, value_size);
    p_length += value_size;

    // Produce Kafka message.
    blocking_produce(topic, payload, p_length, key_group);
  } else {
    const TKey* const keys_end = &keys[num_pairs];
    for (size_t key_group = 0; key_group < num_key_groups_; key_group++) {
      char* payload = nullptr;
      size_t p_length = 0;

      for (const TKey* k = keys; k != keys_end; k++) {
        // Only consider keys that belong to current group.
        if (HCTR_KEY_GROUP_OF_KEY(&k) != key_group) {
          continue;
        }

        // If send buffer buffer already available.
        if (payload) {
          // Not enough space to hold another key-value pair.
          if (p_length + key_value_size > send_buffer_size_) {
            // Send current buffer.
            blocking_produce(topic, payload, p_length, key_group);

            // Get new send buffer.
            payload = acquire_send_buffer(value_size);
            p_length = sizeof(uint32_t) * 2;
          }
        } else {
          // Request send buffer to hold the payload.
          payload = acquire_send_buffer(value_size);
          p_length = sizeof(uint32_t) * 2;
        }

        // Append key & value.
        *reinterpret_cast<TKey*>(&payload[p_length]) = *k;
        p_length += sizeof(TKey);
        memcpy(&payload[p_length], &values[(k - keys) * value_size], value_size);
        p_length += value_size;
      }

      // Sent any unsent payload.
      if (payload) {
        blocking_produce(topic, payload, p_length, key_group);
      }
    }
  }


  // Update metrics.
  TBase::post(tag, num_pairs, keys, values, value_size);
}

template <typename TKey>
void KafkaMessageSink<TKey>::flush() {
  while (true) {
    HCTR_LOG(DEBUG, WORLD, "Awaiting delivery of pending Kafka messages...\n");

    const rd_kafka_resp_err_t err = rd_kafka_flush(rk_, 5000);
    if (err != RD_KAFKA_RESP_ERR__TIMED_OUT) {
      HCTR_KAFKA_CHECK(err);
      break;
    }
  }

  // Update metrics.
  TBase::flush();
}

template <typename TKey>
rd_kafka_topic_t* KafkaMessageSink<TKey>::resolve_topic(const std::string& tag) {
  const auto topics_it = topics_.find(tag);
  if (topics_it != topics_.end()) {
    return topics_it->second;
  }


  // Configure topic.
  rd_kafka_topic_conf_t* const conf = rd_kafka_topic_conf_new();

  rd_kafka_topic_conf_set_partitioner_cb(conf, kafka_key_group_partitioner);
  rd_kafka_topic_conf_set_opaque(conf, this);

  // Create topic.
  HCTR_LOG_S(INFO, WORLD) << "Creating new Kafka topic '" << tag << "'." << std::endl;
  rd_kafka_topic_t* const topic = rd_kafka_topic_new(rk_, tag.c_str(), conf);
  if (!topic) {
    rd_kafka_topic_conf_destroy(conf);
    HCTR_KAFKA_CHECK(rd_kafka_last_error());
  }
  // If we reach here, the ownership for "conf" is transferred to the Kafka topic.

  // Store to speedup next usage.
  topics_.emplace(tag, topic);
  return topic;
}

template <typename TKey>
char* KafkaMessageSink<TKey>::acquire_send_buffer(uint32_t value_size) {
  // Wait until buffer becomes available.
  char* send_buffer;
  {
    std::unique_lock<std::mutex> lock(send_buffer_barrier_);
    if (send_buffers_.empty()) {
      // HCTR_LOG_S(DEBUG, WORLD) << "Awaiting buffer availability..." << std::endl;
      send_buffer_semaphore_.wait(lock);
    }
    send_buffer = send_buffers_.back();
    send_buffers_.pop_back();
    // HCTR_LOG_S(DEBUG, WORLD) << "Borrowed buffer " << send_buffers_.size() << '.' << std::endl;
  }

  // Note the value size.
  *reinterpret_cast<uint32_t*>(&send_buffer[sizeof(uint32_t)]) = value_size;

  return send_buffer;
}

template <typename TKey>
void KafkaMessageSink<TKey>::blocking_produce(rd_kafka_topic_t* const topic, char* const payload,
                                              const size_t payload_length, const size_t key_group) {
  while (rd_kafka_produce(topic, RD_KAFKA_PARTITION_UA,
                          RD_KAFKA_MSG_F_BLOCK | RD_KAFKA_MSG_F_PARTITION, payload, payload_length,
                          nullptr, 0, reinterpret_cast<void*>(key_group))) {
    const rd_kafka_resp_err_t err = rd_kafka_last_error();
    if (err != RD_KAFKA_RESP_ERR__QUEUE_FULL) {
      HCTR_KAFKA_CHECK(err);
      break;
    }

    HCTR_LOG(DEBUG, WORLD, "Kafka producer queue is full. Backing off...\n");
    std::this_thread::sleep_for(queue_full_backoff_delay_);
  }
}

template <typename TKey>
void KafkaMessageSink<TKey>::run() {
  hctr_set_thread_name("kafka sink");

  // Keep polling until exit.
  while (!terminate_) {
    num_events_served_ += rd_kafka_poll(rk_, 1000);
  }
}

template <typename TKey>
void KafkaMessageSink<TKey>::on_error(rd_kafka_resp_err_t err, const char* const reason) {
  HCTR_LOG_S(ERROR, WORLD) << "Kafka error " << rd_kafka_err2name(err) << ". Reason: " << reason
                           << std::endl;

  // If it is not fatal, just move on.
  if (err != RD_KAFKA_RESP_ERR__FATAL) {
    return;
  }

  // Write origin error to log.
  char error[HCTR_KAFKA_ERROR_STRING_LENGTH];
  err = rd_kafka_fatal_error(rk_, error, sizeof(error));
  HCTR_DIE("Fatal Kafka error encountered. Error: '%s', Reason: '%s'.\n", rd_kafka_err2name(err),
           error);
}

template <typename TKey>
void KafkaMessageSink<TKey>::on_delivered(const rd_kafka_message_t& msg) {
  if (msg.err) {
    HCTR_LOG_S(ERROR, WORLD) << "Kafka message delivery failed; Topic: "
                             << rd_kafka_topic_name(msg.rkt) << ", payload: " << msg.len
                             << " bytes; Error " << rd_kafka_err2name(msg.err) << ": "
                             << rd_kafka_err2str(msg.err) << std::endl;
    num_delivered_failure_++;
  } else {
    num_delivered_success_++;

    // Return send buffer back to the pool.
    {
      std::unique_lock<std::mutex> lock(send_buffer_barrier_);
      // HCTR_LOG_S(DEBUG, WORLD) << "Sent " << send_buffers_.size() << std::endl;
      send_buffers_.push_back(reinterpret_cast<char*>(msg.payload));
    }
    send_buffer_semaphore_.notify_one();
  }
}

template class KafkaMessageSink<unsigned int>;
template class KafkaMessageSink<long long>;

template <typename TKey>
KafkaMessageSource<TKey>::KafkaMessageSource(
    const std::string& brokers, const std::string& consumer_group_id,
    const std::vector<std::string>& tag_filters, const size_t metadata_refresh_interval_ms,
    const size_t receive_buffer_size, const size_t poll_timeout_ms, const size_t max_batch_size,
    const size_t failure_backoff_ms, const size_t max_commit_interval)
    : TBase(),
      tag_filters_(tag_filters),
      poll_timeout_ms_(poll_timeout_ms),
      max_batch_size_(max_batch_size),
      failure_backoff_ms_(failure_backoff_ms),
      max_commit_interval_(max_commit_interval) {
  // Make sure that there is at least one valid subscription pattern.
  HCTR_CHECK_HINT(!tag_filters_.empty(),
                  "Must provide at least subscription topic filter for Kafka.");

  // Make sure numeric arguments have sane values.
  HCTR_CHECK(metadata_refresh_interval_ms >= 10);
  HCTR_CHECK(receive_buffer_size >= 1024);
  HCTR_CHECK(poll_timeout_ms > 0 && poll_timeout_ms < std::numeric_limits<int>::max());
  HCTR_CHECK(max_batch_size > 0);
  HCTR_CHECK(max_commit_interval > 0);

  // Configure Kafka.
  rd_kafka_conf_t* conf = rd_kafka_conf_new();

  // Global parameters.
  kafka_conf_set_and_check(conf, "metadata.broker.list", brokers);
  kafka_conf_set_and_check(conf, "message.max.bytes",
                           receive_buffer_size + 1024);  // Default: 1'000'000
  kafka_conf_set_and_check(conf, "receive.message.max.bytes",
                           128 * 1024 * 1024);  // Default: 100'000'000
  kafka_conf_set_and_check(conf, "topic.metadata.refresh.interval.ms",
                           metadata_refresh_interval_ms);                 // Default: 300'000
  kafka_conf_set_and_check(conf, "topic.metadata.refresh.sparse", true);  // Default: true
  rd_kafka_conf_set_events(conf, RD_KAFKA_EVENT_NONE);
  rd_kafka_conf_set_error_cb(
      conf, [](rd_kafka_t* const rk, int err, const char* const reason, void* const opaque) {
        const auto ctx = static_cast<KafkaMessageSource<TKey>*>(opaque);
        ctx->on_error(static_cast<rd_kafka_resp_err_t>(err), reason);
      });
  kafka_conf_set_and_check(conf, "enable.random.seed", true);  // Default: true
  rd_kafka_conf_set_opaque(conf, this);

  // Consumer specific parameters.
  kafka_conf_set_and_check(conf, "group.id", consumer_group_id);
  kafka_conf_set_and_check(conf, "session.timeout.ms", 30'000);    // Default: 45'000
  kafka_conf_set_and_check(conf, "heartbeat.interval.ms", 3'000);  // Default: 3'000
  kafka_conf_set_and_check(conf, "enable.auto.commit", false);     // Default: true, DO NOT CHANGE!
  kafka_conf_set_and_check(conf, "enable.auto.offset.store", true);  // Default: true
  kafka_conf_set_and_check(conf, "queued.min.messages", 64 * 1024);  // Default: 100'000
  kafka_conf_set_and_check(conf, "queued.max.messages.kbytes", 1 * 1024 * 1024);  // Default: 65'536
  kafka_conf_set_and_check(conf, "enable.partition.eof", true);                   // Default: false
  kafka_conf_set_and_check(conf, "check.crcs", false);                            // Default: false
  kafka_conf_set_and_check(conf, "allow.auto.create.topics", true);               // Default: false

  // Consumer default topic parameters.
  kafka_conf_set_and_check(conf, "auto.offset.reset", "smallest");  // Default: largest

  // Create actual consumer context.
  char error[HCTR_KAFKA_ERROR_STRING_LENGTH];
  rk_ = rd_kafka_new(RD_KAFKA_CONSUMER, conf, error, sizeof(error));
  if (!rk_) {
    rd_kafka_conf_destroy(conf);
    HCTR_DIE("Creating Kafka consumer '%s' failed. Reason: %s\n", consumer_group_id.c_str(), error);
  }
  // If we reach here, the ownership for "conf" is transferred to Kafka.
  KafkaLifetimeService::init();

  // Redirect all topics to main queue.
  HCTR_KAFKA_CHECK(rd_kafka_poll_set_consumer(rk_));
}

template <typename TKey>
KafkaMessageSource<TKey>::~KafkaMessageSource() {
  // Stop processing events.
  terminate_ = true;
  if (event_handler_.joinable()) {
    event_handler_.join();
  }

  // Destroy consumer.
  HCTR_KAFKA_CHECK(rd_kafka_consumer_close(rk_));

  rd_kafka_destroy(rk_);
  rk_ = nullptr;
}

template <typename TKey>
void KafkaMessageSource<TKey>::engage(std::function<HCTR_MESSAGE_SOURCE_CALLBACK> callback) {
  // Stop processing events (if already doing so).
  terminate_ = true;
  if (event_handler_.joinable()) {
    event_handler_.join();
  }

  // Start new thread with updated function pointer.
  terminate_ = false;
  event_handler_ = std::thread(&KafkaMessageSource<TKey>::run, this, std::move(callback));
}

template <typename TKey>
void KafkaMessageSource<TKey>::resubscribe() {
  // Get rid of previous subscription (if exits).
  HCTR_KAFKA_CHECK(rd_kafka_unsubscribe(rk_));

  // Define of topic partitions for subscription.
  HCTR_CHECK(tag_filters_.size() <= std::numeric_limits<int>::max());
  std::unique_ptr<rd_kafka_topic_partition_list_t, KafkaTopicPartitionListDeleter> part_list(
      rd_kafka_topic_partition_list_new(static_cast<int>(tag_filters_.size())));

  {
    auto log = HCTR_LOG_S(INFO, WORLD);
    log << "Attempting to (re-)subscribe to Kafka topics { ";

    for (const std::string& tag_filter : tag_filters_) {
      log << tag_filter;
      rd_kafka_topic_partition_list_add(part_list.get(), ("^" + tag_filter).c_str(),
                                        RD_KAFKA_PARTITION_UA);
    }

    log << " }" << std::endl;
  }

  // Enable subscription.
  HCTR_KAFKA_CHECK(rd_kafka_subscribe(rk_, part_list.get()));
}

template <typename TKey>
struct KafkaReceiveBuffer final {
  uint32_t value_size;
  std::vector<TKey> keys;
  std::vector<char> values;
  size_t msg_count = 0;  // messages processed since last commit.
  std::unique_ptr<rd_kafka_topic_partition_list_t, KafkaTopicPartitionListDeleter> next_offsets{
      rd_kafka_topic_partition_list_new(1)};

  KafkaReceiveBuffer() = delete;
  KafkaReceiveBuffer(const uint32_t _value_size, const size_t max_batch_size)
      : value_size{_value_size} {
    keys.reserve(max_batch_size);
    values.reserve(_value_size * max_batch_size);
  }
};

template <typename TKey>
void KafkaMessageSource<TKey>::run(std::function<HCTR_MESSAGE_SOURCE_CALLBACK> callback) {
  hctr_set_thread_name("kafka source");

  // Attempt to subscribe to topics.
  resubscribe();

  // Buffer for the messages and partition updates.
  phmap::flat_hash_map<std::string, KafkaReceiveBuffer<TKey>> recv_buffers;
  auto deliver = [&](const char* const topic, KafkaReceiveBuffer<TKey>& buf) -> bool {
    if (buf.keys.empty()) {
      return true;
    }
    HCTR_LOG_S(TRACE, WORLD) << "Kafka topic: " << topic << ", delivering " << buf.keys.size()
                             << " KV-pairs." << std::endl;

    // Retry until receiver signals that the delivery was successful.
    while (!callback(topic, buf.keys.size(), buf.keys.data(), buf.values.data(), buf.value_size)) {
      if (terminate_) {
        return false;
      }

      HCTR_LOG_S(WARNING, WORLD) << "Unable to deliver " << buf.keys.size()
                                 << " key/value pairs from Kafka topic " << topic << '.'
                                 << std::endl;
      std::this_thread::sleep_for(failure_backoff_ms_);
    }

    num_keys_delivered_ += buf.keys.size();
    buf.keys.clear();
    buf.values.clear();
    return true;
  };

  auto commit = [&](const char* const topic, KafkaReceiveBuffer<TKey>& buf) -> void {
    if (!buf.next_offsets->cnt) {
      return;
    }

    // Do the commit.
    {
      auto log = HCTR_LOG_S(TRACE, WORLD);
      log << "Committing Kafka topic: " << topic;
      for (int i = 0; i < buf.next_offsets->cnt; i++) {
        if (i) {
          log << ',';
        }
        const rd_kafka_topic_partition_t& elem = buf.next_offsets->elems[i];
        log << " { part = " << elem.partition << ", " << elem.offset << " }";
      }
      log << std::endl;
    }
    HCTR_KAFKA_CHECK(rd_kafka_commit(rk_, buf.next_offsets.get(), false));


    // Clear partition list.
    while (buf.next_offsets->cnt) {
      rd_kafka_topic_partition_list_del_by_idx(buf.next_offsets.get(), buf.next_offsets->cnt - 1);
    }

    // Update stats.
    num_keys_committed_ = num_keys_delivered_;
    num_messages_committed_ += buf.msg_count;
    buf.msg_count = 0;
  };

  while (!terminate_) {
    std::unique_ptr<rd_kafka_message_t, KafkaMessageDeleter> msg{
        rd_kafka_consumer_poll(rk_, static_cast<int>(poll_timeout_ms_))};

    // Timeout, or end of partition reached.
    if (!msg || msg->err == RD_KAFKA_RESP_ERR__TIMED_OUT ||
        msg->err == RD_KAFKA_RESP_ERR__PARTITION_EOF) {
      for (auto& recv_buffer_entry : recv_buffers) {
        const char* const topic = recv_buffer_entry.first.c_str();
        KafkaReceiveBuffer<TKey>& buf = recv_buffer_entry.second;

        // Hand over remaining data and commit.
        if (!deliver(topic, buf)) {
          break;
        }
        commit(topic, buf);
      }

      continue;
    }

    // Record in log if anything unexpected happened.
    if (msg->err != RD_KAFKA_RESP_ERR_NO_ERROR) {
      HCTR_LOG(WARNING, WORLD, "Kafka event/error %s: %s.\n", rd_kafka_err2name(msg->err),
               rd_kafka_message_errstr(msg.get()));

      // Backoff a little bit.
      std::this_thread::sleep_for(failure_backoff_ms_);
      continue;
    }

    // Tombstone (deleted record).
    if (!msg->payload) {
      HCTR_LOG(DEBUG, WORLD, "Kafka: Ignored tombstone.\n");
      continue;
    }

    // Retried message (duplicate).
    if (msg->offset == RD_KAFKA_OFFSET_INVALID) {
      HCTR_LOG(DEBUG, WORLD, "Kafka: Ignored duplicate message.\n");
      continue;
    }

    // Messages emitted by a sink shouldn't have a key and need to be at least 8 bytes long.
    if (msg->key_len != 0 || msg->len < sizeof(uint32_t) * 2) {
      auto log = HCTR_LOG_S(WARNING, WORLD);
      log << "Unexpected message. Data corruption? Discarding!" << std::endl
          << "Topic = " << rd_kafka_topic_name(msg->rkt) << std::endl
          << "Offset = " << msg->offset << std::endl
          << "Key (" << std::dec << msg->key_len << " bytes) =";
      {
        const char* const key = static_cast<const char*>(msg->key);
        for (size_t j = 0; j < msg->key_len; j++) {
          if (j % 2 == 0) {
            log << ' ';
          }
          log << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(key[j]);
        }
      }
      log << std::endl << "Payload (" << std::dec << msg->len << " bytes) =";
      {
        const char* const payload = static_cast<const char*>(msg->payload);
        for (size_t j = 0; j < msg->len; j++) {
          if (j % 4 == 0) {
            log << ' ';
          }
          log << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(payload[j]);
        }
      }
      log << std::endl;
      continue;
    }

    // Parse header.
    const char* p = static_cast<char*>(msg->payload);
    const char* const p_end = &p[msg->len];
    if (*reinterpret_cast<const uint32_t*>(p) != HCTR_KAFKA_VALUE_PREFIX) {
      HCTR_LOG(WARNING, WORLD,
               "Kafka message header contains unexpected values. Message discarded!\n");
      continue;
    }
    p += sizeof(uint32_t);
    const uint32_t value_size = *reinterpret_cast<const uint32_t*>(p);
    p += sizeof(uint32_t);

    // If this is just a beacon.
    if (p == p_end) {
      continue;
    }

    // Select receive buffer.
    const char* const topic = rd_kafka_topic_name(msg->rkt);
    KafkaReceiveBuffer<TKey>& buf =
        recv_buffers.try_emplace(topic, value_size, max_batch_size_).first->second;

    // Value size change detected. Hand over remaining data, commit and then change value_size.
    if (buf.value_size != value_size) {
      HCTR_LOG_S(WARNING, WORLD)
          << "The value_size for Kafka topic '" << topic << "' suddenly changed (" << buf.value_size
          << "<>" << value_size
          << "). Attempting to fix. But this might be an indicator of a more serious problem!"
          << std::endl;

      if (!deliver(topic, buf)) {
        break;
      }
      commit(topic, buf);
      buf.value_size = value_size;
    }

    // Copy data to receive buffer.
    while (p != p_end) {
      buf.keys.push_back(*reinterpret_cast<const TKey*>(p));
      p += sizeof(TKey);

      const char* const p_next = &p[value_size];
      buf.values.insert(buf.values.end(), p, p_next);
      p = p_next;

      // Deliver directly if receive buffer is full.
      if (buf.keys.size() >= max_batch_size_) {
        HCTR_LOG_S(TRACE, WORLD) << "Kafka topic '" << topic << "': Receive buffer is full."
                                 << std::endl;
        if (!deliver(topic, buf)) {
          break;
        }
      }
    }
    if (terminate_) {
      break;
    }

    // Message processed. Record offset.
    HCTR_LOG_S(TRACE, WORLD) << "Kafka topic '" << topic
                             << "': Message processed (offset = " << msg->offset << ")."
                             << std::endl;
    rd_kafka_topic_partition_t* part =
        rd_kafka_topic_partition_list_find(buf.next_offsets.get(), topic, msg->partition);
    if (!part) {
      part = rd_kafka_topic_partition_list_add(buf.next_offsets.get(), topic, msg->partition);
    }
    part->offset = msg->offset + 1;

    // If reached maximum commit interval, deliver and commit now.
    if (++buf.msg_count > max_commit_interval_) {
      HCTR_LOG_S(TRACE, WORLD) << " Kafka topic '" << topic << "': Commit interval reached."
                               << std::endl;
      if (!deliver(topic, buf)) {
        break;
      }
      commit(topic, buf);
    }
  }
}

template <typename TKey>
void KafkaMessageSource<TKey>::on_error(rd_kafka_resp_err_t err, const char* const reason) {
  HCTR_LOG_S(ERROR, WORLD) << "Kafka error " << rd_kafka_err2name(err) << ". Reason: " << reason
                           << std::endl;

  // If it is not fatal, just move on.
  if (err != RD_KAFKA_RESP_ERR__FATAL) {
    return;
  }

  // Write origin error to log.
  char error[HCTR_KAFKA_ERROR_STRING_LENGTH];
  err = rd_kafka_fatal_error(rk_, error, sizeof(error));
  HCTR_DIE("Fatal Kafka error encountered. Error: '%s', Reason: '%s'.\n", rd_kafka_err2name(err),
           error);
}

template class KafkaMessageSource<unsigned int>;
template class KafkaMessageSource<long long>;

#ifdef HCTR_KAFKA_CHECK
#undef HCTR_KAFKA_CHECK
#else
#error "HCTR_KAFKA_CHECK not defined?!"
#endif

}  // namespace HugeCTR
