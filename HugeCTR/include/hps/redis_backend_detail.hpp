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

#include <hiredis/hiredis.h>
#include <sw/redis++/redis++.h>

#include <base/debug/logger.hpp>
#include <charconv>
#include <iterator>
#include <type_traits>
#include <vector>

namespace sw {
namespace redis {
namespace reply {

/**
 * WARNING: The pointer becomes invalid once reply has been destroyed. Only use string_view's you
 * can immediately consume/parse values like the iterators below.
 */
inline sw::redis::StringView parse(ParseTag<sw::redis::StringView>, redisReply& reply) {
  if (!reply::is_string(reply) && !reply::is_status(reply)) {
    throw ProtoError("Expect STRING reply.");
  }

  if (reply.str == nullptr) {
    throw ProtoError("A null string reply.");
  }

  return {reply.str, reply.len};
}

}  // namespace reply
}  // namespace redis
}  // namespace sw

namespace HugeCTR {

// TODO: Remove me!
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wconversion"

/**
 * `base_type` for scalar type `std::vector` insert iterators.
 */
template <typename TargetValue, typename SourceValue>
class RedisInsertIterator : public std::iterator<std::output_iterator_tag, void, void, void, void> {
 public:
  static_assert(!std::is_same_v<TargetValue, SourceValue>);

  // To Redis++, we simulate allocation free string views, which are enabled through the parser at
  // the top of this file.
  using container_type = std::vector<SourceValue>;

 protected:
  inline RedisInsertIterator() = default;
};

/**
 * `base_type` for pair type `std::vector` insert iterators.
 */
template <typename Key, typename Value>
class RedisPairInsertIterator
    : public std::iterator<std::output_iterator_tag, void, void, void, void> {
 public:
  static_assert(!std::is_same_v<Key, sw::redis::StringView>);
  static_assert(!std::is_same_v<Value, sw::redis::StringView>);

  // To Redis++, we simulate allocation free string views, which are enabled through the parser at
  // the top of this file.
  using container_type = std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>>;

 protected:
  inline RedisPairInsertIterator() = default;
};

/**
 * Optimized `std::vector` insert iterator to parse redis reponses for commands like `KEYS` or
 * `HKEYS`.
 */
template <typename Key>
class RedisKeyVectorInserter : public RedisInsertIterator<Key, sw::redis::StringView> {
 public:
  static_assert(std::is_integral_v<Key>);

  RedisKeyVectorInserter() = delete;

  inline RedisKeyVectorInserter(std::vector<Key>& container) : container_(&container) {}

  inline RedisKeyVectorInserter& operator=(sw::redis::StringView&& k_View) {
    HCTR_CHECK(k_View.size() == sizeof(Key));
    container_->emplace_back(*reinterpret_cast<const Key*>(k_View.data()));
    return *this;
  }

  inline RedisKeyVectorInserter& operator*() { return *this; }
  inline RedisKeyVectorInserter& operator++() { return *this; }
  inline RedisKeyVectorInserter& operator++(int) { return *this; }

 protected:
  std::vector<Key>* const container_;
};

/**
 * Optimized `std::vector` insert iterator to parse redis `HMGET` and `HGETALL` reponses for LFU
 * meta-data.
 */
template <typename Key>
class RedisKeyAccumulatorVectorInserter : public RedisPairInsertIterator<Key, long long> {
 public:
  static_assert(std::is_integral_v<Key>);

  RedisKeyAccumulatorVectorInserter() = delete;

  inline RedisKeyAccumulatorVectorInserter(std::vector<std::pair<Key, long long>>& container)
      : container_(&container) {}

  inline RedisKeyAccumulatorVectorInserter& operator=(
      std::pair<sw::redis::StringView, sw::redis::StringView>&& kv_view) {
    // Convert Key
    HCTR_CHECK(kv_view.first.size() == sizeof(Key));
    const Key key{*reinterpret_cast<const Key*>(kv_view.first.data())};

    // Parse value.
    long long value;
    {
      const auto& res{std::from_chars(kv_view.second.begin(), kv_view.second.end(), value)};
      if (res.ec != std::errc()) {
        HCTR_LOG_C(
            ERROR, WORLD, "For key `", key,
            "`, Redis returned a character sequence that couldn't be parsed as `long long`.");
        value = 0;
      }
    }

    this->container_->emplace_back(key, value);
    return *this;
  }

  inline RedisKeyAccumulatorVectorInserter& operator*() { return *this; }
  inline RedisKeyAccumulatorVectorInserter& operator++() { return *this; }
  inline RedisKeyAccumulatorVectorInserter& operator++(int) { return *this; }

 protected:
  std::vector<std::pair<Key, long long>>* const container_;
};

/**
 * Optimized `std::vector` insert iterator to parse redis `HMGET` and `HGETALL` reponses for LRU
 * meta-data.
 */
template <typename Key>
class RedisKeyTimeVectorInserter : public RedisPairInsertIterator<Key, time_t> {
 public:
  static_assert(std::is_integral_v<Key>);

  RedisKeyTimeVectorInserter() = delete;

  inline RedisKeyTimeVectorInserter(std::vector<std::pair<Key, time_t>>& container)
      : container_(&container) {}

  inline RedisKeyTimeVectorInserter& operator=(
      std::pair<sw::redis::StringView, sw::redis::StringView>&& kv_view) {
    // Convert Key
    HCTR_CHECK(kv_view.first.size() == sizeof(Key));
    const Key key{*reinterpret_cast<const Key*>(kv_view.first.data())};

    // Convert time.
    HCTR_CHECK(kv_view.first.size() == sizeof(time_t));
    const time_t value{*reinterpret_cast<const time_t*>(kv_view.second.data())};

    this->container_->emplace_back(key, value);
    return *this;
  }

  inline RedisKeyTimeVectorInserter& operator*() { return *this; }
  inline RedisKeyTimeVectorInserter& operator++() { return *this; }
  inline RedisKeyTimeVectorInserter& operator++(int) { return *this; }

 protected:
  std::vector<std::pair<Key, time_t>>* const container_;
};

/**
 * Optimized iterator to parse redis reponses for the `HMGET` command and directly insert
 * them into strided destination memory locations.
 */
template <typename Key>
class RedisDirectValueInserter final
    : public RedisInsertIterator<Key, sw::redis::Optional<sw::redis::StringView>> {
 public:
  static_assert(std::is_integral_v<Key>);

  RedisDirectValueInserter() = delete;

  inline RedisDirectValueInserter(const Key* const keys,
                                  const std::vector<sw::redis::StringView>& k_views,
                                  char* const values, const size_t value_stride,
                                  const std::function<void(size_t)>& on_miss, size_t& miss_count,
                                  const DatabaseOverflowPolicy_t overflow_policy,
                                  std::shared_ptr<std::vector<Key>>& touched_keys)
      : keys{keys},
        k_views{&k_views},
        values{values},
        value_stride{value_stride},
        on_miss{&on_miss},
        miss_count(&miss_count),
        overflow_policy{overflow_policy},
        touched_keys{&touched_keys} {}

  inline RedisDirectValueInserter& operator=(sw::redis::Optional<sw::redis::StringView>&& v_view) {
    const Key* const k{reinterpret_cast<const Key*>(k_views->at(index++).data())};
    if (v_view) {
      HCTR_CHECK(v_view->size() <= value_stride);
      std::copy(v_view->begin(), v_view->end(), &values[(k - keys) * value_stride]);

      if (overflow_policy != DatabaseOverflowPolicy_t::EvictRandom) {
        if (!*touched_keys) {
          *touched_keys = std::make_shared<std::vector<Key>>();
        }
        (*touched_keys)->emplace_back(keys[k - keys]);
      }
    } else {
      (*on_miss)(k - keys);
      ++(*miss_count);
    }

    return *this;
  }

  inline RedisDirectValueInserter& operator*() { return *this; }
  inline RedisDirectValueInserter& operator++() { return *this; }
  inline RedisDirectValueInserter& operator++(int) { return *this; }

 protected:
  const Key* const keys;
  const std::vector<sw::redis::StringView>* const k_views;
  char* const values;
  const size_t value_stride;
  const std::function<void(size_t)>* const on_miss;
  size_t* const miss_count;
  const DatabaseOverflowPolicy_t overflow_policy;
  std::shared_ptr<std::vector<Key>>* touched_keys;
  size_t index{0};
};

/**
 * Optimized iterator to parse redis reponses for the `HMGET` command and directly
 * write them to a dump file.
 */
template <typename Key>
class RedisBinFileInserter final
    : public RedisInsertIterator<Key, sw::redis::Optional<sw::redis::StringView>> {
 public:
  static_assert(std::is_integral_v<Key>);

  RedisBinFileInserter() = delete;

  inline RedisBinFileInserter(const std::vector<sw::redis::StringView>& k_views,
                              uint32_t& value_size, std::ofstream& file, size_t& num_entries)
      : k_views{&k_views}, value_size{&value_size}, file{&file}, num_entries{&num_entries} {}

  inline RedisBinFileInserter& operator=(sw::redis::Optional<sw::redis::StringView>&& v_view) {
    if (v_view) {
      // Write value_size field if not already written.
      if (*value_size) {
        HCTR_CHECK(v_view->size() == *value_size);
      } else {
        HCTR_CHECK(v_view->size() != 0);
        *value_size = static_cast<uint32_t>(v_view->size());
        file->write(reinterpret_cast<const char*>(value_size), sizeof(uint32_t));
      }

      // Write the pair.
      const sw::redis::StringView& k_view{k_views->at(index)};
      file->write(k_view.data(), k_view.size());
      file->write(v_view->data(), v_view->size());
      ++(*num_entries);
    } else {
      HCTR_LOG_C(WARNING, WORLD, "The database was modified while dumping!\n");
    }

    ++index;
    return *this;
  }

  inline RedisBinFileInserter& operator*() { return *this; }
  inline RedisBinFileInserter& operator++() { return *this; }
  inline RedisBinFileInserter& operator++(int) { return *this; }

 protected:
  const std::vector<sw::redis::StringView>* const k_views;
  uint32_t* const value_size;
  std::ofstream* const file;
  size_t* const num_entries;
  size_t index{0};
};

/**
 * Redis Backend / Contains
 */
#ifdef HCTR_HPS_REDIS_CONTAINS_
#error HCTR_HPS_REDIS_CONTAINS_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_REDIS_CONTAINS_(MODE)                                                         \
  [&]() {                                                                                      \
    static_assert(std::is_same_v<decltype(hit_count), size_t>);                                \
                                                                                               \
    sw::redis::Pipeline pipe{redis_->pipeline(hkey_v, false)};                                 \
    HCTR_HPS_DB_APPLY_(MODE,                                                                   \
                       pipe.hexists(hkey_v, {reinterpret_cast<const char*>(k), sizeof(Key)})); \
                                                                                               \
    sw::redis::QueuedReplies replies{pipe.exec()};                                             \
    for (size_t idx{0}; idx < replies.size(); ++idx) {                                         \
      hit_count += replies.get<bool>(idx);                                                     \
    }                                                                                          \
    return true;                                                                               \
  }()

/**
 * Redis Backend / Evict
 */
#ifdef HCTR_HPS_REDIS_EVICT_
#error HCTR_HPS_REDIS_EVICT_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_REDIS_EVICT_(MODE)                                                                \
  [&]() {                                                                                          \
    static_assert(std::is_same_v<decltype(num_deletions), size_t>);                                \
    static_assert(std::is_same_v<decltype(k_views), std::vector<sw::redis::StringView>>);          \
                                                                                                   \
    k_views.clear();                                                                               \
    HCTR_HPS_DB_APPLY_(MODE, k_views.emplace_back(reinterpret_cast<const char*>(k), sizeof(Key))); \
                                                                                                   \
    sw::redis::Pipeline pipe{redis_->pipeline(hkey_v, false)};                                     \
    pipe.hdel(hkey_v, k_views.begin(), k_views.end());                                             \
    pipe.hdel(hkey_m, k_views.begin(), k_views.end());                                             \
                                                                                                   \
    sw::redis::QueuedReplies replies{pipe.exec()};                                                 \
    num_deletions += replies.get<long long>(0);                                                    \
    return true;                                                                                   \
  }()

/**
 * Redis Backend / Fetch
 */
#ifdef HCTR_HPS_REDIS_FETCH_
#error HCTR_HPS_REDIS_FETCH_ already defined. Potential naming conflict!
#endif
#if 0
#define HCTR_HPS_REDIS_FETCH_DEFINE_V_VIEWS() std::vector<sw::redis::Optional<std::string>> v_views
#define HCTR_HPS_REDIS_FETCH_(MODE)                                                                \
  [&]() {                                                                                          \
    static_assert(std::is_same_v<decltype(miss_count), size_t>);                                   \
    static_assert(std::is_same_v<decltype(k_views), std::vector<sw::redis::StringView>>);          \
                                                                                                   \
    k_views.clear();                                                                               \
    HCTR_HPS_DB_APPLY_(MODE, k_views.emplace_back(reinterpret_cast<const char*>(k), sizeof(Key))); \
                                                                                                   \
    v_views.clear();                                                                               \
    v_views.reserve(k_views.size());                                                               \
    redis_->hmget(hkey_v, k_views.begin(), k_views.end(), std::back_inserter(v_views));            \
                                                                                                   \
    for (auto v_it{v_views.begin()}; v_it != v_views.end(); ++v_it) {                              \
      const Key* const k{reinterpret_cast<const Key*>(k_views[v_it - v_views.begin()].data())};    \
      const sw::redis::Optional<std::string>& v_view{*v_it};                                       \
      if (v_view) {                                                                                \
        HCTR_CHECK(v_view->size() <= value_stride);                                                \
        std::copy(v_view->begin(), v_view->end(), &values[(k - keys) * value_stride]);             \
                                                                                                   \
        if (this->params_.overflow_policy != DatabaseOverflowPolicy_t::EvictRandom) {              \
          if (!touched_keys) {                                                                     \
            touched_keys = std::make_shared<std::vector<Key>>();                                   \
          }                                                                                        \
          touched_keys->emplace_back(keys[k - keys]);                                              \
        }                                                                                          \
      } else {                                                                                     \
        on_miss(k - keys);                                                                         \
        ++miss_count;                                                                              \
      }                                                                                            \
    }                                                                                              \
    return true;                                                                                   \
  }()
#else
#define HCTR_HPS_REDIS_FETCH_DEFINE_V_VIEWS() \
  do {                                        \
  } while (0)
#define HCTR_HPS_REDIS_FETCH_(MODE)                                                                \
  [&]() {                                                                                          \
    static_assert(std::is_same_v<decltype(k_views), std::vector<sw::redis::StringView>>);          \
                                                                                                   \
    k_views.clear();                                                                               \
    HCTR_HPS_DB_APPLY_(MODE, k_views.emplace_back(reinterpret_cast<const char*>(k), sizeof(Key))); \
                                                                                                   \
    redis_->hmget(                                                                                 \
        hkey_v, k_views.begin(), k_views.end(),                                                    \
        RedisDirectValueInserter<Key>(keys, k_views, values, value_stride, on_miss, miss_count,    \
                                      this->params_.overflow_policy, touched_keys));               \
    return true;                                                                                   \
  }()
#endif

#ifdef HCTR_HPS_REDIS_INSERT_
#error HCTR_HPS_REDIS_INSERT_ already defined. Potential naming conflict!
#endif
#define HCTR_HPS_REDIS_INSERT_(MODE)                                                           \
  [&]() {                                                                                      \
    static_assert(std::is_same_v<decltype(num_inserts), size_t>);                              \
    static_assert(std::is_same_v<decltype(part_size), size_t>);                                \
    static_assert(std::is_same_v<decltype(value_size), const uint32_t>);                       \
    static_assert(std::is_same_v<decltype(value_stride), const size_t>);                       \
    static_assert(                                                                             \
        std::is_same_v<decltype(kv_views),                                                     \
                       std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>>>); \
    static_assert(                                                                             \
        std::is_same_v<decltype(km_views),                                                     \
                       std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>>>); \
                                                                                               \
    sw::redis::Pipeline pipe{redis_->pipeline(hkey_v, false)};                                 \
                                                                                               \
    kv_views.clear();                                                                          \
    km_views.clear();                                                                          \
                                                                                               \
    switch (this->params_.overflow_policy) {                                                   \
      case DatabaseOverflowPolicy_t::EvictRandom: {                                            \
        HCTR_HPS_DB_APPLY_(                                                                    \
            MODE, kv_views.emplace_back(                                                       \
                      std::piecewise_construct,                                                \
                      std::forward_as_tuple(reinterpret_cast<const char*>(k), sizeof(Key)),    \
                      std::forward_as_tuple(&values[(k - keys) * value_stride], value_size))); \
      } break;                                                                                 \
      case DatabaseOverflowPolicy_t::EvictLeastUsed: {                                         \
        HCTR_HPS_DB_APPLY_(MODE, {                                                             \
          kv_views.emplace_back(                                                               \
              std::piecewise_construct,                                                        \
              std::forward_as_tuple(reinterpret_cast<const char*>(k), sizeof(Key)),            \
              std::forward_as_tuple(&values[(k - keys) * value_stride], value_size));          \
          pipe.hincrby(hkey_m, {reinterpret_cast<const char*>(k), sizeof(Key)}, 1);            \
        });                                                                                    \
      } break;                                                                                 \
      case DatabaseOverflowPolicy_t::EvictOldest: {                                            \
        const time_t now = std::time(nullptr);                                                 \
        HCTR_HPS_DB_APPLY_(MODE, {                                                             \
          kv_views.emplace_back(                                                               \
              std::piecewise_construct,                                                        \
              std::forward_as_tuple(reinterpret_cast<const char*>(k), sizeof(Key)),            \
              std::forward_as_tuple(&values[(k - keys) * value_stride], value_size));          \
          km_views.emplace_back(                                                               \
              std::piecewise_construct,                                                        \
              std::forward_as_tuple(reinterpret_cast<const char*>(k), sizeof(Key)),            \
              std::forward_as_tuple(reinterpret_cast<const char*>(&now), sizeof(time_t)));     \
        });                                                                                    \
        pipe.hset(hkey_m, km_views.begin(), km_views.end());                                   \
      } break;                                                                                 \
    }                                                                                          \
    pipe.hset(hkey_v, kv_views.begin(), kv_views.end());                                       \
    pipe.hlen(hkey_v);                                                                         \
                                                                                               \
    sw::redis::QueuedReplies replies{pipe.exec()};                                             \
    num_inserts += std::max(replies.get<long long>(replies.size() - 2), 0LL);                  \
    part_size = std::max(replies.get<long long>(replies.size() - 1), 0LL);                     \
    return true;                                                                               \
  }()

// TODO: Remove me!
#pragma GCC diagnostic pop

}  // namespace HugeCTR