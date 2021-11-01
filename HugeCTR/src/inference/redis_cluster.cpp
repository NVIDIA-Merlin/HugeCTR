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

// Copy and modify part code from R3C,which is a C++ client for redis based on hiredis
// (https://github.com/redis/hiredis)
#include <assert.h>
#include <inference/redis_cluster.h>
#include <inference/utils.h>

#define R3C_ASSERT assert
#define THROW_REDIS_EXCEPTION(errinfo) throw CRedisException(errinfo, __FILE__, __LINE__)
#define THROW_REDIS_EXCEPTION_WITH_NODE(errinfo, node_ip, node_port) \
  throw CRedisException(errinfo, __FILE__, __LINE__, node_ip, node_port)
#define THROW_REDIS_EXCEPTION_WITH_NODE_AND_COMMAND(errinfo, node_ip, node_port, command, key) \
  throw CRedisException(errinfo, __FILE__, __LINE__, node_ip, node_port, command, key)

namespace r3c {

int NUM_RETRIES = 15;  // The default number of retries is 15 (CLUSTERDOWN cost more than 6s)
int CONNECT_TIMEOUT_MILLISECONDS = 2000;    // Connection timeout in milliseconds
int READWRITE_TIMEOUT_MILLISECONDS = 2000;  // Receive and send timeout in milliseconds

#if R3C_TEST  // for test
static LOG_WRITE g_error_log = r3c_log_write;
static LOG_WRITE g_info_log = r3c_log_write;
static LOG_WRITE g_debug_log = r3c_log_write;
#else
static LOG_WRITE g_error_log = null_log_write;
static LOG_WRITE g_info_log = null_log_write;
static LOG_WRITE g_debug_log = null_log_write;
#endif  // R3C_TEST

void set_error_log_write(LOG_WRITE info_log) {
  g_error_log = info_log;
  if (NULL == g_error_log) g_error_log = null_log_write;
}

void set_info_log_write(LOG_WRITE info_log) {
  g_info_log = info_log;
  if (NULL == g_info_log) g_info_log = null_log_write;
}

void set_debug_log_write(LOG_WRITE debug_log) {
  g_debug_log = debug_log;
  if (NULL == g_debug_log) g_debug_log = null_log_write;
}

static int get_retry_sleep_milliseconds(int loop_counter) {
  static const int sleep_milliseconds_table[] = {10, 100, 200, 500, 1000};
  if (loop_counter < 0 ||
      loop_counter >=
          static_cast<int>(sizeof(sleep_milliseconds_table) / sleep_milliseconds_table[0] - 1))
    return 1000;
  else
    return sleep_milliseconds_table[loop_counter];
}

enum {
  CLUSTER_SLOTS = 16384  // number of slots, defined in cluster.h
};

size_t NodeHasher::operator()(const Node& node) const {
  const unsigned char* s = reinterpret_cast<const unsigned char*>(node.first.c_str());
  const uint64_t l = static_cast<uint64_t>(node.first.size());
  uint64_t crc = 0;
  return static_cast<size_t>(crc64(crc, s, l));
}

std::string& node2string(const Node& node, std::string* str) {
  *str = node.first + std::string(":") + int2string(node.second);
  return *str;
}

std::string node2string(const Node& node) {
  std::string nodestr;
  return node2string(node, &nodestr);
}

std::string NodeInfo::str() const {
  return format_string("nodeinfo://%s/%s:%d/%s", id.c_str(), node.first.c_str(), node.second,
                       flags.c_str());
}

bool NodeInfo::is_master() const {
  const std::string::size_type pos = flags.find("master");
  return (pos != std::string::npos);
}

bool NodeInfo::is_replica() const {
  const std::string::size_type pos = flags.find("slave");
  return (pos != std::string::npos);
}

bool NodeInfo::is_fail() const {
  const std::string::size_type pos = flags.find("fail");
  return (pos != std::string::npos);
}

////////////////////////////////////////////////////////////////////////////////
// CCommandArgs

CommandArgs::CommandArgs() : _argc(0), _argv(NULL), _argvlen(NULL) {}

CommandArgs::~CommandArgs() {
  delete[] _argvlen;
  for (int i = 0; i < _argc; ++i) delete[] _argv[i];
  delete[] _argv;
}

void CommandArgs::set_key(const std::string& key) { _key = key; }

void CommandArgs::set_command(const std::string& command) { _command = command; }

void CommandArgs::add_arg(const std::string& arg) { _args.push_back(arg); }

void CommandArgs::add_arg(int32_t arg) { _args.push_back(int2string(arg)); }

void CommandArgs::add_arg(uint32_t arg) { _args.push_back(int2string(arg)); }

void CommandArgs::add_arg(int64_t arg) { _args.push_back(int2string(arg)); }

void CommandArgs::add_args(const std::vector<std::string>& args) {
  for (std::vector<std::string>::size_type i = 0; i < args.size(); ++i) {
    const std::string& arg = args[i];
    add_arg(arg);
  }
}

void CommandArgs::add_args(const std::vector<std::pair<std::string, std::string> >& values) {
  for (std::vector<std::string>::size_type i = 0; i < values.size(); ++i) {
    const std::string& field = values[i].first;
    const std::string& value = values[i].second;
    add_arg(field);
    add_arg(value);
  }
}

void CommandArgs::add_args(const std::map<std::string, std::string>& map) {
  for (std::map<std::string, std::string>::const_iterator iter = map.begin(); iter != map.end();
       ++iter) {
    add_arg(iter->first);
    add_arg(iter->second);
  }
}

void CommandArgs::add_args(const std::map<std::string, int64_t>& map, bool reverse) {
  for (std::map<std::string, int64_t>::const_iterator iter = map.begin(); iter != map.end();
       ++iter) {
    if (!reverse) {
      add_arg(iter->first);
      add_arg(iter->second);
    } else {
      add_arg(iter->second);
      add_arg(iter->first);
    }
  }
}

void CommandArgs::add_args(const std::vector<FVPair>& fvpairs) {
  for (std::vector<FVPair>::size_type i = 0; i < fvpairs.size(); ++i) {
    const FVPair& fvpair = fvpairs[i];
    add_arg(fvpair.field);
    add_arg(fvpair.value);
  }
}

void CommandArgs::final() {
  _argc = static_cast<int>(_args.size());
  _argv = new char*[_argc];
  _argvlen = new size_t[_argc];

  for (int i = 0; i < _argc; ++i) {
    _argvlen[i] = _args[i].size();
    _argv[i] = new char[_argvlen[i] + 1];
    memcpy(_argv[i], _args[i].c_str(), _argvlen[i]);  // Support binary key&value.
    _argv[i][_argvlen[i]] = '\0';
  }
}

int CommandArgs::get_argc() const { return _argc; }

const char** CommandArgs::get_argv() const { return (const char**)_argv; }

const size_t* CommandArgs::get_argvlen() const { return _argvlen; }

const std::string& CommandArgs::get_command() const { return _command; }

const std::string& CommandArgs::get_key() const { return _key; }

////////////////////////////////////////////////////////////////////////////////
// CRedisNode
// CRedisMasterNode
// CRedisReplicaNode

class CRedisNode {
 public:
  CRedisNode(const NodeId& nodeid, const Node& node, redisContext* redis_context)
      : _nodeid(nodeid), _node(node), _redis_context(redis_context), _conn_errors(0) {}

  ~CRedisNode() { close(); }

  const NodeId& get_nodeid() const { return _nodeid; }

  const Node& get_node() const { return _node; }

  redisContext* get_redis_context() const { return _redis_context; }

  void set_redis_context(redisContext* redis_context) {
    _redis_context = redis_context;
    if (NULL == _redis_context) {
      ++_conn_errors;
      //(*g_debug_log)("%s\n", str().c_str());
    }
  }

  void close() {
    if (_redis_context != NULL) {
      redisFree(_redis_context);
      _redis_context = NULL;
    }
  }

  std::string str() const {
    return format_string("node://(connerrors:%u)%s:%d", _conn_errors, _node.first.c_str(),
                         _node.second);
  }

  unsigned int get_conn_errors() const { return _conn_errors; }

  void inc_conn_errors() { ++_conn_errors; }

  void reset_conn_errors() { _conn_errors = 0; }

  void set_conn_errors(unsigned int conn_errors) { _conn_errors = conn_errors; }

  bool need_refresh_master() const {
    return ((_conn_errors > 3 && 0 == _conn_errors % 3) || (_conn_errors > 2018));
  }

 protected:
  NodeId _nodeid;
  Node _node;
  redisContext* _redis_context;
  unsigned int _conn_errors;
};

class CRedisMasterNode;

class CRedisReplicaNode : public CRedisNode {
 public:
  CRedisReplicaNode(const NodeId& node_id, const Node& node, redisContext* redis_context)
      : CRedisNode(node_id, node, redis_context), _redis_master_node(NULL) {}

 private:
  CRedisMasterNode* _redis_master_node;
};

class CRedisMasterNode : public CRedisNode {
 public:
  CRedisMasterNode(const NodeId& node_id, const Node& node, redisContext* redis_context)
      : CRedisNode(node_id, node, redis_context), _index(0) {}

  ~CRedisMasterNode() { clear(); }

  void clear() {
    for (RedisReplicaNodeTable::iterator iter = _redis_replica_nodes.begin();
         iter != _redis_replica_nodes.end(); ++iter) {
      CRedisReplicaNode* replica_node = iter->second;
      delete replica_node;
    }
    _redis_replica_nodes.clear();
  }

  void add_replica_node(CRedisReplicaNode* redis_replica_node) {
    const Node& node = redis_replica_node->get_node();
    const std::pair<RedisReplicaNodeTable::iterator, bool> ret =
        _redis_replica_nodes.insert(std::make_pair(node, redis_replica_node));
    if (!ret.second) {
      CRedisReplicaNode* old_replica_node = ret.first->second;
      delete old_replica_node;
      ret.first->second = redis_replica_node;
    }
  }

  CRedisNode* choose_node(ReadPolicy read_policy) {
    const unsigned int num_redis_replica_nodes =
        static_cast<unsigned int>(_redis_replica_nodes.size());
    CRedisNode* redis_node = NULL;

    if (0 == num_redis_replica_nodes) {
      redis_node = this;
    } else {
      unsigned int K = _index++ % (num_redis_replica_nodes + 1);  // Included master

      if (RP_READ_REPLICA == read_policy && K == num_redis_replica_nodes) {
        redis_node = this;
      } else {
        RedisReplicaNodeTable::iterator iter = _redis_replica_nodes.begin();

        for (unsigned int i = 0; i < K; ++i) {
          if (++iter == _redis_replica_nodes.end()) iter = _redis_replica_nodes.begin();
        }
        if (iter != _redis_replica_nodes.end()) {
          redis_node = iter->second;
        }
        if (NULL == redis_node) {
          redis_node = this;
        }
      }
    }

    return redis_node;
  }

 private:
#if __cplusplus < 201103L
  typedef std::tr1::unordered_map<Node, CRedisReplicaNode*, NodeHasher> RedisReplicaNodeTable;
#else
  typedef std::unordered_map<Node, CRedisReplicaNode*, NodeHasher> RedisReplicaNodeTable;
#endif  // __cplusplus < 201103L
  RedisReplicaNodeTable _redis_replica_nodes;
  unsigned int _index;
};

////////////////////////////////////////////////////////////////////////////////
// RedisReplyHelper

RedisReplyHelper::RedisReplyHelper() : _redis_reply(NULL) {}

RedisReplyHelper::RedisReplyHelper(const redisReply* redis_reply) : _redis_reply(redis_reply) {}

RedisReplyHelper::RedisReplyHelper(const RedisReplyHelper& other) { _redis_reply = other.detach(); }

RedisReplyHelper::~RedisReplyHelper() { free(); }

RedisReplyHelper::operator bool() const { return _redis_reply != NULL; }

void RedisReplyHelper::free() {
  if (_redis_reply != NULL) {
    freeReplyObject((void*)_redis_reply);
    _redis_reply = NULL;
  }
}

const redisReply* RedisReplyHelper::get() const { return _redis_reply; }

const redisReply* RedisReplyHelper::detach() const {
  const redisReply* redis_reply = _redis_reply;
  _redis_reply = NULL;
  return redis_reply;
}

RedisReplyHelper& RedisReplyHelper::operator=(const redisReply* redis_reply) {
  free();
  _redis_reply = redis_reply;
  return *this;
}

RedisReplyHelper& RedisReplyHelper::operator=(const RedisReplyHelper& other) {
  free();
  _redis_reply = other.detach();
  return *this;
}

const redisReply* RedisReplyHelper::operator->() const { return _redis_reply; }

std::ostream& RedisReplyHelper::operator<<(std::ostream& os) {
  os << *_redis_reply;
  return os;
}

////////////////////////////////////////////////////////////////////////////////
// ErrorInfo

ErrorInfo::ErrorInfo() : errcode(0) {}

ErrorInfo::ErrorInfo(const std::string& raw_errmsg_, const std::string& errmsg_,
                     const std::string& errtype_, int errcode_)
    : raw_errmsg(raw_errmsg_), errmsg(errmsg_), errtype(errtype_), errcode(errcode_) {}

void ErrorInfo::clear() {
  errcode = 0;
  errtype.clear();
  errmsg.clear();
  raw_errmsg.clear();
}

////////////////////////////////////////////////////////////////////////////////
// CRedisException

CRedisException::CRedisException(const struct ErrorInfo& errinfo, const char* file, int line,
                                 const std::string& node_ip, uint16_t node_port,
                                 const std::string& command, const std::string& key) throw()
    : _errinfo(errinfo),
      _line(line),
      _node_ip(node_ip),
      _node_port(node_port),
      _command(command),
      _key(key) {
  const char* slash_position = strrchr(file, '/');
  const std::string* file_cp = &_file;
  std::string* file_p = const_cast<std::string*>(file_cp);

  if (NULL == slash_position) {
    *file_p = file;
  } else {
    *file_p = std::string(slash_position + 1);
  }
}

const char* CRedisException::what() const throw() { return _errinfo.errmsg.c_str(); }

std::string CRedisException::str() const throw() {
  const std::string& errmsg =
      format_string("redis_exception://%s:%d/CMD:%s/%s/(%d)%s@%s:%d", _node_ip.c_str(), _node_port,
                    _command.c_str(), _errinfo.errtype.c_str(), _errinfo.errcode,
                    _errinfo.errmsg.c_str(), _file.c_str(), _line);

#if __cplusplus < 201103L
  return errmsg;
#else
  return std::move(errmsg);
#endif  // __cplusplus < 201103L
}

bool is_general_error(const std::string& errtype) {
  return (errtype.size() == sizeof("ERR") - 1) && (errtype == "ERR");
}

bool is_ask_error(const std::string& errtype) {
  return (errtype.size() == sizeof("ASK") - 1) && (errtype == "ASK");
}

bool is_clusterdown_error(const std::string& errtype) {
  // CLUSTERDOWN The cluster is down
  return (errtype.size() == sizeof("CLUSTERDOWN") - 1) && (errtype == "CLUSTERDOWN");
}

bool is_moved_error(const std::string& errtype) {
  return (errtype.size() == sizeof("MOVED") - 1) && (errtype == "MOVED");
}

bool is_noauth_error(const std::string& errtype) {
  // NOAUTH Authentication required.
  return (errtype.size() == sizeof("NOAUTH") - 1) && (errtype == "NOAUTH");
}

bool is_noscript_error(const std::string& errtype) {
  // NOSCRIPT No matching script. Please use EVAL.
  return (errtype.size() == sizeof("NOSCRIPT") - 1) && (errtype == "NOSCRIPT");
}

bool is_wrongtype_error(const std::string& errtype) {
  return (errtype.size() == sizeof("WRONGTYPE") - 1) && (errtype == "WRONGTYPE");
}

bool is_busygroup_error(const std::string& errtype) {
  return (errtype.size() == sizeof("BUSYGROUP") - 1) && (errtype == "BUSYGROUP");
}

bool is_nogroup_error(const std::string& errtype) {
  return (errtype.size() == sizeof("NOGROUP") - 1) && (errtype == "NOGROUP");
}

bool is_crossslot_error(const std::string& errtype) {
  return (errtype.size() == sizeof("CROSSSLOT") - 1) && (errtype == "CROSSSLOT");
}

////////////////////////////////////////////////////////////////////////////////
// CRedisClient

CRedisClient::CRedisClient(const std::string& raw_nodes_string, int connect_timeout_milliseconds,
                           int readwrite_timeout_milliseconds, const std::string& password,
                           ReadPolicy read_policy)
    : _command_monitor(NULL),
      _raw_nodes_string(raw_nodes_string),
      _connect_timeout_milliseconds(connect_timeout_milliseconds),
      _readwrite_timeout_milliseconds(readwrite_timeout_milliseconds),
      _password(password),
      _read_policy(read_policy) {
  init();
}

CRedisClient::CRedisClient(const std::string& raw_nodes_string, const std::string& password,
                           int connect_timeout_milliseconds, int readwrite_timeout_milliseconds,
                           ReadPolicy read_policy)
    : _command_monitor(NULL),
      _raw_nodes_string(raw_nodes_string),
      _connect_timeout_milliseconds(connect_timeout_milliseconds),
      _readwrite_timeout_milliseconds(readwrite_timeout_milliseconds),
      _password(password),
      _read_policy(read_policy) {
  init();
}

CRedisClient::~CRedisClient() { fini(); }

const std::string& CRedisClient::get_raw_nodes_string() const { return _raw_nodes_string; }

const std::string& CRedisClient::get_nodes_string() const { return _nodes_string; }

std::string CRedisClient::str() const {
  if (cluster_mode())
    return std::string("rediscluster://") + _raw_nodes_string;
  else
    return std::string("redisstandalone://") + _raw_nodes_string;
}

bool CRedisClient::cluster_mode() const { return _nodes.size() > 1; }

const char* CRedisClient::get_mode_str() const { return cluster_mode() ? "CLUSTER" : "STANDALONE"; }

void CRedisClient::enable_debug_log() { _enable_debug_log = true; }

void CRedisClient::disable_debug_log() { _enable_debug_log = false; }

void CRedisClient::enable_info_log() { _enable_info_log = true; }

void CRedisClient::disable_info_log() { _enable_info_log = false; }

void CRedisClient::enable_error_log() { _enable_error_log = true; }

void CRedisClient::disable_error_log() { _enable_error_log = false; }

int CRedisClient::list_nodes(std::vector<struct NodeInfo>* nodes_info) {
  struct ErrorInfo errinfo;

  for (RedisMasterNodeTable::iterator iter = _redis_master_nodes.begin();
       iter != _redis_master_nodes.end(); ++iter) {
    const Node& node = iter->first;
    struct CRedisNode* redis_node = iter->second;
    redisContext* redis_context = redis_node->get_redis_context();

    if (redis_context != NULL) {
      if (list_cluster_nodes(nodes_info, &errinfo, redis_context, node)) break;
    }
  }
  if (nodes_info->empty()) {
    THROW_REDIS_EXCEPTION(errinfo);
  }

  return static_cast<int>(nodes_info->size());
}

void CRedisClient::flushall() {
  const int num_retries = NUM_RETRIES;
  const std::string key;
  CommandArgs cmd_args;
  cmd_args.set_command("FLUSHALL");
  cmd_args.add_arg(cmd_args.get_command());
  cmd_args.final();

  // Simple string reply (REDIS_REPLY_STATUS)
  redis_command(true, num_retries, key, cmd_args, NULL);
}

void CRedisClient::multi(const std::string& key, Node* which) {
  if (cluster_mode()) {
    struct ErrorInfo errinfo;
    errinfo.errcode = ERROR_NOT_SUPPORT;
    errinfo.errmsg = "MULTI not supported in cluster mode";
    THROW_REDIS_EXCEPTION(errinfo);
  } else {
    const int num_retries = 0;
    CommandArgs cmd_args;
    cmd_args.set_key(key);
    cmd_args.set_command("MULTI");
    cmd_args.add_arg(cmd_args.get_command());
    cmd_args.final();

    // Simple string reply (REDIS_REPLY_STATUS):
    // always OK.
    redis_command(false, num_retries, key, cmd_args, which);
  }
}

const RedisReplyHelper CRedisClient::exec(const std::string& key, Node* which) {
  if (cluster_mode()) {
    struct ErrorInfo errinfo;
    errinfo.errcode = ERROR_NOT_SUPPORT;
    errinfo.errmsg = "EXEC not supported in cluster mode";
    THROW_REDIS_EXCEPTION(errinfo);
  } else {
    const int num_retries = 0;
    CommandArgs cmd_args;
    cmd_args.set_key(key);
    cmd_args.set_command("EXEC");
    cmd_args.add_arg(cmd_args.get_command());
    cmd_args.final();

    // Array reply:
    // each element being the reply to each of the commands in the atomic transaction.
    return redis_command(false, num_retries, key, cmd_args, which);
  }
}

//
// KEY/VALUE
//

// Time complexity: O(1)
// EXPIRE key seconds
bool CRedisClient::expire(const std::string& key, uint32_t seconds, Node* which, int num_retries) {
  CommandArgs cmd_args;
  cmd_args.set_key(key);
  cmd_args.set_command("EXPIRE");
  cmd_args.add_arg(cmd_args.get_command());
  cmd_args.add_arg(key);
  cmd_args.add_arg(seconds);
  cmd_args.final();

  // Integer reply, specifically:
  // 1 if the timeout was set.
  // 0 if key does not exist.
  const RedisReplyHelper redis_reply = redis_command(false, num_retries, key, cmd_args, which);
  if (REDIS_REPLY_INTEGER == redis_reply->type) return 1 == redis_reply->integer;
  return true;
}

// Time complexity: O(1)
// EXISTS key [key ...]
bool CRedisClient::exists(const std::string& key, Node* which, int num_retries) {
  CommandArgs cmd_args;
  cmd_args.set_key(key);
  cmd_args.set_command("EXISTS");
  cmd_args.add_arg(cmd_args.get_command());
  cmd_args.add_arg(key);
  cmd_args.final();

  // Integer reply, specifically:
  // 1 if the key exists.
  // 0 if the key does not exist.
  const RedisReplyHelper redis_reply = redis_command(true, num_retries, key, cmd_args, which);
  if (REDIS_REPLY_INTEGER == redis_reply->type) return 1 == redis_reply->integer;
  return true;
}

// Time complexity:
// O(N) where N is the number of keys that will be removed.
// When a key to remove holds a value other than a string,
// the individual complexity for this key is O(M) where M is the number of elements in the list,
// set, sorted set or hash. Removing a single key that holds a string value is O(1).
bool CRedisClient::del(const std::string& key, Node* which, int num_retries) {
  CommandArgs cmd_args;
  cmd_args.set_key(key);
  cmd_args.set_command("DEL");
  cmd_args.add_arg(cmd_args.get_command());
  cmd_args.add_arg(key);
  cmd_args.final();

  // Integer reply:
  // The number of keys that were removed.
  const RedisReplyHelper redis_reply = redis_command(false, num_retries, key, cmd_args, which);
  if (REDIS_REPLY_INTEGER == redis_reply->type) return 1 == redis_reply->integer;
  return true;
}

// GET key
// Time complexity: O(1)
bool CRedisClient::get(const std::string& key, std::string* value, Node* which, int num_retries) {
  CommandArgs cmd_args;
  cmd_args.set_key(key);
  cmd_args.set_command("GET");
  cmd_args.add_arg(cmd_args.get_command());
  cmd_args.add_arg(key);
  cmd_args.final();

  // Bulk string reply:
  // the value of key, or nil when key does not exist.
  RedisReplyHelper redis_reply = redis_command(true, num_retries, key, cmd_args, which);
  if (REDIS_REPLY_NIL == redis_reply->type) return false;
  if (REDIS_REPLY_STRING == redis_reply->type) return get_value(redis_reply.get(), value);
  return true;
}

// SET key value [EX seconds] [PX milliseconds] [NX|XX]
// Time complexity: O(1)
void CRedisClient::set(const std::string& key, const std::string& value, Node* which,
                       int num_retries) {
  CommandArgs cmd_args;
  cmd_args.set_key(key);
  cmd_args.set_command("SET");
  cmd_args.add_arg(cmd_args.get_command());
  cmd_args.add_arg(key);
  cmd_args.add_arg(value);
  cmd_args.final();

  // Simple string reply (REDIS_REPLY_STATUS):
  // OK if SET was executed correctly.

  // Null reply: a Null Bulk Reply is returned if the SET operation was not performed
  // because the user specified the NX or XX option but the condition was not met.
  //
  // OK: redis_reply->str
  redis_command(false, num_retries, key, cmd_args, which);
}

// Time complexity: O(N) where N is the number of keys to retrieve.
// MGET key [key ...]
int CRedisClient::mget(const std::vector<std::string>& keys, std::vector<std::string>* values,
                       Node* which, int num_retries) {
  values->clear();

  if (!cluster_mode()) {
    const std::string key;
    CommandArgs cmd_args;
    cmd_args.set_command("MGET");
    cmd_args.add_arg(cmd_args.get_command());
    cmd_args.add_args(keys);
    cmd_args.final();

    // Array reply:
    // list of values at the specified keys.
    //
    // For every key that does not hold a string value or does not exist,
    // the special value nil is returned. Because of this, the operation never fails.
    const RedisReplyHelper redis_reply = redis_command(true, num_retries, key, cmd_args, which);
    if (REDIS_REPLY_ARRAY == redis_reply->type) return get_values(redis_reply.get(), values);
    return 0;
  } else {
    values->resize(keys.size());

    try {
      for (std::vector<std::string>::size_type i = 0; i < keys.size(); ++i) {
        const std::string& key = keys[i];
        std::string& value = (*values)[i];
        get(key, &value, NULL, num_retries);
      }
    } catch (CRedisException&) {
      values->clear();
      throw;
    }

    return static_cast<int>(values->size());
  }
}

// Time complexity:
// O(N) where N is the number of keys to set.
// MSET key value [key value ...]
int CRedisClient::mset(const std::map<std::string, std::string>& kv_map, Node* which,
                       int num_retries) {
  int success = 0;

  if (kv_map.empty()) {
    struct ErrorInfo errinfo;
    errinfo.errcode = ERROR_PARAMETER;
    errinfo.errmsg = "kv_map is empty";
    THROW_REDIS_EXCEPTION(errinfo);
  }
  if (!cluster_mode()) {
    const std::string key;
    CommandArgs cmd_args;
    cmd_args.set_command("MSET");
    cmd_args.add_arg(cmd_args.get_command());
    cmd_args.add_args(kv_map);
    cmd_args.final();

    // Simple string reply:
    // always OK since MSET can't fail.
    redis_command(false, num_retries, key, cmd_args, which);
    success = static_cast<int>(kv_map.size());
  } else {
    for (std::map<std::string, std::string>::const_iterator iter = kv_map.begin();
         iter != kv_map.end(); ++iter) {
      const std::string& key = iter->first;
      const std::string& value = iter->second;
      set(key, value, which, num_retries);
      ++success;
    }
  }

  return success;
}

bool CRedisClient::key_type(const std::string& key, std::string* key_type, Node* which,
                            int num_retries) {
  CommandArgs cmd_args;
  cmd_args.set_key(key);
  cmd_args.set_command("TYPE");
  cmd_args.add_arg(cmd_args.get_command());
  cmd_args.add_arg(key);
  cmd_args.final();

  // Simple string reply (REDIS_REPLY_STATUS):
  // type of key, or none when key does not exist.
  //
  // (gdb) p *redis_reply._redis_reply
  // $1 = {type = 5, integer = 0, len = 6, str = 0x742b50 "string", elements = 0, element = 0x0}
  const RedisReplyHelper redis_reply = redis_command(true, num_retries, key, cmd_args, which);
  return get_value(redis_reply.get(), key_type);
}

int64_t CRedisClient::ttl(const std::string& key, Node* which, int num_retries) {
  CommandArgs cmd_args;
  cmd_args.set_key(key);
  cmd_args.set_command("TTL");
  cmd_args.add_arg(cmd_args.get_command());
  cmd_args.add_arg(key);
  cmd_args.final();

  // Integer reply:
  // TTL in seconds, or a negative value in order to signal an error
  const RedisReplyHelper redis_reply = redis_command(true, num_retries, key, cmd_args, which);
  if (REDIS_REPLY_INTEGER == redis_reply->type) return redis_reply->integer;
  return 0;
}

////////////////////////////////////////////////////////////////////////////////

const RedisReplyHelper CRedisClient::redis_command(bool readonly, int num_retries,
                                                   const std::string& key,
                                                   const CommandArgs& command_args, Node* which) {
  Node node;
  Node* ask_node = NULL;
  RedisReplyHelper redis_reply;
  struct ErrorInfo errinfo;

  if (cluster_mode() && key.empty()) {
    errinfo.errcode = ERROR_ZERO_KEY;
    errinfo.raw_errmsg =
        format_string("[%s] key is empty in cluster node", command_args.get_command().c_str());
    errinfo.errmsg =
        format_string("[R3C_CMD][%s:%d] %s", __FILE__, __LINE__, errinfo.raw_errmsg.c_str());
    if (_enable_error_log) (*g_error_log)("%s\n", errinfo.errmsg.c_str());
    THROW_REDIS_EXCEPTION(errinfo);
  }
  for (int loop_counter = 0;; ++loop_counter) {
    const int slot = cluster_mode() ? get_key_slot(&key) : -1;
    CRedisNode* redis_node = get_redis_node(slot, readonly, ask_node, &errinfo);
    HandleResult errcode;

    if (NULL == redis_node) {
      node.first.clear();
      node.second = 0;
    } else {
      node = redis_node->get_node();
    }
    if (which != NULL) {
      *which = node;
    }
    if (0 == loop_counter && _command_monitor != NULL) {
      _command_monitor->before_execute(node, command_args.get_command(), command_args, readonly);
    }
    if (NULL == redis_node) {
      errinfo.errcode = ERROR_NO_ANY_NODE;
      errinfo.raw_errmsg =
          format_string("[%s][%s][%s:%d] no any node", command_args.get_command().c_str(),
                        get_mode_str(), node.first.c_str(), node.second);
      errinfo.errmsg =
          format_string("[R3C_CMD][%s:%d] %s", __FILE__, __LINE__, errinfo.raw_errmsg.c_str());
      if (_enable_error_log) (*g_error_log)("[NO_ANY_NODE] %s\n", errinfo.errmsg.c_str());
      break;
    }
    if (NULL == redis_node->get_redis_context()) {
      errcode = HR_RECONN_UNCOND;
    } else {
      // When a slot is set as MIGRATING, the node will accept all queries that are about this hash
      // slot, but only if the key in question exists, otherwise the query is forwarded using a -ASK
      // redirection to the node that is target of the migration.
      //
      // When a slot is set as IMPORTING, the node will accept all queries that are about this hash
      // slot, but only if the request is preceded by an ASKING command. If the ASKING command was
      // not given by the client, the query is redirected to the real hash slot owner via a -MOVED
      // redirection error, as would happen normally.
      if (ask_node != NULL) (void)redisCommand(redis_node->get_redis_context(), "ASKING");
      redis_reply =
          (redisReply*)redisCommandArgv(redis_node->get_redis_context(), command_args.get_argc(),
                                        command_args.get_argv(), command_args.get_argvlen());
      if (!redis_reply)
        errcode = handle_redis_command_error(redis_node, command_args, &errinfo);
      else
        errcode = handle_redis_reply(redis_node, command_args, redis_reply.get(), &errinfo);
    }

    ask_node = NULL;
    if (HR_SUCCESS == errcode) {
      if (_command_monitor != NULL)
        _command_monitor->after_execute(0, node, command_args.get_command(), redis_reply.get());
      return redis_reply;
    } else if (HR_ERROR == errcode) {
      if (_enable_debug_log) {
        (*g_debug_log)("[NOTRETRY][%s:%d][%s][%s:%d] loop: %d\n", __FILE__, __LINE__,
                       get_mode_str(), redis_node->get_node().first.c_str(),
                       redis_node->get_node().second, loop_counter);
      }
      break;
    } else if (HR_RECONN_COND == errcode || HR_RECONN_UNCOND == errcode) {
      redis_node->close();
    } else if (HR_REDIRECT == errcode) {
      if (!parse_moved_string(redis_reply->str, &node)) {
        if (_enable_error_log) {
          (*g_error_log)("[PARSE_MOVED][%s:%d][%s][%s:%d] node string error: %s\n", __FILE__,
                         __LINE__, get_mode_str(), redis_node->get_node().first.c_str(),
                         redis_node->get_node().second, redis_reply->str);
        }
        break;
      } else {
        ask_node = &node;
        if (loop_counter <= 2) continue;
        if (_enable_debug_log) {
          (*g_debug_log)("[REDIRECT][%s:%d][%s][%s:%d] retries more than %d\n", __FILE__, __LINE__,
                         get_mode_str(), redis_node->get_node().first.c_str(),
                         redis_node->get_node().second, loop_counter);
        }
        break;
      }
    }

    if (HR_RECONN_UNCOND == errcode) {
      if (loop_counter > num_retries && loop_counter > 0) {
        if (_enable_debug_log) {
          (*g_debug_log)("[NOTRECONN][%s:%d][%s][%s:%d] retries more than %d\n", __FILE__, __LINE__,
                         get_mode_str(), redis_node->get_node().first.c_str(),
                         redis_node->get_node().second, num_retries);
        }
        break;
      }
    } else if (HR_RETRY_UNCOND == errcode) {
      if (loop_counter > num_retries) {
        if (_enable_debug_log) {
          (*g_debug_log)("[RETRY_UNCOND][%s:%d][%s][%s:%d] retries more than %d\n", __FILE__,
                         __LINE__, get_mode_str(), redis_node->get_node().first.c_str(),
                         redis_node->get_node().second, num_retries);
        }
        break;
      }
    } else if (loop_counter >= num_retries) {
      if (_enable_debug_log) {
        (*g_debug_log)("[OVERRETRY][%s:%d][%s][%s:%d] retries more than %d\n", __FILE__, __LINE__,
                       get_mode_str(), redis_node->get_node().first.c_str(),
                       redis_node->get_node().second, num_retries);
      }
      break;
    }

    if (HR_RETRY_UNCOND == errcode || HR_RECONN_UNCOND == errcode) {
      const int retry_sleep_milliseconds = get_retry_sleep_milliseconds(loop_counter);
      if (retry_sleep_milliseconds > 0) millisleep(retry_sleep_milliseconds);
    }
    if (cluster_mode() && redis_node->need_refresh_master()) {
      if (HR_RECONN_COND == errcode || HR_RECONN_UNCOND == errcode)
        refresh_master_node_table(&errinfo, &node);
      else
        refresh_master_node_table(&errinfo, NULL);
    }
  }

  if (_command_monitor != NULL)
    _command_monitor->after_execute(1, node, command_args.get_command(), redis_reply.get());
  THROW_REDIS_EXCEPTION_WITH_NODE_AND_COMMAND(errinfo, node.first, node.second,
                                              command_args.get_command(), command_args.get_key());
}

CRedisClient::HandleResult CRedisClient::handle_redis_command_error(CRedisNode* redis_node,
                                                                    const CommandArgs& command_args,
                                                                    struct ErrorInfo* errinfo) {
  redisContext* redis_context = redis_node->get_redis_context();

  // REDIS_ERR_EOF (call read() return 0):
  // redis_context->err(3)
  // redis_context->errstr("Server closed the connection")
  //
  // REDIS_ERR_IO (call read() return -1):
  // redis_context->err(1)
  // redis_context->errstr("Bad file descriptor")
  //
  // REDIS_ERR_IO means there was an I/O error
  // and you should use the "errno" variable to find out what is wrong.
  // For other values, the "errstr" field will hold a description.
  const int redis_errcode = redis_context->err;
  errinfo->errcode = errno;
  errinfo->raw_errmsg =
      format_string("[%s] (%d/%d)%s (%s)", redis_node->str().c_str(), redis_errcode,
                    errinfo->errcode, redis_context->errstr, strerror(errinfo->errcode));
  errinfo->errmsg = format_string("[R3C_CMD_ERROR][%s:%d][%s] %s", __FILE__, __LINE__,
                                  command_args.get_command().c_str(), errinfo->raw_errmsg.c_str());
  if (_enable_error_log) {
    const unsigned int conn_errors = redis_node->get_conn_errors();
    if (conn_errors == 0 || conn_errors % 10 == 0)  // Reduce duplicate error logs
      (*g_error_log)("%s\n", errinfo->errmsg.c_str());
  }

  // REDIS_ERR_IO:
  // There was an I/O error while creating the connection, trying to write
  // to the socket or read from the socket. If you included `errno.h` in your
  // application, you can use the global `errno` variable to find out what is wrong.
  //
  // REDIS_ERR_EOF:
  // The server closed the connection which resulted in an empty read.
  //
  // REDIS_ERR_PROTOCOL:
  // There was an error while parsing the protocol.
  //
  // REDIS_ERR_OTHER:
  // Any other error. Currently, it is only used when a specified hostname to connect to cannot be
  // resolved.
  if ((redis_errcode != REDIS_ERR_IO) && (redis_errcode != REDIS_ERR_EOF)) {
    return HR_ERROR;  // Not retry
  } else if (REDIS_ERR_EOF == redis_errcode) {
    redis_node->inc_conn_errors();
    return HR_RECONN_COND;  // Retry unconditionally and reconnect
  } else {
    // $ redis-cli -p 1386 cluster nodes | grep master
    // 49cadd758538f821b922738fd000b5a16ef64fc7 127.0.0.1:1384@11384 master - 0 1546317629187 14
    // connected 10923-16383 95ce22507d469ae84cb760505bba4be0283c5468 127.0.0.1:1381@11381 master -
    // 0 1546317630190 1 connected 0-5460 a27a1ce7f8c5c5f79a1d09227eb80b73919ec795
    // 127.0.0.1:1383@11383 master,fail - 1546317518930 1546317515000 12 connected
    // 03c4ada274ac137c710f3d514700c2e94649c131 127.0.0.1:1382@11382 master - 0 1546317631191 2
    // connected 5461-10922
    redis_node->inc_conn_errors();
    return HR_RECONN_COND;  // Retry conditionally
  }
}

CRedisClient::HandleResult CRedisClient::handle_redis_reply(CRedisNode* redis_node,
                                                            const CommandArgs& command_args,
                                                            const redisReply* redis_reply,
                                                            struct ErrorInfo* errinfo) {
  redis_node->reset_conn_errors();

  if (redis_reply->type != REDIS_REPLY_ERROR)
    return HR_SUCCESS;
  else
    return handle_redis_replay_error(redis_node, command_args, redis_reply, errinfo);
}

CRedisClient::HandleResult CRedisClient::handle_redis_replay_error(CRedisNode* redis_node,
                                                                   const CommandArgs& command_args,
                                                                   const redisReply* redis_reply,
                                                                   struct ErrorInfo* errinfo) {
  // ASK
  // NOAUTH Authentication required
  // MOVED 6474 127.0.0.1:6380
  // WRONGTYPE Operation against a key holding the wrong kind of value
  // CLUSTERDOWN The cluster is down
  // NOSCRIPT No matching script. Please use EVAL.
  //
  // ERR Error running script (call to f_64b2246244d60088e7351656c216c00cb1d7d2fd) : @user_script:1:
  // @user_script: 1: Lua script attempted to access a non local key in a cluster node
  extract_errtype(redis_reply, &errinfo->errtype);
  errinfo->errcode = ERROR_COMMAND;
  errinfo->raw_errmsg = format_string("[%s] %s", redis_node->str().c_str(), redis_reply->str);
  errinfo->errmsg = format_string("[R3C_REPLAY_ERROR][%s:%d][%s] %s", __FILE__, __LINE__,
                                  command_args.get_command().c_str(), errinfo->raw_errmsg.c_str());
  if (_enable_error_log) (*g_error_log)("%s\n", errinfo->errmsg.c_str());

  if (is_clusterdown_error(errinfo->errtype)) {
    return HR_RETRY_UNCOND;
  } else if (is_ask_error(errinfo->errtype)) {
    // ASK 6474 127.0.0.1:6380
    return HR_REDIRECT;
  } else if (is_moved_error(errinfo->errtype)) {
    // MOVED 6474 127.0.0.1:6380
    //
    // Node ask_node;
    // parse_moved_string(redis_reply->str, &ask_node);
    redis_node->set_conn_errors(2019);  // Trigger to refresh master nodes
    return HR_RETRY_UNCOND;
  } else {
    return HR_ERROR;
  }
}

void CRedisClient::fini() { clear_all_master_nodes(); }

void CRedisClient::init() {
  _enable_debug_log = true;
  _enable_info_log = true;
  _enable_error_log = true;

  try {
    const int num_nodes = parse_nodes(&_nodes, _raw_nodes_string);
    struct ErrorInfo errinfo;

    if (0 == num_nodes) {
      errinfo.errcode = ERROR_PARAMETER;
      errinfo.errmsg = format_string("[R3C_INIT][%s:%d] parameter[nodes] error: %s", __FILE__,
                                     __LINE__, _raw_nodes_string.c_str());
      errinfo.raw_errmsg = format_string("parameter[nodes] error: %s", _raw_nodes_string.c_str());
      if (_enable_error_log) (*g_error_log)("%s\n", errinfo.errmsg.c_str());
      THROW_REDIS_EXCEPTION(errinfo);
    } else if (1 == num_nodes) {
      if (!init_standlone(&errinfo)) THROW_REDIS_EXCEPTION(errinfo);
    } else {
      if (!init_cluster(&errinfo)) THROW_REDIS_EXCEPTION(errinfo);
    }
  } catch (...) {
    clear_all_master_nodes();
    throw;
  }
}

bool CRedisClient::init_standlone(struct ErrorInfo* errinfo) {
  const Node& node = _nodes[0];
  _nodes_string = _raw_nodes_string;

  redisContext* redis_context = connect_redis_node(node, errinfo, false);
  if (NULL == redis_context) {
    return false;
  } else {
    CRedisMasterNode* redis_node = new CRedisMasterNode(std::string(""), node, redis_context);
    const std::pair<RedisMasterNodeTable::iterator, bool> ret =
        _redis_master_nodes.insert(std::make_pair(node, redis_node));
    if (ret.second) R3C_ASSERT(ret.second);
    return true;
  }
}

bool CRedisClient::init_cluster(struct ErrorInfo* errinfo) {
  const int num_nodes = static_cast<int>(_nodes.size());
  const uint64_t base = reinterpret_cast<uint64_t>(this);
  uint64_t seed = get_random_number(base);

  _slot2node.resize(CLUSTER_SLOTS);
  for (int i = 0; i < num_nodes; ++i) {
    const int j = static_cast<int>(++seed % num_nodes);
    const Node& node = _nodes[j];
    std::vector<struct NodeInfo> nodes_info;

    redisContext* redis_context = connect_redis_node(node, errinfo, false);
    if (NULL == redis_context) {
      continue;
    }
    if (!list_cluster_nodes(&nodes_info, errinfo, redis_context, node)) {
      redisFree(redis_context);
      continue;
    } else {
      std::vector<struct NodeInfo> replication_nodes_info;

      redisFree(redis_context);
      redis_context = NULL;
      if (init_master_nodes(nodes_info, &replication_nodes_info, errinfo)) {
        if (_read_policy != RP_ONLY_MASTER) init_replica_nodes(replication_nodes_info);
        break;
      }
    }
  }

  return !_redis_master_nodes.empty();
}

bool CRedisClient::init_master_nodes(const std::vector<struct NodeInfo>& nodes_info,
                                     std::vector<struct NodeInfo>* replication_nodes_info,
                                     struct ErrorInfo* errinfo) {
  int connected = 0;

  if (nodes_info.size() > 1) {
    _nodes_string.clear();
  } else {
    _nodes_string = _raw_nodes_string;
  }
  for (std::vector<struct NodeInfo>::size_type i = 0; i < nodes_info.size();
       ++i)  // Traversing all master nodes
  {
    const struct NodeInfo& nodeinfo = nodes_info[i];

    if (nodes_info.size() > 1) {
      update_nodes_string(nodeinfo);
    }
    if (nodeinfo.is_master() && !nodeinfo.is_fail()) {
      update_slots(nodeinfo);
      if (add_master_node(nodeinfo, errinfo)) ++connected;
    } else if (nodeinfo.is_replica() && !nodeinfo.is_fail()) {
      replication_nodes_info->push_back(nodeinfo);
    }
  }

  return connected > 0;
}

void CRedisClient::init_replica_nodes(const std::vector<struct NodeInfo>& replication_nodes_info) {
  for (std::vector<struct NodeInfo>::size_type i = 0; i < replication_nodes_info.size(); ++i) {
    const struct NodeInfo& nodeinfo = replication_nodes_info[i];
    const NodeId& master_nodeid = nodeinfo.master_id;
    const NodeId& replica_nodeid = nodeinfo.id;
    const Node& replica_node = nodeinfo.node;

    CRedisMasterNode* redis_master_node = get_redis_master_node(master_nodeid);
    if (redis_master_node != NULL) {
      struct ErrorInfo errinfo;
      redisContext* redis_context = connect_redis_node(replica_node, &errinfo, true);
      if (redis_context != NULL) {
        CRedisReplicaNode* redis_replica_node =
            new CRedisReplicaNode(replica_nodeid, replica_node, redis_context);
        redis_master_node->add_replica_node(redis_replica_node);
      }
    }
  }
}

void CRedisClient::update_slots(const struct NodeInfo& nodeinfo) {
  for (SlotSegment::size_type i = 0; i < nodeinfo.slots.size(); ++i) {
    const std::pair<int, int>& slot_segment = nodeinfo.slots[i];
    for (int slot = slot_segment.first; slot <= slot_segment.second; ++slot)
      _slot2node[slot] = nodeinfo.node;
  }
}

void CRedisClient::refresh_master_node_table(struct ErrorInfo* errinfo, const Node* error_node) {
  const int num_nodes = static_cast<int>(_redis_master_nodes.size());
  uint64_t seed = reinterpret_cast<uint64_t>(this) - num_nodes;
  const int k = static_cast<int>(seed % num_nodes);
  RedisMasterNodeTable::iterator iter = _redis_master_nodes.begin();

  for (int i = 0; i < k; ++i) {
    ++iter;
    if (iter == _redis_master_nodes.end()) iter = _redis_master_nodes.begin();
  }
  for (int i = 0; i < num_nodes; ++i) {
    const Node& node = iter->first;
    CRedisMasterNode* redis_node = iter->second;
    ++iter;

    if ((NULL == error_node) || (node != *error_node)) {
      redisContext* redis_context = redis_node->get_redis_context();

      if (NULL == redis_context) {
        redis_context = connect_redis_node(node, errinfo, false);
      }
      if (redis_context != NULL) {
        std::vector<struct NodeInfo> nodes_info;

        redis_node->set_redis_context(redis_context);
        if (list_cluster_nodes(&nodes_info, errinfo, redis_context, node)) {
          std::vector<struct NodeInfo> replication_nodes_info;
          clear_and_update_master_nodes(nodes_info, &replication_nodes_info, errinfo);
          if (_read_policy != RP_ONLY_MASTER) init_replica_nodes(replication_nodes_info);
          break;  // Continue is not safe, because `clear_and_update_master_nodes` will modify
                  // _redis_master_nodes
        }
      }
    }
    if (iter == _redis_master_nodes.end()) {
      iter = _redis_master_nodes.begin();
    }
  }
}

void CRedisClient::clear_and_update_master_nodes(
    const std::vector<struct NodeInfo>& nodes_info,
    std::vector<struct NodeInfo>* replication_nodes_info, struct ErrorInfo* errinfo) {
  NodeInfoTable master_nodeinfo_table;
  std::string nodes_string;

  if (nodes_info.size() > 1) {
    _nodes_string.clear();
  }
  for (std::vector<struct NodeInfo>::size_type i = 0; i < nodes_info.size();
       ++i)  // Traversing all master nodes
  {
    const struct NodeInfo& nodeinfo = nodes_info[i];

    if (nodes_info.size() > 1) {
      update_nodes_string(nodeinfo);
    }
    if (nodeinfo.is_master() && !nodeinfo.is_fail()) {
      update_slots(nodeinfo);
      master_nodeinfo_table.insert(std::make_pair(nodeinfo.node, nodeinfo));

      if (_redis_master_nodes.count(nodeinfo.node) <= 0) {
        // New master
        add_master_node(nodeinfo, errinfo);
      }
    } else if (nodeinfo.is_replica() && !nodeinfo.is_fail()) {
      replication_nodes_info->push_back(nodeinfo);
    }
  }

  clear_invalid_master_nodes(master_nodeinfo_table);
}

void CRedisClient::clear_invalid_master_nodes(const NodeInfoTable& master_nodeinfo_table) {
  for (RedisMasterNodeTable::iterator node_iter = _redis_master_nodes.begin();
       node_iter != _redis_master_nodes.end();) {
    const Node& node = node_iter->first;
    const NodeInfoTable::const_iterator info_iter = master_nodeinfo_table.find(node);

    if (info_iter != master_nodeinfo_table.end()) {
      ++node_iter;
    } else {
      CRedisMasterNode* master_node = node_iter->second;
      const NodeId nodeid = master_node->get_nodeid();
      if (_enable_info_log)
        (*g_info_log)("[R3C_CLEAR_INVALID][%s:%d] %s is removed because it is not a master now\n",
                      __FILE__, __LINE__, master_node->str().c_str());

#if __cplusplus < 201103L
      _redis_master_nodes.erase(node_iter++);
#else
      node_iter = _redis_master_nodes.erase(node_iter);
#endif
      delete master_node;
      _redis_master_nodes_id.erase(nodeid);
    }
  }
}

bool CRedisClient::add_master_node(const NodeInfo& nodeinfo, struct ErrorInfo* errinfo) {
  const NodeId& nodeid = nodeinfo.id;
  const Node& node = nodeinfo.node;
  redisContext* redis_context = connect_redis_node(node, errinfo, false);
  CRedisMasterNode* master_node = new CRedisMasterNode(nodeid, node, redis_context);

  const std::pair<RedisMasterNodeTable::iterator, bool> ret =
      _redis_master_nodes.insert(std::make_pair(node, master_node));
  R3C_ASSERT(ret.second);
  if (!ret.second) delete master_node;
  _redis_master_nodes_id[nodeid] = node;
  return redis_context != NULL;
}

void CRedisClient::clear_all_master_nodes() {
  for (RedisMasterNodeTable::iterator iter = _redis_master_nodes.begin();
       iter != _redis_master_nodes.end(); ++iter) {
    CRedisMasterNode* master_node = iter->second;
    delete master_node;
  }
  _redis_master_nodes.clear();
  _redis_master_nodes_id.clear();
}

void CRedisClient::update_nodes_string(const NodeInfo& nodeinfo) {
  std::string node_str;
  if (_nodes_string.empty())
    _nodes_string = node2string(nodeinfo.node, &node_str);
  else
    _nodes_string = _nodes_string + std::string(",") + node2string(nodeinfo.node, &node_str);
}

redisContext* CRedisClient::connect_redis_node(const Node& node, struct ErrorInfo* errinfo,
                                               bool readonly) const {
  redisContext* redis_context = NULL;

  errinfo->clear();
  if (_enable_debug_log) {
    (*g_debug_log)("[R3C_CONN][%s:%d] To connect %s with timeout: %dms\n", __FILE__, __LINE__,
                   node2string(node).c_str(), _connect_timeout_milliseconds);
  }
  if (_connect_timeout_milliseconds <= 0) {
    redis_context = redisConnect(node.first.c_str(), node.second);
  } else {
    struct timeval timeout;
    timeout.tv_sec = _connect_timeout_milliseconds / 1000;
    timeout.tv_usec = (_connect_timeout_milliseconds % 1000) * 1000;
    redis_context = redisConnectWithTimeout(node.first.c_str(), node.second, timeout);
  }

  if (NULL == redis_context) {
    // can't allocate redis context
    errinfo->errcode = ERROR_REDIS_CONTEXT;
    errinfo->raw_errmsg = "can not allocate redis context";
    errinfo->errmsg = format_string("[R3C_CONN][%s:%d][%s:%d] %s", __FILE__, __LINE__,
                                    node.first.c_str(), node.second, errinfo->raw_errmsg.c_str());
    if (_enable_error_log) (*g_error_log)("%s\n", errinfo->errmsg.c_str());
  } else if (redis_context->err != 0) {
    errinfo->errcode = ERROR_INIT_REDIS_CONN;
    errinfo->raw_errmsg = redis_context->errstr;
    if (REDIS_ERR_IO == redis_context->err) {
      errinfo->errmsg = format_string("[R3C_CONN][%s:%d][%s:%d] (errno:%d,err:%d)%s", __FILE__,
                                      __LINE__, node.first.c_str(), node.second, errno,
                                      redis_context->err, errinfo->raw_errmsg.c_str());
    } else {
      errinfo->errmsg = format_string("[R3C_CONN][%s:%d][%s:%d] (err:%d)%s", __FILE__, __LINE__,
                                      node.first.c_str(), node.second, redis_context->err,
                                      errinfo->raw_errmsg.c_str());
    }
    if (_enable_error_log) {
      (*g_error_log)("%s\n", errinfo->errmsg.c_str());
    }
    redisFree(redis_context);
    redis_context = NULL;
  } else {
    if (_enable_debug_log) {
      (*g_debug_log)("[R3C_CONN][%s:%d] Connect %s successfully with readwrite timeout: %dms\n",
                     __FILE__, __LINE__, node2string(node).c_str(),
                     _readwrite_timeout_milliseconds);
    }
    if (_readwrite_timeout_milliseconds > 0) {
      struct timeval data_timeout;
      data_timeout.tv_sec = _readwrite_timeout_milliseconds / 1000;
      data_timeout.tv_usec = (_readwrite_timeout_milliseconds % 1000) * 1000;

      if (REDIS_ERR == redisSetTimeout(redis_context, data_timeout)) {
        // REDIS_ERR_IO == redis_context->err
        errinfo->errcode = ERROR_INIT_REDIS_CONN;
        errinfo->raw_errmsg = redis_context->errstr;
        errinfo->errmsg = format_string("[R3C_CONN][%s:%d][%s:%d] (errno:%d,err:%d)%s", __FILE__,
                                        __LINE__, node.first.c_str(), node.second, errno,
                                        redis_context->err, errinfo->raw_errmsg.c_str());
        if (_enable_error_log) (*g_error_log)("%s\n", errinfo->errmsg.c_str());
        redisFree(redis_context);
        redis_context = NULL;
      }
    }
    if ((0 == errinfo->errcode) && !_password.empty()) {
      const RedisReplyHelper redis_reply =
          (redisReply*)redisCommand(redis_context, "AUTH %s", _password.c_str());

      if (redis_reply && 0 == strcmp(redis_reply->str, "OK")) {
        // AUTH success
        if (_enable_info_log)
          (*g_info_log)("[R3C_AUTH][%s:%d] Connect redis://%s:%d success\n", __FILE__, __LINE__,
                        node.first.c_str(), node.second);
      } else {
        // AUTH failed
        if (!redis_reply) {
          errinfo->errcode = ERROR_REDIS_AUTH;
          errinfo->raw_errmsg = "authorization failed";
        } else {
          extract_errtype(redis_reply.get(), &errinfo->errtype);
          errinfo->errcode = ERROR_REDIS_AUTH;
          errinfo->raw_errmsg = redis_reply->str;
        }

        errinfo->errmsg =
            format_string("[R3C_AUTH][%s:%d][%s:%d] %s", __FILE__, __LINE__, node.first.c_str(),
                          node.second, errinfo->raw_errmsg.c_str());
        if (_enable_error_log) (*g_error_log)("%s\n", errinfo->errmsg.c_str());
        redisFree(redis_context);
        redis_context = NULL;
      }
    }
  }
  if (readonly && redis_context != NULL) {
    const RedisReplyHelper redis_reply = (redisReply*)redisCommand(redis_context, "READONLY");

    if (redis_reply && 0 == strcmp(redis_reply->str, "OK")) {
      // READONLY success
      if (_enable_info_log)
        (*g_debug_log)("[R3C_READONLY][%s:%d] READONLY redis://%s:%d success\n", __FILE__, __LINE__,
                       node.first.c_str(), node.second);
    } else {
      // READONLY failed
      if (!redis_reply) {
        errinfo->errcode = ERROR_REDIS_READONLY;
        errinfo->raw_errmsg = "readonly failed";
      } else {
        extract_errtype(redis_reply.get(), &errinfo->errtype);
        errinfo->errcode = ERROR_REDIS_READONLY;
        errinfo->raw_errmsg = redis_reply->str;
      }

      errinfo->errmsg = format_string("[R3C_READONLY][%s:%d][%s:%d] %s", __FILE__, __LINE__,
                                      node.first.c_str(), node.second, errinfo->raw_errmsg.c_str());
      if (_enable_error_log) (*g_error_log)("%s\n", errinfo->errmsg.c_str());
      redisFree(redis_context);
      redis_context = NULL;
    }
  }

  return redis_context;
}

CRedisNode* CRedisClient::get_redis_node(int slot, bool readonly, const Node* ask_node,
                                         struct ErrorInfo* errinfo) {
  CRedisNode* redis_node = NULL;
  redisContext* redis_context = NULL;

  do {
    if (-1 == slot) {
      // Standalone
      R3C_ASSERT(!_redis_master_nodes.empty());
      redis_node = _redis_master_nodes.begin()->second;
      redis_context = redis_node->get_redis_context();
      if (NULL == redis_context) {
        redis_context = connect_redis_node(redis_node->get_node(), errinfo, false);
        redis_node->set_redis_context(redis_context);
      }
      break;
    } else {
      // Cluster
      R3C_ASSERT(slot >= 0 && slot < CLUSTER_SLOTS);

      // clear_invalid_master_nodes
      if (_redis_master_nodes.empty()) {
        const int num_nodes = parse_nodes(&_nodes, _nodes_string);
        if (num_nodes > 1) R3C_ASSERT(num_nodes > 1);
        if (!init_cluster(errinfo)) {
          break;
        }
      }
      {
        const Node& node = (NULL == ask_node) ? _slot2node[slot] : *ask_node;
        const RedisMasterNodeTable::const_iterator iter = _redis_master_nodes.find(node);
        if (iter != _redis_master_nodes.end()) {
          redis_node = iter->second;
        }
      }
    }

    if (NULL == redis_node) {
      redis_node = random_redis_master_node();
    }
    if (redis_node != NULL) {
      redis_context = redis_node->get_redis_context();

      if (NULL == redis_context) {
        redis_context = connect_redis_node(redis_node->get_node(), errinfo, false);
        redis_node->set_redis_context(redis_context);
      }
      if (!readonly || RP_ONLY_MASTER == _read_policy) {
        break;
      }
      if (redis_context != NULL && RP_PRIORITY_MASTER == _read_policy) {
        break;
      }

      CRedisMasterNode* redis_master_node = (CRedisMasterNode*)redis_node;
      redis_node = redis_master_node->choose_node(_read_policy);
      redis_context = redis_node->get_redis_context();
      if (NULL == redis_context) {
        redis_context = connect_redis_node(redis_node->get_node(), errinfo, false);
        redis_node->set_redis_context(redis_context);
      }
      if (NULL == redis_context) {
        redis_node = redis_master_node;
      }
    }
  } while (false);

  return redis_node;
}

CRedisMasterNode* CRedisClient::get_redis_master_node(const NodeId& nodeid) const {
  CRedisMasterNode* redis_master_node = NULL;
  RedisMasterNodeIdTable::const_iterator nodeid_iter = _redis_master_nodes_id.find(nodeid);
  if (nodeid_iter != _redis_master_nodes_id.end()) {
    const Node& node = nodeid_iter->second;
    RedisMasterNodeTable::const_iterator node_iter = _redis_master_nodes.find(node);
    if (node_iter != _redis_master_nodes.end()) redis_master_node = node_iter->second;
  }
  return redis_master_node;
}

CRedisMasterNode* CRedisClient::random_redis_master_node() const {
  if (_redis_master_nodes.empty()) {
    return NULL;
  } else {
    const int num_nodes = static_cast<int>(_nodes.size());
    const uint64_t base = reinterpret_cast<uint64_t>(this);
    const uint64_t seed = get_random_number(base);
    const int K = static_cast<int>(seed % num_nodes);

    RedisMasterNodeTable::const_iterator iter = _redis_master_nodes.begin();
    for (int i = 0; i < K; ++i) {
      ++iter;
      if (iter == _redis_master_nodes.end()) iter = _redis_master_nodes.begin();
    }
    return iter->second;
  }
}

bool CRedisClient::list_cluster_nodes(std::vector<struct NodeInfo>* nodes_info,
                                      struct ErrorInfo* errinfo, redisContext* redis_context,
                                      const Node& node) {
  const RedisReplyHelper redis_reply = (redisReply*)redisCommand(redis_context, "CLUSTER NODES");

  errinfo->clear();
  if (!redis_reply) {
    const int sys_errcode = errno;
    const int redis_errcode = redis_context->err;
    errinfo->errcode = ERROR_COMMAND;
    if (redis_errcode != 0)
      errinfo->raw_errmsg = redis_context->errstr;
    else
      errinfo->raw_errmsg = "redisCommand failed";
    errinfo->errmsg = format_string("[R3C_LIST_NODES][%s:%d][NODE:%s] (sys:%d,redis:%d)%s",
                                    __FILE__, __LINE__, node2string(node).c_str(), sys_errcode,
                                    redis_errcode, errinfo->raw_errmsg.c_str());
    if (_enable_error_log) (*g_error_log)("%s\n", errinfo->errmsg.c_str());
    if (_enable_info_log) (*g_info_log)("[%s:%d] %s\n", __FILE__, __LINE__, _nodes_string.c_str());
  } else if (REDIS_REPLY_ERROR == redis_reply->type) {
    // ERR This instance has cluster support disabled
    errinfo->errcode = ERROR_COMMAND;
    errinfo->raw_errmsg = redis_reply->str;
    errinfo->errmsg = format_string("[R3C_LIST_NODES][%s:%d][NODE:%s] %s", __FILE__, __LINE__,
                                    node2string(node).c_str(), redis_reply->str);
    if (_enable_error_log) (*g_error_log)("%s\n", errinfo->errmsg.c_str());
  } else if (redis_reply->type != REDIS_REPLY_STRING) {
    // Unexpected reply type
    errinfo->errcode = ERROR_UNEXCEPTED_REPLY_TYPE;
    errinfo->raw_errmsg = redis_reply->str;
    errinfo->errmsg =
        format_string("[R3C_LIST_NODES][%s:%d][NODE:%s] (type:%d)%s", __FILE__, __LINE__,
                      node2string(node).c_str(), redis_reply->type, redis_reply->str);
    if (_enable_error_log) (*g_error_log)("%s\n", errinfo->errmsg.c_str());
  } else {
    std::vector<std::string> lines;
    const int num_lines = split(&lines, std::string(redis_reply->str), std::string("\n"));

    if (num_lines < 1) {
      errinfo->errcode = ERROR_REPLY_FORMAT;
      errinfo->raw_errmsg = "reply nothing";
      errinfo->errmsg =
          format_string("[R3C_LIST_NODES][%s:%d][NODE:%s][REPLY:%s] %s", __FILE__, __LINE__,
                        node2string(node).c_str(), redis_reply->str, "reply nothing");
      if (_enable_error_log) (*g_error_log)("%s\n", errinfo->errmsg.c_str());
    }
    for (int row = 0; row < num_lines; ++row) {
      std::vector<std::string> tokens;
      const std::string& line = lines[row];
      const int num_tokens = split(&tokens, line, std::string(" "));

      if (0 == num_tokens) {
        // Over
        errinfo->clear();
        break;
      } else if (num_tokens < 8) {
        nodes_info->clear();
        errinfo->errcode = ERROR_REPLY_FORMAT;
        errinfo->raw_errmsg = "reply format error";
        errinfo->errmsg =
            format_string("[R3C_LIST_NODES][%s:%d][NODE:%s][LINE:%s] %s", __FILE__, __LINE__,
                          node2string(node).c_str(), line.c_str(), "reply format error");
        if (_enable_error_log) (*g_error_log)("%s\n", errinfo->errmsg.c_str());
        break;
      } else {
        NodeInfo nodeinfo;
        nodeinfo.id = tokens[0];

        if (!parse_node_string(tokens[1], &nodeinfo.node.first, &nodeinfo.node.second)) {
          nodes_info->clear();
          errinfo->errcode = ERROR_REPLY_FORMAT;
          errinfo->raw_errmsg = "reply format error";
          errinfo->errmsg = format_string("[R3C_LIST_NODES][%s:%d][NODE:%s][TOKEN:%s][LINE:%s] %s",
                                          __FILE__, __LINE__, node2string(node).c_str(),
                                          tokens[1].c_str(), line.c_str(), "reply format error");
          if (_enable_error_log) (*g_error_log)("%s\n", errinfo->errmsg.c_str());
          break;
        } else {
          nodeinfo.flags = tokens[2];
          nodeinfo.master_id = tokens[3];
          nodeinfo.ping_sent = atoi(tokens[4].c_str());
          nodeinfo.pong_recv = atoi(tokens[5].c_str());
          nodeinfo.epoch = atoi(tokens[6].c_str());
          nodeinfo.connected = (tokens[7] == "connected");
          if (nodeinfo.is_master() && !nodeinfo.is_fail()) {
            for (int col = 8; col < num_tokens; ++col) {
              const std::string& token = tokens[col];

              // [14148->-ec19be9a50b5416999ac0305c744d9b6c957c18d]
              if (token[0] != '[') {
                std::pair<int, int> slot;
                parse_slot_string(token, &slot.first, &slot.second);
                nodeinfo.slots.push_back(slot);
              }
            }
          }

          if (_enable_debug_log)
            (*g_debug_log)("[R3C_LIST_NODES][%s:%d][NODE:%s] %s\n", __FILE__, __LINE__,
                           node2string(node).c_str(), nodeinfo.str().c_str());
          nodes_info->push_back(nodeinfo);
        }
      }
    }
  }

  return !nodes_info->empty();
}

// Extract error type, such as ERR, MOVED, WRONGTYPE, ...
void CRedisClient::extract_errtype(const redisReply* redis_reply, std::string* errtype) const {
  if (redis_reply->len > 2) {
    const char* space_pos = strchr(redis_reply->str, ' ');

    if (space_pos != NULL) {
      const size_t len = static_cast<size_t>(space_pos - redis_reply->str);

      if (len > 2) {
        if (isupper(redis_reply->str[0]) && isupper(redis_reply->str[1]) &&
            isupper(redis_reply->str[2])) {
          errtype->assign(redis_reply->str, len);
        }
      }
    }
  }
}

bool CRedisClient::get_value(const redisReply* redis_reply, std::string* value) {
  value->clear();

  if (REDIS_REPLY_NIL == redis_reply->type) {
    return false;
  } else {
    if (redis_reply->len > 0)
      value->assign(redis_reply->str, redis_reply->len);
    else
      value->clear();
    return true;
  }
}

int CRedisClient::get_values(const redisReply* redis_reply, std::vector<std::string>* values) {
  values->clear();

  if (redis_reply->elements > 0) {
    values->resize(redis_reply->elements);
    for (size_t i = 0; i < redis_reply->elements; ++i) {
      const struct redisReply* value_reply = redis_reply->element[i];
      std::string& value = (*values)[i];

      if (value_reply->type != REDIS_REPLY_NIL) {
        value.assign(value_reply->str, value_reply->len);
      }
    }
  }
  return static_cast<int>(redis_reply->elements);
}

}  // namespace r3c
