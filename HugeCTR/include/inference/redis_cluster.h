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

// Copy and modify part code from R3C,which is a C++ open source client for redis based on hiredis (https://github.com/redis/hiredis)
#ifndef REDIS_CLUSTER_CLIENT_H
#define REDIS_CLUSTER_CLIENT_H
#include <hiredis/hiredis.h>
#include <inttypes.h>
#include <stdint.h>
#include <map>
#include <set>
#include <string>
#include <vector>
#if __cplusplus < 201103L
#   include <tr1/unordered_map>
#else
#   include <unordered_map>
#endif // __cplusplus < 201103L

#define R3C_VERSION 0x000020
#define R3C_MAJOR 0x00
#define R3C_MINOR 0x00
#define R3C_PATCH 0x20

namespace r3c {

extern int NUM_RETRIES /*=15*/; // The default number of retries is 15 (CLUSTERDOWN cost more than 6s)
extern int CONNECT_TIMEOUT_MILLISECONDS /*=2000*/; // Connection timeout in milliseconds
extern int READWRITE_TIMEOUT_MILLISECONDS /*=2000*/; // Receive and send timeout in milliseconds

enum ReadPolicy
{
    RP_ONLY_MASTER, // Always read from master
    RP_PRIORITY_MASTER,
    RP_PRIORITY_REPLICA,
    RP_READ_REPLICA
};


////////////////////////////////////////////////////////////////////////////////

typedef std::pair<std::string, uint16_t> Node; // first is IP, second is port
typedef std::string NodeId;
typedef std::vector<std::pair<int, int> > SlotSegment; // first is begin slot, second is end slot

struct NodeHasher
{
    size_t operator ()(const Node& node) const;
};

std::string& node2string(const Node& node, std::string* str);
std::string node2string(const Node& node);

struct NodeInfo
{
    Node node;
    std::string id;         // The node ID, a 40 characters random string generated when a node is created and never changed again (unless CLUSTER RESET HARD is used)
    std::string flags;      // A list of comma separated flags: myself, master, slave, fail?, fail, handshake, noaddr, noflags
    std::string master_id;  // The replication master
    int ping_sent;          // Milliseconds unix time at which the currently active ping was sent, or zero if there are no pending pings
    int pong_recv;          // Milliseconds unix time the last pong was received
    int epoch;              // The configuration epoch (or version) of the current node (or of the current master if the node is a slave). Each time there is a failover, a new, unique, monotonically increasing configuration epoch is created. If multiple nodes claim to serve the same hash slots, the one with higher configuration epoch wins
    bool connected;         // The state of the link used for the node-to-node cluster bus
    SlotSegment slots;      // A hash slot number or range

    std::string str() const;
    bool is_master() const;
    bool is_replica() const;
    bool is_fail() const;
};

#if __cplusplus < 201103L
    typedef std::tr1::unordered_map<Node, NodeInfo, NodeHasher> NodeInfoTable;
#else
    typedef std::unordered_map<Node, NodeInfo, NodeHasher> NodeInfoTable;
#endif // __cplusplus < 201103L

extern std::ostream& operator <<(std::ostream& os, const struct NodeInfo& nodeinfo);

// The helper for freeing redisReply automatically
// DO NOT use RedisReplyHelper for any nested redisReply
class RedisReplyHelper
{
public:
    RedisReplyHelper();
    RedisReplyHelper(const redisReply* redis_reply);
    RedisReplyHelper(const RedisReplyHelper& other);
    ~RedisReplyHelper();
    operator bool() const;
    void free();
    const redisReply* get() const;
    const redisReply* detach() const;
    RedisReplyHelper& operator =(const redisReply* redis_reply);
    RedisReplyHelper& operator =(const RedisReplyHelper& other);
    const redisReply* operator ->() const;
    std::ostream& operator <<(std::ostream& os);

private:
    mutable const redisReply* _redis_reply;
};

bool is_general_error(const std::string& errtype);
bool is_ask_error(const std::string& errtype);
bool is_clusterdown_error(const std::string& errtype);
bool is_moved_error(const std::string& errtype);
bool is_noauth_error(const std::string& errtype);
bool is_noscript_error(const std::string& errtype);
bool is_wrongtype_error(const std::string& errtype);
bool is_busygroup_error(const std::string& errtype);
bool is_nogroup_error(const std::string& errtype);
bool is_crossslot_error(const std::string& errtype);


struct FVPair;
struct SlotInfo;
class CRedisNode;
class CRedisMasterNode;
class CRedisReplicaNode;
class CommandMonitor;

// Redis Command args
class CommandArgs
{
public:
    CommandArgs();
    ~CommandArgs();
    void set_key(const std::string& key);
    void set_command(const std::string& command);

public:
    void add_arg(const std::string& arg);
    void add_arg(int32_t arg);
    void add_arg(uint32_t arg);
    void add_arg(int64_t arg);
    void add_args(const std::vector<std::string>& args);
    void add_args(const std::vector<std::pair<std::string, std::string> >& values);
    void add_args(const std::map<std::string, std::string>& map);
    void add_args(const std::map<std::string, int64_t>& map, bool reverse);
    void add_args(const std::vector<FVPair>& fvpairs);
    void final();

public:
    int get_argc() const;
    const char** get_argv() const;
    const size_t* get_argvlen() const;
    const std::string& get_command() const;
    const std::string& get_key() const;

private:
    std::string _key;
    std::string _command;

private:
    std::vector<std::string> _args;
    int _argc;
    char** _argv;
    size_t* _argvlen;
};

struct ErrorInfo
{
    std::string raw_errmsg;
    std::string errmsg;
    std::string errtype; // The type of error, such as: ERR, MOVED, WRONGTYPE, ...
    int errcode;

    ErrorInfo();
    ErrorInfo(const std::string& raw_errmsg_, const std::string& errmsg_, const std::string& errtype_, int errcode_);
    void clear();
};

class CRedisException: public std::exception
{
public:
    // key maybe a binary value
    CRedisException(
            const struct ErrorInfo& errinfo,
            const char* file, int line,
            const std::string& node_ip=std::string("-"), uint16_t node_port=0,
            const std::string& command=std::string(""), const std::string& key=std::string("")) throw ();
    virtual ~CRedisException() throw () {}
    virtual const char* what() const throw ();
    int errcode() const { return _errinfo.errcode; }
    std::string str() const throw ();

public:
    const char* file() const throw () { return _file.c_str(); }
    int line() const throw () { return _line; }
    const char* node_ip() const throw () { return _node_ip.c_str(); }
    uint16_t node_port() const throw () { return _node_port; }
    const std::string& command() const throw() { return _command; }
    const std::string& key() const throw() { return _key; }
    const std::string& errtype() const throw () { return _errinfo.errtype; }
    const std::string& raw_errmsg() const throw () { return _errinfo.raw_errmsg; }
    const ErrorInfo& get_errinfo() const throw() { return _errinfo; }

private:
    const ErrorInfo _errinfo;
    const std::string _file;
    const int _line;
    const std::string _node_ip;
    const uint16_t _node_port;

private:
    std::string _command;
    std::string _key;
};

// FVPair: field-value pair
struct FVPair
{
    std::string field;
    std::string value;
};

// Entry uniquely identified by a id
struct StreamEntry
{
    std::string id; // Stream entry ID (milliseconds-sequence)
    std::vector<struct FVPair> fvpairs; // field-value pairs
};

// Stream uniquely identified by a key
struct Stream
{
    std::string key;
    std::vector<struct StreamEntry> entries;
};

std::ostream& operator <<(std::ostream& os, const std::vector<struct Stream>& streams);
std::ostream& operator <<(std::ostream& os, const std::vector<struct StreamEntry>& entries);
// Returns the number of IDs
int extract_ids(const std::vector<struct StreamEntry>& entries, std::vector<std::string>* ids);

struct ConsumerPending
{
    std::string name; // Consumer name
    int count; // Number of pending messages consumer has
};

struct GroupPending
{
    int count; // The total number of pending messages for this consumer group
    std::string start; // The smallest ID among the pending messages
    std::string end; // The greatest ID among the pending messages
    std::vector<struct ConsumerPending> consumers; // All consumers in the group with at least one pending message
};

// detailed information for a message in the pending entries list
struct DetailedPending
{
    std::string id; // The ID of the message (milliseconds-sequence)
    std::string consumer; // The name of the consumer that fetched the message and has still to acknowledge it. We call it the current owner of the message..
    int64_t elapsed; // Number of milliseconds that elapsed since the last time this message was delivered to this consumer.
    int64_t delivered; // Number of times this message was delivered
};

struct ConsumerInfo
{
    std::string name; // Consumer name
    int pendings; // Number of pending messages for this specific consumer
    int64_t idletime; // The idle time in milliseconds
};

struct GroupInfo
{
    std::string name; // Group name
    std::string last_delivered_id;
    int consumers; // Number of consumers known in that group
    int pendings; // Number of pending messages (delivered but not yet acknowledged) in that group
};

struct StreamInfo
{
    int entries; // Number of entries inside this stream
    int radix_tree_keys;
    int radix_tree_nodes;
    int groups; // Number of consumer groups associated with the stream
    std::string last_generated_id; // The last generated ID that may not be the same as the last entry ID in case some entry was deleted
    struct StreamEntry first_entry;
    struct StreamEntry last_entry;
};
std::ostream& operator <<(std::ostream& os, const struct StreamInfo& streaminfo);

// NOTICE:
// 1) ALL keys and values can be binary except EVAL commands.
class CRedisClient
{
public:
    // raw_nodes_string - Redis cluster nodes separated by comma,
    //                    EXAMPLE: 127.0.0.1:6379,127.0.0.1:6380,127.0.0.2:6379,127.0.0.3:6379,
    //                    standalone mode if only one node, else cluster mode.
    //
    // Particularly same nodes are allowed for cluster mode:
    // const std::string nodes = "127.0.0.1:6379,127.0.0.1:6379";
    CRedisClient(
            const std::string& raw_nodes_string,
            int connect_timeout_milliseconds=CONNECT_TIMEOUT_MILLISECONDS,
            int readwrite_timeout_milliseconds=READWRITE_TIMEOUT_MILLISECONDS,
            const std::string& password=std::string(""),
            ReadPolicy read_policy=RP_ONLY_MASTER
            );
    CRedisClient(
            const std::string& raw_nodes_string,
            const std::string& password,
            int connect_timeout_milliseconds=CONNECT_TIMEOUT_MILLISECONDS,
            int readwrite_timeout_milliseconds=READWRITE_TIMEOUT_MILLISECONDS,
            ReadPolicy read_policy=RP_ONLY_MASTER
            );
    ~CRedisClient();
    const std::string& get_raw_nodes_string() const;
    const std::string& get_nodes_string() const;
    std::string str() const;

    // Returns true if parameter nodes of ctor is composed of two or more nodes,
    // or false when only a node for standlone mode.
    bool cluster_mode() const;
    const char* get_mode_str() const;

public: // Control logs
    void enable_debug_log();
    void disable_debug_log();
    void enable_info_log();
    void disable_info_log();
    void enable_error_log();
    void disable_error_log();

public:
    int list_nodes(std::vector<struct NodeInfo>* nodes_info);

    // NOT SUPPORT CLUSTER
    //
    // Remove all keys from all databases.
    //
    // The time-complexity for this operation is O(N), N being the number of keys in all existing databases.
    void flushall();

    // NOT SUPPORT cluster mode
    void multi(const std::string& key=std::string(""), Node* which=NULL);

    // NOT SUPPORT cluster mode
    const RedisReplyHelper exec(const std::string& key=std::string(""), Node* which=NULL);

public: // KV
    // Set a key's time to live in seconds.
    // Time complexity: O(1)
    //
    // Returns true if the timeout was set, or false when key does not exist.
    bool expire(const std::string& key, uint32_t seconds, Node* which=NULL, int num_retries=NUM_RETRIES);

    // Determine if a key exists.
    // Time complexity: O(1)
    // Returns true if the key exists, or false when the key does not exist.
    bool exists(const std::string& key, Node* which=NULL, int num_retries=NUM_RETRIES);

    // Time complexity:
    // O(N) where N is the number of keys that will be removed.
    // When a key to remove holds a value other than a string,
    // the individual complexity for this key is O(M) where M is the number of elements in the list, set, sorted set or hash.
    // Removing a single key that holds a string value is O(1).
    //
    // Returns true, or false when key does not exist.
    bool del(const std::string& key, Node* which=NULL, int num_retries=NUM_RETRIES);

    // Get the value of a key
    // Time complexity: O(1)
    // Returns false if key does not exist.
    bool get(const std::string& key, std::string* value, Node* which=NULL, int num_retries=NUM_RETRIES);

    // Set the string value of a key.
    // Time complexity: O(1)
    void set(const std::string& key, const std::string& value, Node* which=NULL, int num_retries=NUM_RETRIES);

    
    // Get the values of all the given keys.
    //
    // Time complexity:
    // O(N) where N is the number of keys to retrieve.
    //
    // For every key that does not hold a string value or does not exist, the value will be empty string value.
    //
    // Returns the number of values.
    int mget(const std::vector<std::string>& keys, std::vector<std::string>* values, Node* which=NULL, int num_retries=NUM_RETRIES);

    // Set multiple keys to multiple values.
    //
    // In cluster mode, mset guaranteed atomicity, or may partially success.
    //
    // Time complexity:
    // O(N) where N is the number of keys to set.
    int mset(const std::map<std::string, std::string>& kv_map, Node* which=NULL, int num_retries=NUM_RETRIES);

    
    // Determine the type stored at key.
    // Time complexity: O(1)
    bool key_type(const std::string& key, std::string* key_type, Node* which=NULL, int num_retries=NUM_RETRIES);

    // Get the time to live for a key
    //
    // Time complexity: O(1)
    // Returns the remaining time to live of a key that has a timeout in seconds.
    // Returns -2 if the key does not exist.
    // Returns -1 if the key exists but has no associated expire.
    int64_t ttl(const std::string& key, Node* which=NULL, int num_retries=NUM_RETRIES);

    
public:
    // Standlone: key should be empty
    // Cluse mode: key used to locate node
    const RedisReplyHelper redis_command(
            bool readonly, int num_retries,
            const std::string& key, const CommandArgs& command_args,
            Node* which);

private:
    enum HandleResult { HR_SUCCESS, HR_ERROR, HR_RETRY_COND, HR_RETRY_UNCOND, HR_RECONN_COND, HR_RECONN_UNCOND, HR_REDIRECT };

    // Handle the redis command error
    // Return -1 to break, return 1 to retry conditionally
    HandleResult handle_redis_command_error(CRedisNode* redis_node, const CommandArgs& command_args, struct ErrorInfo* errinfo);

    // Handle the redis reply
    // Success returns 0,
    // return -1 to break, return 2 to retry unconditionally
    HandleResult handle_redis_reply(CRedisNode* redis_node, const CommandArgs& command_args, const redisReply* redis_reply, struct ErrorInfo* errinfo);
    HandleResult handle_redis_replay_error(CRedisNode* redis_node, const CommandArgs& command_args, const redisReply* redis_reply, struct ErrorInfo* errinfo);

private:
    void fini();
    void init();
    bool init_standlone(struct ErrorInfo* errinfo);
    bool init_cluster(struct ErrorInfo* errinfo);
    bool init_master_nodes(const std::vector<struct NodeInfo>& nodes_info, std::vector<struct NodeInfo>* replication_nodes_info, struct ErrorInfo* errinfo);
    void init_replica_nodes(const std::vector<struct NodeInfo>& replication_nodes_info);
    void update_slots(const struct NodeInfo& nodeinfo);
    void refresh_master_node_table(struct ErrorInfo* errinfo, const Node* error_node);
    void clear_and_update_master_nodes(const std::vector<struct NodeInfo>& nodes_info, std::vector<struct NodeInfo>* replication_nodes_info, struct ErrorInfo* errinfo);
    void clear_invalid_master_nodes(const NodeInfoTable& master_nodeinfo_table);
    bool add_master_node(const NodeInfo& nodeinfo, struct ErrorInfo* errinfo);
    void clear_all_master_nodes();
    void update_nodes_string(const NodeInfo& nodeinfo);
    redisContext* connect_redis_node(const Node& node, struct ErrorInfo* errinfo, bool readonly) const;
    CRedisNode* get_redis_node(int slot, bool readonly, const Node* ask_node, struct ErrorInfo* errinfo);
    CRedisMasterNode* get_redis_master_node(const NodeId& nodeid) const;
    CRedisMasterNode* random_redis_master_node() const;

private:
    // List the information of all cluster nodes
    bool list_cluster_nodes(std::vector<struct NodeInfo>* nodes_info, struct ErrorInfo* errinfo, redisContext* redis_context, const Node& node);

    // Called by: redis_command
    void extract_errtype(const redisReply* redis_reply, std::string* errtype) const;

public:
    // Called by: get,hget,key_type,lpop,rpop,srandmember
    bool get_value(const redisReply* redis_reply, std::string* value);

    // Called by: hkeys,hvals,lrange,mget,scan,smembers,spop,srandmember,sscan
    int get_values(const redisReply* redis_reply, std::vector<std::string>* values);

public:
    void set_command_monitor(CommandMonitor* command_monitor) { _command_monitor = command_monitor; }
    CommandMonitor* get_command_monitor() const { return _command_monitor; }

private:
    bool _enable_debug_log; // Default: true
    bool _enable_info_log;  // Default: true
    bool _enable_error_log; // Default: true

private:
    CommandMonitor* _command_monitor;
    std::string _raw_nodes_string; 
    std::string _nodes_string; 
    int _connect_timeout_milliseconds; // The connect timeout in milliseconds
    int _readwrite_timeout_milliseconds; // The receive and send timeout in milliseconds
    std::string _password;
    ReadPolicy _read_policy;

private:
#if __cplusplus < 201103L
    typedef std::tr1::unordered_map<Node, CRedisMasterNode*, NodeHasher> RedisMasterNodeTable;
    typedef std::tr1::unordered_map<NodeId, Node> RedisMasterNodeIdTable;
#else
    typedef std::unordered_map<Node, CRedisMasterNode*, NodeHasher> RedisMasterNodeTable;
    typedef std::unordered_map<NodeId, Node> RedisMasterNodeIdTable;
#endif // __cplusplus < 201103L
    RedisMasterNodeTable _redis_master_nodes; // Node -> CMasterNode
    RedisMasterNodeIdTable _redis_master_nodes_id; // NodeId -> Node

private:
    std::vector<Node> _nodes; // All nodes array
    std::vector<Node> _slot2node; // Slot -> Node

private:
    std::string _hincrby_shastr1;
    std::string _hmincrby_shastr1;
};

// Monitor the execution of the command by setting a CommandMonitor.
//
// Execution order:
// 1) before_command
// 2) command
// 3) after_command
class CommandMonitor
{
public:
    virtual ~CommandMonitor() {}

    // Called before each command is executed
    virtual void before_execute(const Node& node, const std::string& command, const CommandArgs& command_args, bool readonly) = 0;

    // Called after each command is executed
    // result The result of the execution of the command (0 success, 1 error, 2 timeout)
    virtual void after_execute(int result, const Node& node, const std::string& command, const redisReply* reply) = 0;
};

// Error code
enum
{
    ERROR_PARAMETER = -1,              // Parameter error
    ERROR_INIT_REDIS_CONN = -2,        // Initialize redis connection error
    ERROR_COMMAND = -3,                // Command error
    ERROR_CONNECT_REDIS = -4,          // Can not connect any cluster node
    ERROR_FORMAT = -5,                 // Format error
    ERROR_NOT_SUPPORT = -6,            // Not support
    ERROR_SLOT_NOT_EXIST = -7,         // Slot not exists
    ERROR_NOSCRIPT = -8,               // NOSCRIPT No matching script
    ERROR_UNKNOWN_REPLY_TYPE = -9,     // unknhown reply type
    ERROR_NIL = -10,                   // Redis return Nil
    ERROR_INVALID_COMMAND = -11,       // Invalid command
    ERROR_ZERO_KEY = -12,              // Key size is zero
    ERROR_REDIS_CONTEXT = -13,         // Can't allocate redis context
    ERROR_REDIS_AUTH = -14,             // Authorization failed
    ERROR_UNEXCEPTED_REPLY_TYPE = -15, // Unexcepted reply type
    ERROR_REPLY_FORMAT = -16,          // Reply format error
    ERROR_REDIS_READONLY = -17,
    ERROR_NO_ANY_NODE = -18
};

// Set NULL to discard log
typedef void (*LOG_WRITE)(const char* format, ...) __attribute__((format(printf, 1, 2)));
void set_error_log_write(LOG_WRITE error_log);
void set_info_log_write(LOG_WRITE info_log);
void set_debug_log_write(LOG_WRITE debug_log);

std::string strsha1(const std::string& str);
void debug_redis_reply(const char* command, const redisReply* redis_reply, int depth=0, int index=0);
uint16_t crc16(const char *buf, int len);
uint64_t crc64(uint64_t crc, const unsigned char *s, uint64_t l);

void millisleep(int milliseconds);
std::string get_formatted_current_datetime(bool with_milliseconds=false);
std::string format_string(const char* format, ...) __attribute__((format(printf, 1, 2)));
int split(std::vector<std::string>* tokens, const std::string& source, const std::string& sep, bool skip_sep=false);
int get_key_slot(const std::string* key);
bool keys_crossslots(const std::vector<std::string>& keys);
std::string int2string(int64_t n);
std::string int2string(int32_t n);
std::string int2string(int16_t n);
std::string int2string(uint64_t n);
std::string int2string(uint32_t n);
std::string int2string(uint16_t n);

// Convert a string into a int64_t. Returns true if the string could be parsed into a
// (non-overflowing) int64_t, false otherwise. The value will be set to the parsed value when appropriate.
bool string2int(const char* s, size_t len, int64_t* val, int64_t errval=-1);
bool string2int(const char* s, size_t len, int32_t* val, int32_t errval=-1);

} // namespace r3c {
#endif // REDIS_CLUSTER_CLIENT_H
