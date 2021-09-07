#include "HugeCTR/include/inference/database.hpp"

#include <HugeCTR/include/inference/hierarchicaldb.hpp>
#include <HugeCTR/include/inference/localized_db.hpp>
#include <HugeCTR/include/inference/redis.hpp>
#include <HugeCTR/include/inference/rocksdb.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
namespace HugeCTR {

template <typename TypeHashKey>
DataBase<TypeHashKey>::DataBase() {}

template <typename TypeHashKey>
DataBase<TypeHashKey>::~DataBase() {}

template <typename TypeHashKey>
DataBase<TypeHashKey>* DataBase<TypeHashKey>::load_base(DATABASE_TYPE type,
                                                        parameter_server_config ps_config) {
  if (type == DATABASE_TYPE::REDIS) {
    DataBase<TypeHashKey>* db = new redis<TypeHashKey>(ps_config);
    return db;
  } else if (type == DATABASE_TYPE::ROCKSDB) {
    DataBase<TypeHashKey>* db = new rocks_db<TypeHashKey>(ps_config);
    return db;
  } else if (type == DATABASE_TYPE::HIERARCHY) {
    DataBase<TypeHashKey>* db = new hierarchical_db<TypeHashKey>(ps_config);
    return db;
  } else {
    DataBase<TypeHashKey>* db = new localdb<TypeHashKey>(ps_config);
    return db;
  }
}

template <typename TypeHashKey>
DataBase<TypeHashKey>* DataBase<TypeHashKey>::get_base(const std::string& db_type) {
  DataBase<TypeHashKey>* db = new redis<TypeHashKey>();
  return db;
}

template class DataBase<unsigned int>;
template class DataBase<long long>;
}  // namespace HugeCTR