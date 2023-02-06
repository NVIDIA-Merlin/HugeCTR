#include "HugeCTR/core/hctr_impl/hctr_backend.hpp"
#include "HugeCTR/embedding/common.hpp"
#include "HugeCTR/embedding/data_distributor/data_distributor.hpp"
#include "HugeCTR/include/data_readers/async_reader/async_reader_common.hpp"
#include "HugeCTR/include/resource_managers/resource_manager_ext.hpp"
#include "gtest/gtest.h"

using namespace HugeCTR;
using namespace embedding;
//
// template <typename T>
// T round_up(T x, T y) {
//  return ((x + y - 1) / y) * y;
//}
//
// typedef uint32_t KeyType;
// typedef uint32_t OffsetType;
// typedef uint32_t IndexType;
// typedef float EmbType;
//
// const int batch_size = 8192;
// const int sparse_dim = 4;
//// table params
// const int num_tables = 4;
// const std::vector<int> max_hotnesses = {8, 20, 10, 1};
//
//// lookup params
// const std::vector<LookupParam> lookup_params = {
//    {0, 0, Combiner::Sum, max_hotnesses[0], 128},
//    {1, 1, Combiner::Sum, max_hotnesses[1], 128},
//    {2, 2, Combiner::Sum, max_hotnesses[2], 128},
//    {3, 3, Combiner::Sum, max_hotnesses[3], 128},
//};
//
// class DataDistributorTestFixture : public ::testing::Test {
// protected:
//  DataDistributorTestFixture() {}
//
//  bool gpus_found(std::vector<int> device_list) {
//    int num_available_gpus = 0;
//    HCTR_LIB_THROW(cudaGetDeviceCount(&num_available_gpus));
//
//    return num_available_gpus >= device_list.size();
//  }
//
//  auto allocate_output(EmbeddingCollectionParam& ebc_param, std::vector<int> device_list) {
//    std::map<size_t, size_t> group_num_features;
//    std::map<size_t, size_t> group_num_buckets;
//    for (int lookup_id = 0; lookup_id < ebc_param.num_lookup; ++lookup_id) {
//      embedding::LookupParam& lookup_param = ebc_param.lookup_params[lookup_id];
//      for (int group_id = 0; group_id < ebc_param.grouped_emb_params.size(); ++group_id) {
//        auto group_table_ids = ebc_param.grouped_emb_params[group_id].table_ids;
//        if (std::find(group_table_ids.begin(), group_table_ids.end(), lookup_param.table_id) !=
//            group_table_ids.end()) {
//          if (group_num_buckets.find(group_id) != group_num_buckets.end()) {
//            group_num_buckets[group_id]++;
//            group_num_features[group_id] += lookup_param.max_hotness;
//          } else {
//            group_num_buckets[group_id] = 1;
//            group_num_features[group_id] = lookup_param.max_hotness;
//          }
//        }
//      }
//    }
//
//    core::DataType key_type =
//        sizeof(KeyType) == 8 ? TensorScalarType::Int64 : TensorScalarType::UInt32;
//    std::vector<DataDistributor::CommResult> output;
//
//    for (auto device_id : device_list) {
//      CudaDeviceContext ctx(device_id);
//
//      DataDistributor::CommResult result;
//
//      for (int group_id = 0; group_id < ebc_param.grouped_emb_params.size(); ++group_id) {
//        size_t num_keys = group_num_features[group_id] * batch_size;
//        size_t num_bucket_ranges = group_num_buckets[group_id] * batch_size + 1;
//
//        //                printf("group_num_features[group_id]: %zu\n",
//        //                group_num_features[group_id]); printf("group_id: %d, h_num_keys: %zu,
//        //                num_bucket_ranges: %zu\n", group_id, h_num_keys, num_bucket_ranges);
//
//        auto core_resource_manager =
//            std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager_,
//            device_id);
//        auto buffer = core::GetBuffer(core_resource_manager);
//
//        result.num_keys.push_back(num_keys);
//        result.features.push_back(
//            buffer->reserve({num_keys}, core::DeviceType::GPU, key_type.type()));
//        result.bucket_ranges.push_back(buffer->reserve({num_bucket_ranges}, core::DeviceType::GPU,
//                                                       core::TensorScalarType::UInt32));
//
//        buffer->allocate();
//      }
//
//      output.push_back(result);
//    }
//
//    return output;
//  }
//
//  void create_data_distributor(std::vector<int> device_list,
//                               const std::vector<std::vector<int>>& shard_matrix,
//                               const std::vector<GroupedEmbeddingParam>& grouped_emb_params) {
//    std::vector<std::vector<int>> vvgpu;
//    vvgpu.push_back(device_list);
//    resource_manager_ = ResourceManagerExt::create(vvgpu, 424242);
//
//    EmbeddingCollectionParam ebc_param(
//        num_tables, {}, static_cast<int>(lookup_params.size()), lookup_params, shard_matrix,
//        grouped_emb_params, batch_size, HugeCTR::TensorScalarTypeFunc<KeyType>::get_type(),
//        HugeCTR::TensorScalarTypeFunc<IndexType>::get_type(),
//        HugeCTR::TensorScalarTypeFunc<OffsetType>::get_type(),
//        HugeCTR::TensorScalarTypeFunc<EmbType>::get_type(), EmbeddingLayout::FeatureMajor,
//        EmbeddingLayout::FeatureMajor, false);
//
//    int num_local_gpus = static_cast<int>(device_list.size());
//    std::vector<std::shared_ptr<core::CoreResourceManager>> core_resource_managers;
//    for (int local_gpu_id = 0; local_gpu_id < num_local_gpus; ++local_gpu_id) {
//      auto core_resource_manager =
//          std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager_,
//          local_gpu_id);
//      core_resource_managers.push_back(core_resource_manager);
//    }
//    output_ = allocate_output_for_data_distributor(core_resource_managers, ebc_param);
//
//    data_distributor_ = std::make_shared<DataDistributor>(
//        batch_size, TensorScalarType::UInt32, resource_manager_, core_resource_managers,
//        ebc_param);
//  }
//
//  template <typename T>
//  static core::Tensor to_core_tensor(Tensor2<T> native_tensor) {
//    core::Device device = core::DeviceType::GPU;
//    core::Storage storage = std::make_shared<hctr_internal::NativeHCTRStorageWrapper>(
//        native_tensor.get_ptr(), native_tensor.get_size_in_bytes());
//
//    auto t_impl =
//        std::make_shared<core::TensorImpl>(storage, 0, native_tensor.get_dimensions(), device,
//                                           HugeCTR::TensorScalarTypeFunc<T>::get_type());
//    return core::Tensor(t_impl);
//  }
//
//  auto generate_dp_feature_major_shards(size_t num_features, std::vector<int> device_list) {
//    std::vector<std::vector<core::Tensor>> dp_shards;
//    for (auto device_id : device_list) {
//      CudaDeviceContext ctx(device_id);
//      auto allocator = GeneralBuffer2<CudaAllocator>::create();
//
//      std::vector<Tensor2<KeyType>> features(num_features);
//
//      for (int lookup_id = 0; lookup_id < num_features; ++lookup_id) {
//        size_t max_hotness = lookup_params[lookup_id].max_hotness;
//        allocator->reserve({batch_size / device_list.size(), max_hotness}, &features[lookup_id]);
//      }
//
//      allocator->allocate();
//
//      std::vector<core::Tensor> core_features;
//      for (int lookup_id = 0; lookup_id < num_features; ++lookup_id) {
//        auto core_tensor = to_core_tensor(features[lookup_id]);
//        std::vector<KeyType> keys(core_tensor.get_num_elements(), lookup_id);
//        core_tensor.copy_from(keys);
//        core_features.push_back(core_tensor);
//      }
//
//      dp_shards.push_back(core_features);
//    }
//    return dp_shards;
//  }
//
//  //    void check_data()
//  //    {
//  //        for (int i = 0; i < device_list.size(); ++i)
//  //        {
//  //            const auto& result = output_[i];
//  //
//  //            for (int group_id = 0; group_id < ebc_param_->grouped_emb_params.size();
//  ++group_id)
//  //            {
//  //                switch (ebc_param_->grouped_emb_params[group_id].table_placement_strategy)
//  //                {
//  //                    case embedding::TablePlacementStrategy::ModelParallel:
//  //                    {
//  //                        break;
//  //                    }
//  //                    case embedding::TablePlacementStrategy::DataParallel:
//  //                    {
//  //                        break;
//  //                    }
//  //                    default:
//  //                        throw std::runtime_error("Table placement strategy not supported in
//  //                        DataDistributor");
//  //                }
//  //            }
//  //            }
//  //        }
//  //    }
//
//  void test_distributor(size_t num_features, std::vector<int> device_list) {
//    auto dp_shards = generate_dp_feature_major_shards(num_features, device_list);
//
//    // distribute
//    for (int i = 0; i < device_list.size(); ++i) {
//      CudaDeviceContext ctx(device_list[i]);
//      data_distributor_->distribute(dp_shards[i], i, output_[i]);
//    }
//
//    // synchronize
//    for (int i = 0; i < device_list.size(); ++i) {
//      CudaDeviceContext ctx(device_list[i]);
//      HCTR_LIB_THROW(cudaStreamSynchronize(resource_manager_->get_local_gpu(i)->get_stream()));
//    }
//
//    // check
//  }
//
//  //    virtual void SetUp()
//  //    {
//  //        auto device_list = std::get<0>(GetParam());
//  //        batch_size_ = std::get<1>(GetParam());
//  //        sparse_dim_ = 5;//26; // num features
//  //
//  //        EXPECT_EQ(batch_size_ % device_list.size(), 0);
//  //        size_t shard_size = batch_size_ / device_list.size();
//  //
//  //        // initialize resource managers
//  //        std::vector<std::vector<int>> vvgpu;
//  //        vvgpu.push_back(device_list);
//  //        auto resource_manager_ext = ResourceManagerExt::create(vvgpu, 424242);
//  //        for (auto gpu_id : device_list)
//  //        {
//  //
//  resource_managers_.push_back(std::make_shared<hctr_internal::HCTRCoreResourceManager>(resource_manager_ext,
//  //            gpu_id));
//  //
//  //            CudaDeviceContext ctx(gpu_id);
//  //        }
//  //
//  //        // initialize pooling factor
//  //        std::generate_n(std::back_inserter(feature_pooling_factors_), sparse_dim_, [](){
//  return
//  //        rand() % 10 + 1; } ); max_dim_ = std::accumulate(feature_pooling_factors_.begin(),
//  //                                   feature_pooling_factors_.end(), 0);
//  //
//  //        std::cout << "Batch size: " << batch_size_ << std::endl;
//  //        std::cout << "Sparse dim: " << sparse_dim_ << std::endl;
//  //        std::cout << "Pooling factors: ";
//  //        for (auto factor : feature_pooling_factors_)
//  //            std::cout << factor << " ";
//  //        std::cout << std::endl;
//  //        std::cout << "Max sample dim: " << max_dim_ << std::endl;
//  //
//  //        // allocate feature shards
//  //        feature_shards_.resize(device_list.size());
//  //        for (size_t i = 0; i < device_list.size(); ++i)
//  //        {
//  //            auto gpu_id = device_list[i];
//  //            CudaDeviceContext ctx(gpu_id);
//  //
//  //            for (size_t feat_id = 0; feat_id < sparse_dim_; ++feat_id)
//  //            {
//  //                size_t shard_size_bytes = shard_size * feature_pooling_factors_[feat_id] *
//  //                sizeof(SparseType);
//  //
//  //                auto buffer = std::make_shared<RawPtrBuffer>(shard_size_bytes);
//  //                Tensor2<SparseType> tensor({shard_size_bytes / sizeof(SparseType)}, buffer);
//  //
//  //                std::vector<SparseType> init_data(shard_size_bytes / sizeof(SparseType),
//  //                feat_id);
//  //
//  //                HCTR_LIB_THROW(cudaMemcpy(tensor.get_ptr(), init_data.data(),
//  shard_size_bytes,
//  //                cudaMemcpyHostToDevice));
//  //
//  //                feature_shards_[i].emplace_back(tensor);
//  //            }
//  //        }
//  //    }
//
//  //    virtual void TearDown()
//  //    {
//  //        resource_managers_.clear();
//  //        feature_shards_.clear();
//  //        feature_pooling_factors_.clear();
//  //    }
//
//  //    template <typename T>
//  //    core::Tensor convert(HugeCTR::Tensor2<T> native_tensor)
//  //    {
//  //        auto storage =
//  //        std::make_shared<hctr_internal::NativeHCTRStorageWrapper>(native_tensor.get_ptr(),
//  // native_tensor.get_size_in_bytes());
//  //
//  //        auto t_impl = std::make_shared<core::TensorImpl>(storage, 0,
//  //        native_tensor.get_dimensions(),
//  //                                                         core::DeviceType::GPU,
//  // HugeCTR::TensorScalarTypeFunc<T>::get_type());
//  //        return core::Tensor(t_impl);
//  //    }
//  //
//  //    std::vector<size_t> get_device_feature_ids(std::vector<std::vector<bool>>
//  //    resident_feature_tables, int device_id)
//  //    {
//  //        std::vector<size_t> gpu_feature_ids;
//  //        for (size_t feat_id=0; feat_id<sparse_dim_; ++feat_id) {
//  //            if (resident_feature_tables[device_id][feat_id])
//  //                gpu_feature_ids.push_back(feat_id);
//  //        }
//  //        return gpu_feature_ids;
//  //    }
//
//  //    void check_features(core::Tensor features, std::vector<size_t> feature_ids)
//  //    {
//  //        size_t num_expected_elements = 0;
//  //        for (auto feat_id : feature_ids)
//  //            num_expected_elements += feature_pooling_factors_[feat_id] * batch_size_;
//  //
//  //        EXPECT_EQ(features.get_num_elements(), num_expected_elements);
//  //        std::vector<SparseType> result_data(features.get_num_elements(), 0);
//  //
//  //        EXPECT_EQ(cudaMemcpy(result_data.data(), features.get(),features.nbytes(),
//  //        cudaMemcpyDeviceToHost), cudaSuccess);
//  //
//  //        // validate feature tensors
//  //        size_t offset = 0;
//  //        for (auto feat_id : feature_ids)
//  //        {
//  //            const size_t num_elems = batch_size_ * feature_pooling_factors_[feat_id];
//  //            std::vector<SparseType> expected(num_elems, feat_id); // memset to feat_id
//  //            std::vector<SparseType> actual(result_data.begin() + offset, result_data.begin() +
//  //            offset + num_elems);
//  //
//  //            EXPECT_EQ(actual, expected); // tensors match
//  //
//  //            offset += num_elems;
//  //        }
//  //    }
//
//  //    void check_bucket_ranges(core::Tensor bucket_ranges, std::vector<size_t> feature_ids)
//  //    {
//  //        std::vector<DataDistributionLayer::BucketRangeType> ranges;
//  //        bucket_ranges.to(&ranges);
//  //
//  //        EXPECT_EQ(bucket_ranges.get_num_elements(), feature_ids.size() * batch_size_ + 1);
//  //
//  //        std::vector<DataDistributionLayer::BucketRangeType> expected_ranges;
//  //
//  //        size_t sum = 0;
//  //        size_t idx = 0;
//  //        for (auto feat_id : feature_ids)
//  //        {
//  //            auto factor = feature_pooling_factors_[feat_id];
//  //            for (size_t i = 0; i < batch_size_; ++i)
//  //            {
//  //                expected_ranges.push_back(sum);
//  //                sum += factor;
//  //                idx++;
//  //            }
//  //        }
//  //        expected_ranges.push_back(sum);
//  //        EXPECT_EQ(expected_ranges, ranges);
//  //    }
//
//  std::shared_ptr<ResourceManager> resource_manager_;
//  std::shared_ptr<DataDistributor> data_distributor_;
//  std::vector<DataDistributor::CommResult> output_;
//};
//
// TEST_F(DataDistributorTestFixture, dp_one_gpu) {
//  const std::vector<int> device_list = {0};
//  if (!gpus_found(device_list)) {
//    GTEST_SKIP() << "Skipping tests. Not enough GPUs.";
//  }
//
//  const std::vector<std::vector<int>> shard_matrix = {
//      {1, 1, 1, 1},
//  };
//
//  const std::vector<GroupedEmbeddingParam> grouped_emb_params = {
//      {TablePlacementStrategy::DataParallel, {0, 1, 2, 3}}};
//
//  create_data_distributor(device_list, shard_matrix, grouped_emb_params);
//
//  test_distributor(num_tables, device_list);
//}
//
// TEST_F(DataDistributorTestFixture, dp_multi_gpu) {
//  const std::vector<int> device_list = {0, 1};
//  if (!gpus_found(device_list)) {
//    GTEST_SKIP() << "Skipping tests. Not enough GPUs.";
//  }
//
//  const std::vector<std::vector<int>> shard_matrix = {
//      {1, 1, 1, 1},
//      {1, 1, 1, 1},
//  };
//
//  const std::vector<GroupedEmbeddingParam> grouped_emb_params = {
//      {TablePlacementStrategy::DataParallel, {0, 1, 2, 3}}};
//
//  create_data_distributor(device_list, shard_matrix, grouped_emb_params);
//
//  test_distributor(num_tables, device_list);
//}
//
// TEST_F(DataDistributorTestFixture, mp_only) {}
//
// TEST_F(DataDistributorTestFixture, mp_and_dp) {}

// TEST_P(DataDistributorTestFixture, single_node_all_gather)
//{
//     DataDistributionLayer data_distributor(batch_size_, core::TensorScalarType::Float32,
//     resource_managers_, feature_pooling_factors_, std::nullopt);
//
//     const size_t num_local_gpus = resource_managers_.size();
// #pragma omp parallel for num_threads(num_local_gpus)
//     for (size_t gpu_i = 0; gpu_i < num_local_gpus; ++gpu_i)
//     {
//         CudaDeviceContext ctx(resource_managers_[gpu_i]->get_device_id());
//
//         // This might seem pointless to initialize the test with Tensor2, but this is because
//         these will be the tensor
//         // types returned by the data reader.
//         std::vector<core::Tensor> tensors;
//         std::transform(feature_shards_[gpu_i].begin(), feature_shards_[gpu_i].end(),
//                        std::back_inserter(tensors),
//                        [this](Tensor2<SparseType>& t){ return convert(t); });
//
//         EXPECT_EQ(tensors.size(), feature_shards_[gpu_i].size());
//
//         // do distribution - communicate the shards between GPUs
//         DataDistributionLayer::CommResult result = data_distributor.distribute(batch_size_,
//         tensors, gpu_i);
//         HCTR_LIB_THROW(cudaStreamSynchronize(resource_managers_[gpu_i]->get_local_gpu()->get_stream()));
//
//         // gpu has all feature tables
//         std::vector<size_t> gpu_feature_ids(sparse_dim_);
//         std::iota(gpu_feature_ids.begin(), gpu_feature_ids.end(), 0);
//
//         check_features(result.features, gpu_feature_ids);
//         check_bucket_ranges(result.bucket_ranges, gpu_feature_ids);
//     }
// }
//
// TEST_P(DDLTestFixture, single_node_selected_gather)
//{
//     std::vector<std::vector<bool>> resident_feature_tables;
//     for (size_t i = 0; i < resource_managers_.size(); ++i)
//     {
//         std::vector<bool> gpu_resident_feature_tables(sparse_dim_);
//         std::generate_n(gpu_resident_feature_tables.begin(), sparse_dim_, [](){ return rand() %
//         2; });
//
//         std::cout << "GPU(" << i << ") resident feature tables: ";
//         for (auto b : gpu_resident_feature_tables) {
//             std::cout << b << " ";
//         }
//         std::cout << std::endl;
//         resident_feature_tables.emplace_back(gpu_resident_feature_tables);
//     }
//
//     DataDistributionLayer data_distributor(batch_size_, core::TensorScalarType::Float32,
//     resource_managers_, feature_pooling_factors_,
//                               resident_feature_tables);
//
//     const size_t num_local_gpus = resource_managers_.size();
// #pragma omp parallel for num_threads(num_local_gpus)
//     for (size_t i=0; i<num_local_gpus; ++i)
//     {
//         CudaDeviceContext ctx(resource_managers_[i]->get_device_id());
//
//         // This might seem pointless to initialize the test with Tensor2, but this is because
//         these will be the tensor
//         // types returned by the data reader.
//         std::vector<core::Tensor> tensors;
//         std::transform(feature_shards_[i].begin(), feature_shards_[i].end(),
//                        std::back_inserter(tensors), [this](Tensor2<SparseType>& t){ return
//                        convert(t); });
//
//         EXPECT_EQ(tensors.size(), feature_shards_[i].size());
//
//         // do distribution - communicate the shards between GPUs
//         DataDistributionLayer::CommResult result = data_distributor.distribute(batch_size_,
//         tensors, i);
//         HCTR_LIB_THROW(cudaStreamSynchronize(resource_managers_[i]->get_local_gpu()->get_stream()));
//
//         auto gpu_feature_ids = get_device_feature_ids(resident_feature_tables, i);
//
//         check_features(result.features, gpu_feature_ids);
//         check_bucket_ranges(result.bucket_ranges, gpu_feature_ids);
//     }
// }
