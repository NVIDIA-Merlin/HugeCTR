cmake_minimum_required(VERSION 3.8)
include_directories(/opt/conda/include)

include(FetchContent)

FetchContent_Declare(
  pybind11_sources
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.2
)

FetchContent_GetProperties(pybind11_sources)

if(NOT pybind11_sources_POPULATED)
  FetchContent_Populate(pybind11_sources)

  add_subdirectory(
    ${pybind11_sources_SOURCE_DIR}
    ${pybind11_sources_BINARY_DIR}
    )
endif()

file(GLOB huge_ctr_src
  ../src/cpu_resource.cpp
  ../src/gpu_resource.cpp
  ../src/resource_manager.cpp
  ../src/data_simulator.cu
  ../src/data_reader.cpp
  ../src/layer.cpp
  ../src/layers/batch_norm_layer.cu
  ../src/layers/cast_layer.cu
  ../src/layers/concat_layer.cu
  ../src/layers/dropout_layer.cu
  ../src/layers/dropout_cudnn_layer.cu
  ../src/layers/elu_layer.cu
  ../src/layers/fully_connected_layer.cu
  ../src/layers/fully_connected_layer_half.cu
  ../src/layers/fused_fully_connected_layer.cu
  ../src/layers/interaction_layer.cu
  ../src/layers/relu_layer.cu
  ../src/layers/reshape_layer.cu
  ../src/layers/slice_layer.cu
  ../src/layers/fm_order2_layer.cu
  ../src/layers/multiply_layer.cu
  ../src/layers/multi_cross_layer.cu
  ../src/layers/add_layer.cu
  ../src/layers/reduce_sum_layer.cu
  ../src/layers/dot_product_layer.cu
  ../src/loss.cu
  ../src/network.cu
  ../src/network.cpp
  ../src/data_readers/metadata.cpp
  ../src/metrics.cu
  ../src/optimizers/adam_optimizer.cu
  ../src/optimizers/momentum_sgd_optimizer.cu
  ../src/optimizers/nesterov_optimizer.cu
  ../src/optimizers/sgd_optimizer.cu
  ../src/optimizer.cpp
  ../src/regularizer.cu
  ../src/regularizers/l1_regularizer.cu
  ../src/regularizers/l2_regularizer.cu
  ../src/regularizers/no_regularizer.cu
  ../src/parser.cpp
  ../src/parsers/solver_parser.cpp
  ../src/parsers/learning_rate_scheduler_parser.cpp
  ../src/session.cpp
  ../src/plan_parser.cpp
  ../src/data_readers/data_collector.cu
  ../src/data_readers/parquet_data_converter.cu
  ../src/hashtable/nv_hashtable.cu
  ../src/embeddings/sync_all_gpus_functor.cu
  ../src/embeddings/init_embedding_functor.cu
  ../src/embeddings/forward_per_gpu_functor.cu
  ../src/embeddings/forward_scale_functor.cu
  ../src/embeddings/forward_reorder_functor.cu
  ../src/embeddings/forward_mapping_per_gpu_functor.cu
  ../src/embeddings/forward_fuse_per_gpu_functor.cu
  ../src/embeddings/store_slot_id_functor.cu
  ../src/embeddings/backward_functor.cu
  ../src/embeddings/backward_reorder_functor.cu
  ../src/embeddings/backward_fuse_per_gpu_functor.cu
  ../src/embeddings/update_params_functor.cu
  ../src/embeddings/get_update_params_results_functor.cu
  ../src/embeddings/reduce_scatter_functor.cu
  ../src/embeddings/all_reduce_functor.cu
  ../src/embeddings/all_gather_functor.cu
  ../src/embeddings/all2all_init_forward_functor.cu
  ../src/embeddings/all2all_init_backward_functor.cu
  ../src/embeddings/all2all_forward_functor.cu
  ../src/embeddings/all2all_backward_functor.cu
  ../src/embeddings/all2all_exec_functor.cu
  ../src/embeddings/get_forward_results_functor.cu
  ../src/embeddings/get_backward_results_functor.cu
  ../src/embeddings/utils_functor.cu
  ../src/embeddings/distributed_slot_sparse_embedding_hash.cu
  ../src/embeddings/localized_slot_sparse_embedding_hash.cu
  ../src/embeddings/localized_slot_sparse_embedding_one_hot.cu
  ../src/model_oversubscriber/distributed_parameter_server_delegate.cpp
  ../src/model_oversubscriber/model_oversubscriber_impl.cpp
  ../src/model_oversubscriber/parameter_server.cpp
  ../src/model_oversubscriber/parameter_server_manager.cpp
  ../src/diagnose.cu
)


set(CMAKE_CXX_STANDARD 14)
add_library(huge_ctr_shared SHARED ${huge_ctr_src})

if(MPI_FOUND)
  target_link_libraries(huge_ctr_shared PUBLIC cublas curand cudnn nccl nvToolsExt ${CMAKE_THREAD_LIBS_INIT} ${MPI_CXX_LIBRARIES} hwloc ucp ucs ucm uct cudf)
  message(STATUS "${MPI_CXX_LIBRARIES}")
else()
  target_link_libraries(huge_ctr_shared PUBLIC cublas curand cudnn nccl nvToolsExt ${CMAKE_THREAD_LIBS_INIT} cudf)
endif()

#target_link_libraries(huge_ctr_shared PRIVATE nlohmann_json::nlohmann_json)

target_compile_features(huge_ctr_shared PUBLIC cxx_std_14)

set_target_properties(huge_ctr_shared PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)



link_directories(
    /usr/local/cuda/lib64
  )

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")


# create python module
add_library(hugectr
  MODULE
    module_main.cpp
  )

target_link_libraries(hugectr
    PUBLIC
      pybind11::module
      ${PYTHON_LIBRARIES}
      ${CUDA_LIBRARIES}
      huge_ctr_shared

  )

set_target_properties(hugectr
  PROPERTIES
    PREFIX ""
    #LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/HugeCTR/pybind
    #PREFIX "${PYTHON_MODULE_PREFIX}"
    #SUFFIX "${PYTHON_MODULE_EXTENSION}"
  )

