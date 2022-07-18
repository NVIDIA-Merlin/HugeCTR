#include <map>
#include <memory>
#include <random>
#include <vector>

#include "HugeCTR/include/general_buffer2.hpp"
#include "HugeCTR/include/resource_managers/resource_manager_ext.hpp"
#include "HugeCTR/include/tensor2.hpp"
#include "gtest/gtest.h"
#include "input_generator.hpp"
#include "utest/test_utils.h"

// all your base are belong to us
#define private public
#define protected public
#include "HugeCTR/include/embeddings/hybrid_sparse_embedding.hpp"

using namespace HugeCTR;

constexpr bool debug_print = false;
int global_seed = 0;

template <typename dtype, typename emtype>
void end_to_end_impl(std::vector<int> device_list, HybridEmbeddingInputGenerator<dtype> *generator,
                     size_t batch_size, size_t embedding_vec_size, double bw_ratio_a2a_over_ar,
                     size_t seed, size_t num_evals) {
  constexpr double epsilon = sizeof(emtype) < 4 ? 1e-2 : 1e-3;

  int rank = 0, num_procs = 1;
#ifdef ENABLE_MPI
  HCTR_MPI_THROW(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  HCTR_MPI_THROW(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
#endif
  HCTR_LIB_THROW(nvmlInit_v2());

  std::vector<std::vector<int>> vvgpu;
  size_t num_local_gpus = device_list.size();
  // size_t num_total_gpus = num_procs*num_local_gpus;

  // if there are multi-node, we assume each node has the same gpu device_list
  for (int i = 0; i < num_procs; i++) {
    vvgpu.push_back(device_list);
  }
  const auto resource_manager = ResourceManagerExt::create(vvgpu, seed);

  size_t total_gpu_count = resource_manager->get_global_gpu_count();
  size_t local_gpu_count = resource_manager->get_local_gpu_count();
  size_t local_batch_size = batch_size / total_gpu_count;
  assert(batch_size % total_gpu_count == 0);

  auto table_sizes = generator->get_table_sizes();
  size_t num_tables = table_sizes.size();
  size_t total_categories = std::accumulate(table_sizes.begin(), table_sizes.end(), 0);
  HCTR_LOG(INFO, WORLD, "total categories: %lu\n", total_categories);

  size_t num_init_batches = 50;

  SparseTensors<dtype> inputs;
  SparseTensors<dtype> inits;
  for (size_t i = 0; i < local_gpu_count; i++) {
    CudaDeviceContext context(resource_manager->get_local_gpu(i)->get_device_id());
    auto buf = GeneralBuffer2<CudaManagedAllocator>::create();
    Tensor2<dtype> value_tensor;
    buf->reserve({batch_size, num_tables}, &value_tensor);
    auto dummy_row_offset_tensor = Tensor2<dtype>();
    std::shared_ptr<size_t> dummy_nnz(new size_t);
    inputs.emplace_back(SparseTensor<dtype>(value_tensor, dummy_row_offset_tensor, dummy_nnz));

    buf->reserve({num_init_batches * batch_size, num_tables}, &value_tensor);
    inits.emplace_back(SparseTensor<dtype>(value_tensor, dummy_row_offset_tensor, dummy_nnz));
    buf->allocate();
  }

  const float lr = 0.42f;

  GpuLearningRateSchedulers lr_scheds;
  for (size_t i = 0; i < local_gpu_count; i++) {
    lr_scheds.emplace_back(new GpuLearningRateScheduler(2 * lr, 2, 0, 1, 2.f, 0.f,
                                                        resource_manager->get_local_gpu(i)));
    lr_scheds.back()->update();
  }

  HybridSparseEmbeddingParams params = {
      batch_size,
      batch_size,
      num_init_batches,
      2 * num_tables * batch_size,
      -1,
      0.01,  // p_max_dup ?
      embedding_vec_size,
      num_tables,
      generator->get_table_sizes(),
      num_procs == 1 ? hybrid_embedding::CommunicationType::NVLink_SingleNode
                     : hybrid_embedding::CommunicationType::IB_NVLink,
      1.0,
      bw_ratio_a2a_over_ar,
      1.0,
      false,
      false,
      HybridEmbeddingType::Distributed,
      OptParams{Optimizer_t::SGD, lr, {}, Update_t::Global, 1.0f}};

  std::vector<std::shared_ptr<BufferBlock2<emtype>>> placeholder(
      resource_manager->get_local_gpu_count(), NULL);
  auto embedding = std::make_unique<HybridSparseEmbedding<dtype, emtype>>(
      inputs, inputs, params, placeholder, lr_scheds, false, resource_manager, false, false);

  // Table offsets
  std::vector<size_t> table_offsets(num_tables);
  size_t total = 0;
  for (size_t table = 0; table < num_tables; table++) {
    table_offsets[table] = total;
    total += generator->get_table_sizes()[table];
  }

  auto initial_input = generator->generate_categorical_input(num_init_batches * batch_size);

  if (debug_print) {
    std::map<dtype, int> unique_cat;
    HCTR_LOG(INFO, ROOT, "Generated INIT unique categories:  ");
    for (size_t i = 0; i < num_init_batches * batch_size; i++) {
      for (size_t j = 0; j < num_tables; j++) {
        unique_cat[initial_input[i * num_tables + j] + table_offsets[j]] = 1;
      }
    }
    for (auto c : unique_cat) {
      HCTR_PRINT(INFO, " %d", (int)c.first);
    }
    HCTR_PRINT(INFO, "\n");
  }

  for (size_t lgpu = 0; lgpu < local_gpu_count; ++lgpu) {
    CudaDeviceContext context(resource_manager->get_local_gpu(lgpu)->get_device_id());
    auto stream = resource_manager->get_local_gpu(lgpu)->get_stream();
    upload_tensor(initial_input, inits[lgpu].get_value_tensor(), stream);
  }
  size_t tmp_size = 0;
  embedding->init_model(inits, tmp_size);

  size_t num_frequent = embedding->model_[0].num_frequent;
  if (rank == 0) {
    HCTR_LOG(INFO, WORLD, "Number of frequent categories: %ld\n", num_frequent);
  }
  std::vector<size_t> num_infrequent(local_gpu_count);
  for (size_t i = 0; i < local_gpu_count; i++) {
    num_infrequent[i] = embedding->model_[i].h_infrequent_model_table_offsets[num_tables];
    // if (debug_print) {
    HCTR_LOG(INFO, WORLD, "local_gpu = %ld, Number of infrequent categories: %ld\n", i,
             num_infrequent[i]);
    //}
  }

  std::vector<float> full_emb_table(total_categories * embedding_vec_size);
  {
    std::mt19937 gen(seed + 2);
    std::uniform_real_distribution<float> distr(-1, 1);
    for (auto &e : full_emb_table) {
      e = distr(gen);
    }
  }

  // Set frequent embeddings
  for (size_t device = 0; device < local_gpu_count; device++) {
    CudaDeviceContext context(resource_manager->get_local_gpu(device)->get_device_id());

    std::vector<dtype> h_frequent_categories;
    download_tensor(h_frequent_categories, embedding->model_[device].frequent_categories, 0);

    for (size_t i = 0; i < num_frequent; ++i) {
      dtype cat = h_frequent_categories[i];
      cudaMemcpy(embedding->frequent_embeddings_single_node_[device]
                         .frequent_data_.frequent_embedding_vectors_.get_ptr() +
                     i * embedding_vec_size,
                 full_emb_table.data() + cat * embedding_vec_size,
                 sizeof(float) * embedding_vec_size, cudaMemcpyHostToDevice);
    }

    if (debug_print && device == 0) {
      HCTR_LOG(INFO, ROOT, "Frequent categories: ");
      for (size_t i = 0; i < num_frequent; i++) {
        HCTR_PRINT(INFO, " %d", h_frequent_categories[i]);
      }
      HCTR_PRINT(INFO, "\n");
    }
  }

  // Set infrequent embeddings
  for (size_t device = 0; device < local_gpu_count; device++) {
    CudaDeviceContext context(resource_manager->get_local_gpu(device)->get_device_id());
    int global_id = resource_manager->get_local_gpu(device)->get_global_id();
    HCTR_LIB_THROW(cudaDeviceSynchronize());

    size_t num_infrequent = embedding->model_[device].h_infrequent_model_table_offsets[num_tables];

    float *h_infrequent_embedding_vectors;
    dtype *h_category_location;
    HCTR_LIB_THROW(cudaMallocHost((void **)&h_infrequent_embedding_vectors,
                                  (num_infrequent + 1) * embedding_vec_size * sizeof(float)));
    HCTR_LIB_THROW(
        cudaMallocHost((void **)&h_category_location, total_categories * 2 * sizeof(dtype)));

    HCTR_LIB_THROW(cudaMemcpy(h_category_location,
                              embedding->model_[device].category_location.get_ptr(),
                              total_categories * 2 * sizeof(dtype), cudaMemcpyDeviceToHost));

    if (debug_print) {
      HCTR_LOG(INFO, ROOT, "Category location array:\n");
      for (size_t i = 0; i < total_categories; i++) {
        HCTR_PRINT(INFO, "  (%d, %d)\n", h_category_location[2 * i],
                   h_category_location[2 * i + 1]);
      }
    }

    for (size_t i = 0; i < total_categories; ++i) {
      if ((int)h_category_location[2 * i] == global_id &&
          (size_t)h_category_location[2 * i + 1] < total_categories) {
        auto loc = h_category_location[2 * i + 1];
        memcpy(h_infrequent_embedding_vectors + loc * embedding_vec_size,
               full_emb_table.data() + i * embedding_vec_size, sizeof(float) * embedding_vec_size);
        /*
        if(device == 0)
        {
          HCTR_LOG(INFO, WORLD, "i = %ld, loc = %d, embed[0]  = %f\n", i, loc,
        *(h_infrequent_embedding_vectors+loc*embedding_vec_size));
        }
        */
      }
    }

    if (embedding->embedding_params_.communication_type == CommunicationType::NVLink_SingleNode) {
      cudaMemcpy(embedding->infrequent_embeddings_single_node_[device]
                     .infrequent_embedding_vectors_.get_ptr(),
                 h_infrequent_embedding_vectors,
                 num_infrequent * embedding_vec_size * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (embedding->embedding_params_.communication_type == CommunicationType::IB_NVLink) {
      cudaMemcpy(embedding->infrequent_embeddings_ib_nvlink_[device]
                     .infrequent_embedding_vectors_.get_ptr(),
                 h_infrequent_embedding_vectors,
                 num_infrequent * embedding_vec_size * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (embedding->embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
      cudaMemcpy(embedding->infrequent_embeddings_ib_nvlink_hier_[device]
                     .infrequent_embedding_vectors_.get_ptr(),
                 h_infrequent_embedding_vectors,
                 num_infrequent * embedding_vec_size * sizeof(float), cudaMemcpyHostToDevice);
    }
    // HCTR_LOG(INFO, WORLD, "gpu = %ld, num_infrequent = %ld, infrequent_embedding_vectors_ =
    // 0x%lx\n", device, num_infrequent,
    // (size_t)(embedding->infrequent_embeddings_[device].infrequent_embedding_vectors_.get_ptr()));
    HCTR_LIB_THROW(cudaFreeHost(h_infrequent_embedding_vectors));
    HCTR_LIB_THROW(cudaFreeHost(h_category_location));
  }

  if (debug_print) {
    HCTR_LOG(INFO, ROOT, "Generated full embedding table\n");
    for (size_t i = 0; i < full_emb_table.size(); i++) {
      HCTR_PRINT(INFO, "%8.5f ", (float)full_emb_table[i]);
      if (i % embedding_vec_size == embedding_vec_size - 1) {
        HCTR_PRINT(INFO, "\n");
      }
    }
    HCTR_PRINT(INFO, "\n");
  }

  auto outputs = embedding->get_train_output_tensors();
  //======================================================================================
  // Do the forward step
  //======================================================================================
  auto input = generator->generate_categorical_input(batch_size);
  for (size_t lgpu = 0; lgpu < local_gpu_count; ++lgpu) {
    CudaDeviceContext context(resource_manager->get_local_gpu(lgpu)->get_device_id());
    auto stream = resource_manager->get_local_gpu(lgpu)->get_stream();
    upload_tensor(input, inputs[lgpu].get_value_tensor(), stream);
  }

  if (debug_print) {
    HCTR_LOG(INFO, ROOT, "Generated input:\n");
    HCTR_PRINT(INFO, "  Table sizes: ");
    for (auto sz : generator->get_table_sizes()) {
      HCTR_PRINT(INFO, "%ld ", sz);
    }
    HCTR_PRINT(INFO, "\n");
    HCTR_PRINT(INFO, "  Input:\n");
    for (size_t i = 0; i < batch_size; i++) {
      HCTR_PRINT(INFO, "   [ ");
      for (size_t j = 0; j < num_tables; j++) {
        HCTR_PRINT(INFO, "%7d ", input[i * num_tables + j]);
      }
      HCTR_PRINT(INFO, " ]\n");
    }
  }

  embedding->forward(true);

  if (debug_print) {
    const int device = 0;
    CudaDeviceContext context(resource_manager->get_local_gpu(device)->get_device_id());
    int global_id = resource_manager->get_local_gpu(device)->get_global_id();
    HCTR_LIB_THROW(cudaDeviceSynchronize());

    {
      std::vector<dtype> tmp;
      if (embedding->embedding_params_.communication_type == CommunicationType::NVLink_SingleNode) {
        download_tensor(
            tmp, embedding->infrequent_embeddings_single_node_[device].indices_->model_indices_, 0);
      }
      if (embedding->embedding_params_.communication_type == CommunicationType::IB_NVLink) {
        download_tensor(
            tmp, embedding->infrequent_embeddings_ib_nvlink_[device].indices_->model_indices_, 0);
      }
      if (embedding->embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
        download_tensor(
            tmp, embedding->infrequent_embeddings_ib_nvlink_hier_[device].indices_->model_indices_,
            0);
      }

      // download_tensor(tmp, embedding->infrequent_embeddings_[device].indices_->model_indices_,
      // 0);

      HCTR_LOG(INFO, ROOT, "Instance %d model indices: ", global_id);
      for (size_t j = 0; j < tmp.size(); j++) {
        HCTR_PRINT(INFO, " %d", (int)tmp[j]);
      }
      HCTR_PRINT(INFO, "\n");

      HCTR_LOG(INFO, ROOT, "Instance %d model indices OFFSETS: ", global_id);
      for (int j = 0; j < num_procs + 1; j++) {
        if (embedding->embedding_params_.communication_type ==
            CommunicationType::NVLink_SingleNode) {
          HCTR_PRINT(INFO, " %d",
                     (int)embedding->infrequent_embeddings_single_node_[device]
                         .indices_->model_indices_offsets_.get_ptr()[j]);
        }
        if (embedding->embedding_params_.communication_type == CommunicationType::IB_NVLink) {
          HCTR_PRINT(INFO, " %d",
                     (int)embedding->infrequent_embeddings_ib_nvlink_[device]
                         .indices_->model_indices_offsets_.get_ptr()[j]);
        }
        if (embedding->embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
          HCTR_PRINT(INFO, " %d",
                     (int)embedding->infrequent_embeddings_ib_nvlink_hier_[device]
                         .indices_->model_indices_offsets_.get_ptr()[j]);
        }
      }
      HCTR_PRINT(INFO, "\n");

      int num_batch_frequent;
      HCTR_LIB_THROW(cudaMemcpy(&num_batch_frequent,
                                embedding->frequent_embeddings_single_node_[device]
                                    .indices_->d_num_frequent_sample_indices_.get_ptr(),
                                sizeof(uint32_t), cudaMemcpyDeviceToHost));
      HCTR_LOG(INFO, ROOT, "Instance %d found %d frequent categories in positions: ", global_id,
               num_batch_frequent);
      download_tensor(
          tmp,
          embedding->frequent_embeddings_single_node_[device].indices_->frequent_sample_indices_,
          0);
      for (int j = 0; j < num_batch_frequent; j++) {
        HCTR_PRINT(INFO, " %d", (int)tmp[j]);
      }
      HCTR_PRINT(INFO, "\n");
    }

    {
      std::vector<dtype> tmp;
      if (embedding->embedding_params_.communication_type == CommunicationType::NVLink_SingleNode) {
        download_tensor(
            tmp, embedding->infrequent_embeddings_single_node_[device].indices_->network_indices_,
            0);
      }
      if (embedding->embedding_params_.communication_type == CommunicationType::IB_NVLink) {
        download_tensor(
            tmp, embedding->infrequent_embeddings_ib_nvlink_[device].indices_->network_indices_, 0);
      }
      if (embedding->embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
        download_tensor(
            tmp,
            embedding->infrequent_embeddings_ib_nvlink_hier_[device].indices_->network_indices_, 0);
      }

      HCTR_LOG(INFO, ROOT, "Instance %d network indices: ", global_id);
      for (size_t j = 0; j < tmp.size(); j++) {
        HCTR_PRINT(INFO, " %d", (int)tmp[j]);
      }
      HCTR_PRINT(INFO, "\n");

      HCTR_LOG(INFO, ROOT, "Instance %d network indices OFFSETS: ", global_id);
      for (int j = 0; j < num_procs + 1; j++) {
        // HCTR_PRINT(INFO, " %d",
        //(int)embedding->infrequent_embeddings_[device]
        //.indices_->network_indices_offsets_.get_ptr()[j]);

        if (embedding->embedding_params_.communication_type ==
            CommunicationType::NVLink_SingleNode) {
          HCTR_PRINT(INFO, " %d",
                     (int)embedding->infrequent_embeddings_single_node_[device]
                         .indices_->network_indices_offsets_.get_ptr()[j]);
        }
        if (embedding->embedding_params_.communication_type == CommunicationType::IB_NVLink) {
          HCTR_PRINT(INFO, " %d",
                     (int)embedding->infrequent_embeddings_ib_nvlink_[device]
                         .indices_->network_indices_offsets_.get_ptr()[j]);
        }
        if (embedding->embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
          HCTR_PRINT(INFO, " %d",
                     (int)embedding->infrequent_embeddings_ib_nvlink_hier_[device]
                         .indices_->network_indices_offsets_.get_ptr()[j]);
        }
      }
      HCTR_PRINT(INFO, "\n");
    }
  }

  // Check
  for (size_t device = 0; device < local_gpu_count; device++) {
    CudaDeviceContext context(resource_manager->get_local_gpu(device)->get_device_id());
    int global_id = resource_manager->get_local_gpu(device)->get_global_id();
    HCTR_LIB_THROW(cudaDeviceSynchronize());

    std::vector<emtype> h_output;
    std::vector<emtype> expected(embedding_vec_size);
    ASSERT_EQ(local_batch_size, embedding->get_batch_size_per_gpu(true));

    download_tensor(h_output, Tensor2<emtype>::stretch_from(outputs[device]), 0);
    ASSERT_EQ(h_output.size() % embedding_vec_size, 0);
    ASSERT_EQ(h_output.size(), local_batch_size * num_tables * embedding_vec_size);

    for (size_t i = 0; i < h_output.size() / embedding_vec_size; i++) {
      size_t table = i % num_tables;
      size_t cat_id = table_offsets[table] + input[i + global_id * local_batch_size * num_tables];
      auto expected_ptr = full_emb_table.data() + cat_id * embedding_vec_size;
      auto actual_ptr = h_output.data() + i * embedding_vec_size;

      if (debug_print) {
        HCTR_LOG(INFO, ROOT, " Instance %d sample %ld slot %ld comparing category %ld: ", global_id,
                 i, table, cat_id);
        for (size_t j = 0; j < embedding_vec_size; j++) {
          HCTR_PRINT(INFO, " (%8.5f : %8.5f) ", (float)actual_ptr[j], (float)expected_ptr[j]);
        }
        HCTR_PRINT(INFO, "\n");
      }

      for (size_t j = 0; j < embedding_vec_size; j++) {
        expected[j] = (emtype)expected_ptr[j];
      }

      ASSERT_EQ(memcmp(expected.data(), actual_ptr, embedding_vec_size * sizeof(emtype)), 0)
          << "Data mismatch on instance " << global_id << " in sample " << i / num_tables
          << " feature " << table << std::endl;
    }
  }

  //======================================================================================
  // Do the backward step and update
  //======================================================================================
  for (size_t device = 0; device < local_gpu_count; device++) {
    CudaDeviceContext context(resource_manager->get_local_gpu(device)->get_device_id());

    std::vector<emtype> h_output(local_batch_size * num_tables * embedding_vec_size);

    // Per-GPU generator
    std::mt19937 gen(seed + 3 + resource_manager->get_local_gpu(device)->get_global_id());
    std::uniform_real_distribution<float> distr(-1, 1);
    for (auto &grad : h_output) {
      grad = (emtype)distr(gen);
    }
    upload_tensor(h_output, Tensor2<emtype>::stretch_from(outputs[device]), 0);
  }

  // We can't allreduce __half type with MPI, so need to recreate all the output tensors locally.
  std::vector<double> gradients(total_categories * embedding_vec_size, 0);
  for (size_t device = 0; device < total_gpu_count; device++) {
    std::mt19937 gen(seed + 3 + device);
    std::uniform_real_distribution<float> distr(-1, 1);

    for (size_t i = 0; i < local_batch_size * num_tables; i++) {
      size_t table = i % num_tables;
      size_t cat_id = table_offsets[table] + input[i + device * local_batch_size * num_tables];
      auto grad_ptr = gradients.data() + cat_id * embedding_vec_size;

      for (size_t j = 0; j < embedding_vec_size; j++) {
        grad_ptr[j] += distr(gen);
      }
    }
  }

  if (debug_print) {
    HCTR_LOG(INFO, ROOT, "Generated embedding gradients");
    for (size_t i = 0; i < gradients.size(); i++) {
      if (i % embedding_vec_size == 0) {
        HCTR_PRINT(INFO, "\nRank %d cat %ld :: ", rank, i / embedding_vec_size);
      }
      HCTR_PRINT(INFO, "%8.5f ", (float)gradients[i]);
    }
    HCTR_PRINT(INFO, "\n");
  }

  embedding->backward();
  embedding->update_params();

  // Check
  // Check frequent embeddings
  for (size_t device = 0; device < local_gpu_count; device++) {
    CudaDeviceContext context(resource_manager->get_local_gpu(device)->get_device_id());
    int global_id = resource_manager->get_local_gpu(device)->get_global_id();
    HCTR_LIB_THROW(cudaDeviceSynchronize());

    std::vector<dtype> h_frequent_categories;
    download_tensor(h_frequent_categories, embedding->model_[device].frequent_categories, 0);

    float *h_frequent_embedding_vectors;
    HCTR_LIB_THROW(
        cudaMallocHost((void **)&h_frequent_embedding_vectors, embedding_vec_size * sizeof(float)));

    // Only checking the categories that the instance owns
    size_t chunk = num_frequent / resource_manager->get_global_gpu_count();
    ASSERT_EQ(num_frequent % resource_manager->get_global_gpu_count(), 0);

    size_t start = device * chunk;
    size_t end = (device + 1) * chunk;
    for (size_t i = start; i < end; ++i) {
      dtype cat_id = h_frequent_categories[i];
      HCTR_LIB_THROW(cudaMemcpy(h_frequent_embedding_vectors,
                                embedding->frequent_embeddings_single_node_[device]
                                        .frequent_data_.frequent_embedding_vectors_.get_ptr() +
                                    i * embedding_vec_size,
                                sizeof(float) * embedding_vec_size, cudaMemcpyDeviceToHost));
      for (size_t j = 0; j < embedding_vec_size; j++) {
        ASSERT_NEAR((double)h_frequent_embedding_vectors[j],
                    (double)full_emb_table.data()[cat_id * embedding_vec_size + j] -
                        (double)gradients.data()[cat_id * embedding_vec_size + j] * lr,
                    epsilon)
            << "Gradient (frequent) mismatch on instance " << global_id << " in category " << cat_id
            << " dimension " << j << "/" << embedding_vec_size << std::endl;
      }
    }
    HCTR_LIB_THROW(cudaFreeHost(h_frequent_embedding_vectors));
  }

  // Check infrequent embeddings
  for (size_t device = 0; device < local_gpu_count; device++) {
    CudaDeviceContext context(resource_manager->get_local_gpu(device)->get_device_id());
    int global_id = resource_manager->get_local_gpu(device)->get_global_id();

    size_t num_infrequent = embedding->model_[device].h_infrequent_model_table_offsets[num_tables];

    float *h_infrequent_embedding_vectors;
    dtype *h_category_location;
    HCTR_LIB_THROW(cudaMallocHost((void **)&h_infrequent_embedding_vectors,
                                  num_infrequent * embedding_vec_size * sizeof(float)));
    HCTR_LIB_THROW(
        cudaMallocHost((void **)&h_category_location, total_categories * 2 * sizeof(dtype)));

    HCTR_LIB_THROW(cudaMemcpy(h_category_location,
                              embedding->model_[device].category_location.get_ptr(),
                              total_categories * 2 * sizeof(dtype), cudaMemcpyDeviceToHost));

    // if (embedding_params_.)
    // cudaMemcpy(h_infrequent_embedding_vectors,
    // embedding->infrequent_embeddings_[device].infrequent_embedding_vectors_.get_ptr(),
    // num_infrequent * embedding_vec_size * sizeof(float), cudaMemcpyDeviceToHost);

    if (embedding->embedding_params_.communication_type == CommunicationType::NVLink_SingleNode) {
      cudaMemcpy(h_infrequent_embedding_vectors,
                 embedding->infrequent_embeddings_single_node_[device]
                     .infrequent_embedding_vectors_.get_ptr(),
                 num_infrequent * embedding_vec_size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    if (embedding->embedding_params_.communication_type == CommunicationType::IB_NVLink) {
      cudaMemcpy(h_infrequent_embedding_vectors,
                 embedding->infrequent_embeddings_ib_nvlink_[device]
                     .infrequent_embedding_vectors_.get_ptr(),
                 num_infrequent * embedding_vec_size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    if (embedding->embedding_params_.communication_type == CommunicationType::IB_NVLink_Hier) {
      cudaMemcpy(h_infrequent_embedding_vectors,
                 embedding->infrequent_embeddings_ib_nvlink_hier_[device]
                     .infrequent_embedding_vectors_.get_ptr(),
                 num_infrequent * embedding_vec_size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    for (size_t cat_id = 0; cat_id < total_categories; ++cat_id) {
      if ((int)h_category_location[2 * cat_id] == global_id) {
        auto local_cat_id = h_category_location[2 * cat_id + 1];

        for (size_t j = 0; j < embedding_vec_size; j++) {
          ASSERT_NEAR((double)h_infrequent_embedding_vectors[local_cat_id * embedding_vec_size + j],
                      (double)full_emb_table.data()[cat_id * embedding_vec_size + j] -
                          (double)gradients.data()[cat_id * embedding_vec_size + j] * lr,
                      epsilon)
              << "Gradient (infrequent) mismatch on instance " << global_id << " in category "
              << cat_id << " dimension " << j << "/" << embedding_vec_size << std::endl;
        }
      }
    }

    HCTR_LIB_THROW(cudaFreeHost(h_infrequent_embedding_vectors));
    HCTR_LIB_THROW(cudaFreeHost(h_category_location));
  }
}

template <typename dtype, typename emtype>
void end_to_end(std::vector<int> device_list, size_t num_tables, size_t total_categories,
                size_t batch_size, size_t embedding_vec_size, double bw_ratio_a2a_over_ar,
                size_t seed = 42, size_t num_evals = 1) {
  int num_procs = 1;
#ifdef ENABLE_MPI
  HCTR_MPI_THROW(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
#endif
  size_t num_total_gpus = num_procs * device_list.size();

  HybridEmbeddingConfig<dtype> test_config = {
      (size_t)num_procs,
      num_total_gpus,
      num_tables,
      embedding_vec_size,
      (dtype)total_categories,
      (dtype)0,  // irrelevent here
      1.0f,      // irrelevent here
      num_procs == 1 ? hybrid_embedding::CommunicationType::NVLink_SingleNode
                     : hybrid_embedding::CommunicationType::IB_NVLink,
  };

  auto generator = std::make_unique<HybridEmbeddingInputGenerator<dtype>>(test_config, seed + 1);
  end_to_end_impl<dtype, emtype>(device_list, generator.get(), batch_size, embedding_vec_size,
                                 bw_ratio_a2a_over_ar, seed, num_evals);
}

template <typename dtype, typename emtype>
void end_to_end(std::vector<int> device_list, std::vector<size_t> table_sizes, size_t batch_size,
                size_t embedding_vec_size, double bw_ratio_a2a_over_ar, size_t seed = 42,
                size_t num_evals = 1) {
  int num_procs = 1;
#ifdef ENABLE_MPI
  HCTR_MPI_THROW(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
#endif
  size_t num_total_gpus = num_procs * device_list.size();

  HybridEmbeddingConfig<dtype> test_config = {
      (size_t)num_procs,
      num_total_gpus,
      0,  // irrelevent here
      embedding_vec_size,
      0,         // irrelevent here
      (dtype)0,  // irrelevent here
      1.0f,      // irrelevent here
      num_procs == 1 ? hybrid_embedding::CommunicationType::NVLink_SingleNode
                     : hybrid_embedding::CommunicationType::IB_NVLink,
  };

  auto generator =
      std::make_unique<HybridEmbeddingInputGenerator<dtype>>(test_config, table_sizes, seed + 1);
  end_to_end_impl<dtype, emtype>(device_list, generator.get(), batch_size, embedding_vec_size,
                                 bw_ratio_a2a_over_ar, seed, num_evals);
}

class MPIEnvironment : public ::testing::Environment {
 protected:
  virtual void SetUp() { test::mpi_init(); }
  virtual void TearDown() { test::mpi_finalize(); }
  virtual ~MPIEnvironment(){};
};

::testing::Environment *const mpi_env = ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
//
TEST(hybrid_e2e, test1) { end_to_end<uint32_t, float>({0}, 2, 16, 20, 2, 1.0e10, global_seed); }
TEST(hybrid_e2e, test2) { end_to_end<uint32_t, float>({0}, 2, 16, 20, 2, 1.0e-10, global_seed++); }
TEST(hybrid_e2e, test3) {
  end_to_end<uint32_t, float>({0, 1}, 2, 128, 20, 2, 1.0e10, global_seed++);
}
TEST(hybrid_e2e, test4) {
  end_to_end<uint32_t, float>({0, 1}, 2, 128, 20, 2, 1.0e-10, global_seed++);
}
TEST(hybrid_e2e, test5) { end_to_end<uint32_t, float>({0, 1}, 2, 128, 20, 2, 1.0, global_seed++); }
TEST(hybrid_e2e, test6) { end_to_end<uint32_t, float>({0, 1}, 7, 128, 20, 2, 1.0, global_seed++); }
TEST(hybrid_e2e, test7) {
  end_to_end<uint32_t, float>({0, 1, 2}, 3, 192, 96, 5, 1.0, global_seed++);
}
TEST(hybrid_e2e, test8) {
  end_to_end<uint32_t, float>({0, 1, 2, 3}, 6, 651, 96, 128, 1.5, global_seed++);
}
TEST(hybrid_e2e, test9) {
  end_to_end<uint32_t, float>({0, 1, 2, 3}, 18, 6531, 256, 64, 1.7, global_seed++);
}
TEST(hybrid_e2e, test10) {
  end_to_end<uint32_t, float>({0, 1, 2, 3, 4, 5, 6, 7}, 18, 6531, 256, 64, 1.7, global_seed++);
}
TEST(hybrid_e2e, test11) {
  end_to_end<uint32_t, float>({0, 1, 2, 3, 4, 5, 6, 7}, 26, 16531, 512, 48, 1.33, global_seed++);
}
TEST(hybrid_e2e, test12) {
  end_to_end<uint32_t, float>({0, 1, 6, 7}, 13, 21345, 256, 32, 0.6, global_seed++);
}
TEST(hybrid_e2e, test13) {
  std::vector<size_t> slot_size_array{
      39884406, 39043,    17289,    7420,     20263,  3,     7120, 1543, 63,
      38532951, 2953546,  403346,   10,       2208,   11938, 155,  4,    976,
      14,       39979771, 25641295, 39664984, 585935, 12972, 108,  36};
  // for (auto& s : slot_size_array) {
  //   s = s/16 + 1;
  // }

  end_to_end<uint32_t, float>({0, 1, 2, 3, 4, 5, 6, 7}, slot_size_array, 1024, 128, 1.9 / 1.3,
                              global_seed++);
}

TEST(hybrid_e2e, test21) { end_to_end<uint32_t, __half>({0}, 2, 16, 20, 2, 1.0e10, global_seed++); }
TEST(hybrid_e2e, test22) {
  end_to_end<uint32_t, __half>({0}, 2, 16, 20, 2, 1.0e-10, global_seed++);
}
TEST(hybrid_e2e, test23) {
  end_to_end<uint32_t, __half>({0, 1}, 2, 128, 20, 2, 1.0e10, global_seed++);
}
TEST(hybrid_e2e, test24) {
  end_to_end<uint32_t, __half>({0, 1}, 2, 128, 20, 2, 1.0e-10, global_seed++);
}
TEST(hybrid_e2e, test25) {
  end_to_end<uint32_t, __half>({0, 1}, 2, 128, 20, 2, 1.0, global_seed++);
}
TEST(hybrid_e2e, test26) {
  end_to_end<uint32_t, __half>({0, 1}, 7, 128, 20, 2, 1.0, global_seed++);
}
TEST(hybrid_e2e, test27) {
  end_to_end<uint32_t, __half>({0, 1, 2}, 3, 192, 96, 5, 1.0, global_seed++);
}
TEST(hybrid_e2e, test28) {
  end_to_end<uint32_t, __half>({0, 1, 2, 3}, 6, 651, 96, 128, 1.5, global_seed++);
}
TEST(hybrid_e2e, test29) {
  end_to_end<uint32_t, __half>({0, 1, 2, 3}, 18, 6531, 256, 64, 1.7, global_seed++);
}
TEST(hybrid_e2e, test30) {
  end_to_end<uint32_t, __half>({0, 1, 2, 3, 4, 5, 6, 7}, 18, 6531, 256, 64, 1.7, global_seed++);
}
TEST(hybrid_e2e, test31) {
  end_to_end<uint32_t, __half>({0, 1, 2, 3, 4, 5, 6, 7}, 26, 16531, 512, 48, 1.33, global_seed++);
}
TEST(hybrid_e2e, test32) {
  end_to_end<uint32_t, __half>({0, 1, 6, 7}, 13, 21345, 256, 32, 0.6, global_seed++);
}
TEST(hybrid_e2e, test33) {
  std::vector<size_t> slot_size_array{
      39884406, 39043,    17289,    7420,     20263,  3,     7120, 1543, 63,
      38532951, 2953546,  403346,   10,       2208,   11938, 155,  4,    976,
      14,       39979771, 25641295, 39664984, 585935, 12972, 108,  36};
  // for (auto& s : slot_size_array) {
  //   s = s/16 + 1;
  // }

  end_to_end<uint32_t, float>({0, 1, 2, 3, 4, 5, 6, 7}, slot_size_array, 1024, 128, 1.9 / 1.3,
                              global_seed++);
}
