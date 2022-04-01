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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pybind/model.hpp>

namespace HugeCTR {

namespace python_lib {

void ModelPybind(pybind11::module &m) {
  pybind11::class_<HugeCTR::DataReaderParams, std::shared_ptr<HugeCTR::DataReaderParams>>(
      m, "DataReaderParams")
      .def(pybind11::init<DataReaderType_t, std::string, std::string, std::string, Check_t, int,
                          long long, long long, bool, int, std::vector<long long> &,
                          const AsyncParam &>(),
           pybind11::arg("data_reader_type"), pybind11::arg("source"), pybind11::arg("keyset") = "",
           pybind11::arg("eval_source"), pybind11::arg("check_type"),
           pybind11::arg("cache_eval_data") = 0, pybind11::arg("num_samples") = 0,
           pybind11::arg("eval_num_samples") = 0, pybind11::arg("float_label_dense") = false,
           pybind11::arg("num_workers") = 12,
           pybind11::arg("slot_size_array") = std::vector<long long>(),
           pybind11::arg("async_param") =
               AsyncParam{16, 4, 512000, 4, 512, false, Alignment_t::None})
      .def(pybind11::init<DataReaderType_t, std::vector<std::string>, std::vector<std::string>,
                          std::string, Check_t, int, long long, long long, bool, int,
                          std::vector<long long> &, const AsyncParam &>(),
           pybind11::arg("data_reader_type"), pybind11::arg("source"),
           pybind11::arg("keyset") = std::vector<std::string>(), pybind11::arg("eval_source"),
           pybind11::arg("check_type"), pybind11::arg("cache_eval_data") = 0,
           pybind11::arg("num_samples") = 0, pybind11::arg("eval_num_samples") = 0,
           pybind11::arg("float_label_dense") = false, pybind11::arg("num_workers") = 12,
           pybind11::arg("slot_size_array") = std::vector<long long>(),
           pybind11::arg("async_param") =
               AsyncParam{16, 4, 512000, 4, 512, false, Alignment_t::None});
  pybind11::class_<HugeCTR::Input, std::shared_ptr<HugeCTR::Input>>(m, "Input")
      .def(pybind11::init<int, std::string, int, std::string,
                          std::vector<DataReaderSparseParam> &>(),
           pybind11::arg("label_dim"), pybind11::arg("label_name"), pybind11::arg("dense_dim"),
           pybind11::arg("dense_name"), pybind11::arg("data_reader_sparse_param_array"))
      .def(pybind11::init<std::vector<int>, std::vector<std::string>, int, std::string,
                          std::vector<DataReaderSparseParam> &>(),
           pybind11::arg("label_dims"), pybind11::arg("label_names"), pybind11::arg("dense_dim"),
           pybind11::arg("dense_name"), pybind11::arg("data_reader_sparse_param_array"))
      .def(pybind11::init<std::vector<int>, std::vector<std::string>, std::vector<float>, int,
                          std::string, std::vector<DataReaderSparseParam> &>(),
           pybind11::arg("label_dims"), pybind11::arg("label_names"),
           pybind11::arg("label_weights"), pybind11::arg("dense_dim"), pybind11::arg("dense_name"),
           pybind11::arg("data_reader_sparse_param_array"));
  pybind11::class_<HugeCTR::SparseEmbedding, std::shared_ptr<HugeCTR::SparseEmbedding>>(
      m, "SparseEmbedding")
      .def(pybind11::init<Embedding_t, size_t, size_t, const std::string &, std::string,
                          std::string, std::vector<size_t> &, std::shared_ptr<OptParamsPy> &,
                          const HybridEmbeddingParam &>(),
           pybind11::arg("embedding_type"), pybind11::arg("workspace_size_per_gpu_in_mb"),
           pybind11::arg("embedding_vec_size"), pybind11::arg("combiner"),
           pybind11::arg("sparse_embedding_name"), pybind11::arg("bottom_name"),
           pybind11::arg("slot_size_array") = std::vector<size_t>(),
           pybind11::arg("optimizer") = std::shared_ptr<OptParamsPy>(new OptParamsPy()),
           pybind11::arg("hybrid_embedding_param") =
               HybridEmbeddingParam{1, -1, 0.01, 1.3e11, 1.9e11, 1.0, false, false,
                                    hybrid_embedding::CommunicationType::NVLink_SingleNode,
                                    hybrid_embedding::HybridEmbeddingType::Distributed});
  pybind11::class_<HugeCTR::DenseLayer, std::shared_ptr<HugeCTR::DenseLayer>>(m, "DenseLayer")
      .def(
          pybind11::init<Layer_t, std::vector<std::string> &, std::vector<std::string> &, float,
                         float, Initializer_t, Initializer_t, float, float, size_t, Initializer_t,
                         Initializer_t, int, size_t, size_t, size_t, size_t, size_t, bool,
                         std::vector<int> &, std::vector<std::pair<int, int>> &, std::vector<int> &,
                         std::vector<size_t> &, size_t, int, std::vector<float> &, bool,
                         Regularizer_t, float, FcPosition_t, Activation_t>(),
          pybind11::arg("layer_type"), pybind11::arg("bottom_names"), pybind11::arg("top_names"),
          pybind11::arg("factor") = 1.0, pybind11::arg("eps") = 0.00001,
          pybind11::arg("gamma_init_type") = Initializer_t::Default,
          pybind11::arg("beta_init_type") = Initializer_t::Default,
          pybind11::arg("dropout_rate") = 0.5, pybind11::arg("elu_alpha") = 1.0,
          pybind11::arg("num_output") = 1,
          pybind11::arg("weight_init_type") = Initializer_t::Default,
          pybind11::arg("bias_init_type") = Initializer_t::Default, pybind11::arg("num_layers") = 0,
          pybind11::arg("leading_dim") = 1, pybind11::arg("time_step") = 0,
          pybind11::arg("batchsize") = 1, pybind11::arg("SeqLength") = 1,
          pybind11::arg("vector_size") = 1, pybind11::arg("selected") = false,
          pybind11::arg("selected_slots") = std::vector<int>(),
          pybind11::arg("ranges") = std::vector<std::pair<int, int>>(),
          pybind11::arg("indices") = std::vector<int>(),
          pybind11::arg("weight_dims") = std::vector<size_t>(), pybind11::arg("out_dim") = 0,
          pybind11::arg("axis") = 1, pybind11::arg("target_weight_vec") = std::vector<float>(),
          pybind11::arg("use_regularizer") = false,
          pybind11::arg("regularizer_type") = Regularizer_t::L1, pybind11::arg("lambda") = 0,
          pybind11::arg("pos_type") = FcPosition_t::None,
          pybind11::arg("act_type") = Activation_t::Relu);
  pybind11::class_<HugeCTR::GroupDenseLayer, std::shared_ptr<HugeCTR::GroupDenseLayer>>(
      m, "GroupDenseLayer")
      .def(pybind11::init<GroupLayer_t, std::vector<std::string> &, std::vector<std::string> &,
                          std::vector<size_t> &, Activation_t>(),
           pybind11::arg("group_layer_type"), pybind11::arg("bottom_name_list"),
           pybind11::arg("top_name_list"), pybind11::arg("num_outputs"),
           pybind11::arg("last_act_type") = Activation_t::Relu);
  pybind11::class_<HugeCTR::Model, std::shared_ptr<HugeCTR::Model>>(m, "Model")
      .def(pybind11::init<const Solver &, const DataReaderParams &, std::shared_ptr<OptParamsPy> &,
                          std::shared_ptr<EmbeddingTrainingCacheParams> &>(),
           pybind11::arg("solver"), pybind11::arg("reader_params"), pybind11::arg("opt_params"),
           pybind11::arg("etc_params") =
               std::shared_ptr<EmbeddingTrainingCacheParams>(new EmbeddingTrainingCacheParams()))
      .def("compile", pybind11::overload_cast<>(&HugeCTR::Model::compile))
      .def("compile",
           pybind11::overload_cast<std::vector<std::string> &, std::vector<float> &>(
               &HugeCTR::Model::compile),
           pybind11::arg("loss_names"), pybind11::arg("loss_weights"))
      .def("summary", &HugeCTR::Model::summary)
      .def("fit", &HugeCTR::Model::fit, pybind11::arg("num_epochs") = 0,
           pybind11::arg("max_iter") = 2000, pybind11::arg("display") = 200,
           pybind11::arg("eval_interval") = 1000, pybind11::arg("snapshot") = 10000,
           pybind11::arg("snapshot_prefix") = "")
      .def("set_source",
           pybind11::overload_cast<std::vector<std::string>, std::vector<std::string>, std::string>(
               &HugeCTR::Model::set_source),
           pybind11::arg("source"), pybind11::arg("keyset"), pybind11::arg("eval_source"))
      .def("set_source",
           pybind11::overload_cast<std::string, std::string>(&HugeCTR::Model::set_source),
           pybind11::arg("source"), pybind11::arg("eval_source"))
      .def("graph_to_json", &HugeCTR::Model::graph_to_json, pybind11::arg("graph_config_file"))
      .def("construct_from_json", &HugeCTR::Model::construct_from_json,
           pybind11::arg("graph_config_file"), pybind11::arg("include_dense_network"))
      .def("reset_learning_rate_scheduler", &HugeCTR::Model::reset_learning_rate_scheduler,
           pybind11::arg("base_lr"), pybind11::arg("warmup_steps") = 1,
           pybind11::arg("decay_start") = 0, pybind11::arg("decay_steps") = 1,
           pybind11::arg("decay_power") = 2.f, pybind11::arg("end_lr") = 0.f)
      .def("freeze_embedding", pybind11::overload_cast<>(&HugeCTR::Model::freeze_embedding))
      .def("freeze_embedding",
           pybind11::overload_cast<const std::string &>(&HugeCTR::Model::freeze_embedding),
           pybind11::arg("embedding_name"))
      .def("freeze_dense", &HugeCTR::Model::freeze_dense)
      .def("unfreeze_embedding", pybind11::overload_cast<>(&HugeCTR::Model::unfreeze_embedding))
      .def("unfreeze_embedding",
           pybind11::overload_cast<const std::string &>(&HugeCTR::Model::unfreeze_embedding),
           pybind11::arg("embedding_name"))
      .def("unfreeze_dense", &HugeCTR::Model::unfreeze_dense)
      .def("load_dense_weights", &HugeCTR::Model::load_dense_weights,
           pybind11::arg("dense_model_file"))
      .def("load_sparse_weights",
           pybind11::overload_cast<const std::vector<std::string> &>(
               &HugeCTR::Model::load_sparse_weights),
           pybind11::arg("sparse_embedding_files"))
      .def("load_sparse_weights",
           pybind11::overload_cast<const std::map<std::string, std::string> &>(
               &HugeCTR::Model::load_sparse_weights),
           pybind11::arg("sparse_embedding_files_map"))
      .def("load_dense_optimizer_states", &HugeCTR::Model::load_dense_optimizer_states,
           pybind11::arg("dense_opt_states_file"))
      .def("load_sparse_optimizer_states",
           pybind11::overload_cast<const std::vector<std::string> &>(
               &HugeCTR::Model::load_sparse_optimizer_states),
           pybind11::arg("sparse_opt_states_files"))
      .def("load_sparse_optimizer_states",
           pybind11::overload_cast<const std::map<std::string, std::string> &>(
               &HugeCTR::Model::load_sparse_optimizer_states),
           pybind11::arg("sparse_opt_states_files_map"))
      .def("add", pybind11::overload_cast<Input &>(&HugeCTR::Model::add), pybind11::arg("input"))
      .def("add", pybind11::overload_cast<SparseEmbedding &>(&HugeCTR::Model::add),
           pybind11::arg("sparse_embedding"))
      .def("add", pybind11::overload_cast<DenseLayer &>(&HugeCTR::Model::add),
           pybind11::arg("dense_layer"))
      .def("add", pybind11::overload_cast<GroupDenseLayer &>(&HugeCTR::Model::add),
           pybind11::arg("group_dense_layer"))
      .def("set_learning_rate", &HugeCTR::Model::set_learning_rate, pybind11::arg("lr"))
      .def("train", &HugeCTR::Model::train, pybind11::arg("is_first_batch") = true)
      .def("eval", &HugeCTR::Model::eval, pybind11::arg("is_first_batch") = true)
      .def("start_data_reading", &HugeCTR::Model::start_data_reading)
      .def("get_current_loss",
           [](HugeCTR::Model &self) {
             float loss = 0;
             self.get_current_loss(&loss);
             return loss;
           })
      .def("get_eval_metrics", &HugeCTR::Model::get_eval_metrics)
      .def("get_incremental_model",
           [](HugeCTR::Model &self) {
             auto inc_sparse_model = self.get_incremental_model();
             std::vector<std::pair<pybind11::array_t<long long>, pybind11::array_t<float>>>
                 array_inc_sparse_model;
             for (const auto &pair : inc_sparse_model) {
               size_t num_keys = pair.first.size();
               size_t emb_vec_size = pair.second.size() / num_keys;
               long long *keys = new long long[num_keys];
               float *emb_vecs = new float[pair.second.size()];
               memcpy(keys, pair.first.data(), num_keys * sizeof(long long));
               memcpy(emb_vecs, pair.second.data(), num_keys * emb_vec_size * sizeof(float));
               auto keys_capsule = pybind11::capsule(keys, [](void *v) {
                 long long *vv = reinterpret_cast<long long *>(v);
                 delete[] vv;
               });
               auto emb_vecs_capsule = pybind11::capsule(emb_vecs, [](void *v) {
                 float *vv = reinterpret_cast<float *>(v);
                 delete[] vv;
               });
               array_inc_sparse_model.push_back(std::make_pair(
                   pybind11::array_t<long long>(num_keys, keys, keys_capsule),
                   pybind11::array_t<float>({num_keys, emb_vec_size}, emb_vecs, emb_vecs_capsule)));
             }
             inc_sparse_model.clear();
             return array_inc_sparse_model;
           })
      .def("dump_incremental_model_2kafka", &HugeCTR::Model::dump_incremental_model_2kafka)
      .def("save_params_to_files", &HugeCTR::Model::download_params_to_files,
           pybind11::arg("prefix"), pybind11::arg("iter") = 0)
      .def("get_embedding_training_cache", &HugeCTR::Model::get_embedding_training_cache)
      .def("get_data_reader_train", &HugeCTR::Model::get_train_data_reader)
      .def("get_data_reader_eval", &HugeCTR::Model::get_evaluate_data_reader)
      .def("get_learning_rate_scheduler", &HugeCTR::Model::get_learning_rate_scheduler)
      .def("export_predictions", &HugeCTR::Model::export_predictions,
           pybind11::arg("output_prediction_file_name"), pybind11::arg("output_label_file_name"));
}

}  // namespace python_lib

}  // namespace HugeCTR
