/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <parser.hpp>

namespace HugeCTR {

std::unique_ptr<LearningRateScheduler> get_learning_rate_scheduler__(const nlohmann::json& j_hparam,
                                                                     float base_lr) {
  auto warmup_steps = get_value_from_json_soft<size_t>(j_hparam, "warmup_steps", 1);
  auto decay_start = get_value_from_json_soft<size_t>(j_hparam, "decay_start", 0);
  auto decay_steps = get_value_from_json_soft<size_t>(j_hparam, "decay_steps", 1);
  auto decay_power = get_value_from_json_soft<float>(j_hparam, "decay_power", 2.f);
  auto end_lr = get_value_from_json_soft<float>(j_hparam, "end_lr", 0.f);
  std::unique_ptr<LearningRateScheduler> lr_sch(new LearningRateScheduler(
      base_lr, warmup_steps, decay_start, decay_steps, decay_power, end_lr));

  return lr_sch;
}

std::unique_ptr<LearningRateScheduler> get_learning_rate_scheduler(
    const std::string configure_file) {

  /* file read to json */
  nlohmann::json config;
  std::ifstream file_stream(configure_file);
  if (!file_stream.is_open()) {
    CK_THROW_(Error_t::FileCannotOpen, "file_stream.is_open() failed: " + configure_file);
  }
  file_stream >> config;
  file_stream.close();

  /* parse the solver */
  auto j_optimizer = get_json(config, "optimizer");

  auto optimizer_name = get_value_from_json<std::string>(j_optimizer, "type");
  Optimizer_t optimizer_type;
  if (!find_item_in_map(optimizer_type, optimizer_name, OPTIMIZER_TYPE_MAP)) {
    CK_THROW_(Error_t::WrongInput, "No such optimizer: " + optimizer_name);
  }

  float lr = 0;
  nlohmann::json j_hparam;
  switch (optimizer_type) {
    case Optimizer_t::Adam: {
      j_hparam = get_json(j_optimizer, "adam_hparam");
      lr = get_value_from_json<float>(j_hparam, "learning_rate");
      break;
    }
    case Optimizer_t::AdaGrad: {
      j_hparam = get_json(j_optimizer, "adagrad_hparam");
      lr = get_value_from_json<float>(j_hparam, "learning_rate");
      break;
    }
    case Optimizer_t::MomentumSGD: {
      j_hparam = get_json(j_optimizer, "momentum_sgd_hparam");
      lr = get_value_from_json<float>(j_hparam, "learning_rate");
      break;
    }
    case Optimizer_t::Nesterov: {
      j_hparam = get_json(j_optimizer, "nesterov_hparam");
      lr = get_value_from_json<float>(j_hparam, "learning_rate");
      break;
    }
    case Optimizer_t::SGD: {
      j_hparam = get_json(j_optimizer, "sgd_hparam");
      lr = get_value_from_json<float>(j_hparam, "learning_rate");
      break;
    }
    default:
      assert(!"Error: no such optimizer && should never get here!");
  }
  return get_learning_rate_scheduler__(j_hparam, lr);
}

}  // namespace HugeCTR
