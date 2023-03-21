/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <pybind11/pybind11.h>

#include <training_callback.hpp>

namespace pybind11::detail {

using TrainingCallback = HugeCTR::TrainingCallback;

template <>
struct type_caster<std::shared_ptr<TrainingCallback>> {
  PYBIND11_TYPE_CASTER(std::shared_ptr<TrainingCallback>, _("TrainingCallback"));

  using BaseCaster = copyable_holder_caster<TrainingCallback, std::shared_ptr<TrainingCallback>>;

  bool load(pybind11::handle src, bool b) {
    BaseCaster bc;
    bool success = bc.load(src, b);
    if (!success) {
      return false;
    }

    auto py_obj = pybind11::reinterpret_borrow<pybind11::object>(src);
    auto base_ptr = static_cast<std::shared_ptr<TrainingCallback>>(bc);

    auto py_obj_ptr = std::shared_ptr<object>{new object{py_obj}, [](auto py_obj_ptr) {
                                                gil_scoped_acquire gil;
                                                delete py_obj_ptr;
                                              }};
    value = std::shared_ptr<TrainingCallback>(py_obj_ptr, base_ptr.get());
    return true;
  }

  static handle cast(std::shared_ptr<TrainingCallback> training_callback, return_value_policy rvp,
                     handle h) {
    return BaseCaster::cast(training_callback, rvp, h);
  }
};

template <>
struct is_holder_type<TrainingCallback, std::shared_ptr<TrainingCallback>> : std::true_type {};

}  // namespace pybind11::detail

namespace HugeCTR {

namespace python_lib {
struct PyTrainingCallback : public TrainingCallback {
  using TrainingCallback::TrainingCallback;

  void on_training_start() override {
    PYBIND11_OVERRIDE_PURE(void, TrainingCallback, on_training_start);
  }
  void on_training_end(int current_iter) override {
    PYBIND11_OVERRIDE_PURE(void, TrainingCallback, on_training_end, current_iter);
  }
  bool on_eval_start(int current_iter) override {
    PYBIND11_OVERRIDE_PURE(bool, TrainingCallback, on_eval_start, current_iter);
  }
  bool on_eval_end(int current_iter, const std::map<std::string, float>& eval_results) override {
    PYBIND11_OVERRIDE_PURE(bool, TrainingCallback, on_eval_end, current_iter, eval_results);
  }
};

void TrainingCallbackPybind(pybind11::module& m) {
  pybind11::class_<TrainingCallback, PyTrainingCallback, std::shared_ptr<TrainingCallback>>
      training_callback_wrap(m, "TrainingCallback");
  training_callback_wrap.def(pybind11::init<>());
}

}  // namespace python_lib

}  // namespace HugeCTR