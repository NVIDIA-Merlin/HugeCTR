"""
 Copyright (c) 2022, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import tensorflow as tf
from tensorflow.python.framework import ops

from sparse_operation_kit.experiment.dynamic_variable import DynamicVariable


def OptimizerWrapper(optimizer):
    if isinstance(optimizer, tf.keras.optimizers.Optimizer):
        return OptimizerWrapperV2(optimizer)
    else:
        return OptimizerWrapperV1(optimizer)


class OptimizerWrapperV1(object):
    def __init__(self, optimizer):
        self._optimizer = optimizer
        # slots
        unused = tf.Variable([0.0], dtype=tf.float32, name="unused", trainable=False)
        self._optimizer._create_slots([unused])
        names, slots = [], []
        for name in self._optimizer.get_slot_names():
            names.append(name)
            slots.append(self._optimizer.get_slot(unused, name))
        unused_key = self._var_key(unused)
        for name in names:
            assert unused_key in self._optimizer._slots[name]
            self._optimizer._slots[name].pop(unused_key)
        self._initial_vals = {}
        for i, name in enumerate(names):
            self._initial_vals[name] = slots[i]
        # non-slots
        self._optimizer._prepare()
        self._non_slot_dict = {}
        for name, v in self._optimizer._non_slot_dict.items():
            self._non_slot_dict[name] = tf.Variable(v)

    def _var_key(self, var):
        if hasattr(var, "op"):
            return (var.op.graph, var.op.name)
        return var._unique_id

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        # 1. Create slots and do sparse_read
        to_static_ops = []
        grad_list, var_list = [], []
        for g, v in grads_and_vars:
            if g is not None:
                unique, indices = tf.unique(g.indices)
                grad_list.append(ops.IndexedSlices(g.values, indices, g.dense_shape))
                # TODO: Check multi-thread safety of DET
                # with tf.control_dependencies([g.values]):
                to_static_ops.append(v.to_static(unique))
                var_list.append(v)
                key = self._var_key(v)
                for slot_name in self._initial_vals:
                    if key not in self._optimizer._slots[slot_name]:
                        slot = DynamicVariable(
                            dimension=v.dimension,
                            initializer=self._initial_vals[slot_name],
                            name="DynamicSlot",
                            trainable=False,
                        )
                        self._optimizer._slots[slot_name][key] = slot
                    else:
                        slot = self._optimizer._slots[slot_name][key]
                    to_static_ops.append(slot.to_static(unique))

        if len(grad_list) == 0:
            return

        # 2. Switch non_slot_dict
        non_slot_dict = self._optimizer._non_slot_dict
        self._optimizer._non_slot_dict = self._non_slot_dict

        # 3. Call tf-optimizer
        with tf.control_dependencies(to_static_ops):
            train_op = self._optimizer.apply_gradients(
                zip(grad_list, var_list), global_step=global_step, name=name
            )

        # 4. Switch non_slot_dict
        self._optimizer._non_slot_dict = non_slot_dict

        # 5. Write buffer back to dynamic variables
        to_dynamic_ops = []
        with tf.control_dependencies([train_op]):
            for v in var_list:
                key = self._var_key(v)
                to_dynamic_ops.append(v.to_dynamic())
                for name in self._initial_vals:
                    slot = self._optimizer._slots[name][key]
                    to_dynamic_ops.append(slot.to_dynamic())

        return tf.group(to_dynamic_ops)


class OptimizerWrapperV2(object):
    def __init__(self, optimizer):
        self._optimizer = optimizer
        # slots
        if tf.__version__[0] == "1":
            unused = tf.Variable([0.0], name="unused", trainable=False, use_resource=True)
        else:
            unused = tf.Variable([0.0], name="unused", trainable=False)
        self._optimizer._create_slots([unused])
        names, slots = [], []
        for name in self._optimizer.get_slot_names():
            names.append(name)
            slots.append(self._optimizer.get_slot(unused, name))
        unused_key = self._var_key(unused)
        if unused_key in self._optimizer._slots:
            self._optimizer._slots.pop(unused_key)
        self._initial_vals = {}
        for i, name in enumerate(names):
            self._initial_vals[name] = slots[i]
        self._iterations = tf.Variable(0)

    @property
    def lr(self):
        return self._optimizer.lr

    def _var_key(self, var):
        if hasattr(var, "_distributed_container"):
            var = var._distributed_container()
        if var._in_graph_mode:
            return var._shared_name
        return var._unique_id

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        # 1. Create slots and do sparse_read
        to_static_ops = []
        grad_list, var_list = [], []
        for g, v in grads_and_vars:
            if g is not None:
                unique, indices = tf.unique(g.indices)
                grad_list.append(ops.IndexedSlices(g.values, indices, g.dense_shape))
                # TODO: Check multi-thread safety of DET
                # with tf.control_dependencies([g.values]):
                to_static_ops.append(v.to_static(unique))
                var_list.append(v)
                key = self._var_key(v)
                if key not in self._optimizer._slots:
                    self._optimizer._slots[key] = {}
                for slot_name in self._initial_vals:
                    if slot_name not in self._optimizer._slots[key]:
                        slot = DynamicVariable(
                            dimension=v.dimension,
                            initializer=self._initial_vals[slot_name],
                            name="DynamicSlot",
                            trainable=False,
                        )
                        self._optimizer._slots[key][slot_name] = slot
                    else:
                        slot = self._optimizer._slots[key][slot_name]
                    to_static_ops.append(slot.to_static(unique))

        if len(grad_list) == 0:
            return

        # 2. Switch iterations
        iterations = self._optimizer._iterations
        self._optimizer._iterations = self._iterations

        # 3. Call tf-optimizer
        with tf.control_dependencies(to_static_ops):
            train_op = self._optimizer.apply_gradients(zip(grad_list, var_list), name=name)

        # 4. Switch iterations
        self._optimizer._iterations = iterations

        # 5. Write buffer back to dynamic variables
        to_dynamic_ops = []
        with tf.control_dependencies([train_op]):
            for v in var_list:
                key = self._var_key(v)
                to_dynamic_ops.append(v.to_dynamic())
                for name in self._initial_vals:
                    slot = self._optimizer._slots[key][name]
                    to_dynamic_ops.append(slot.to_dynamic())

        return tf.group(to_dynamic_ops)


class SGD(object):
    def __init__(self, lr):
        self._lr = tf.Variable(lr)

    @property
    def lr(self):
        return self._lr

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        train_ops = []
        for g, v in grads_and_vars:
            if g is not None:
                scaled_g = ops.IndexedSlices(g.values * self._lr, g.indices, g.dense_shape)
                train_ops.append(v.scatter_sub(scaled_g))
        return tf.group(train_ops)
