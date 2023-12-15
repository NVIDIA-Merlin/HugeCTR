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

import time
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

import sparse_operation_kit as sok


if __name__ == "__main__":
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
    sok.init()

    gpu_num = hvd.size()
    rows = [8192] * gpu_num
    cols = [128 - 8 * x for x in range(gpu_num)]
    hotness = [1 + x for x in range(gpu_num)]
    combiners = ["mean"] * np.floor(gpu_num / 2).astype(np.int32) + ["sum"] * np.ceil(
        gpu_num - gpu_num / 2
    ).astype(np.int32)

    batch_size = 8192
    iters = 100
    initial_vals = [3 + x for x in range(gpu_num)]
    gpus = np.arange(gpu_num)

    optimizers = [
        tf.optimizers.SGD(learning_rate=1.0),
        tf.optimizers.SGD(learning_rate=1.0, momentum=0.9),
        tf.optimizers.Adamax(learning_rate=1.0, beta_1=0.9, beta_2=0.999),
        tf.optimizers.Adadelta(learning_rate=1.0),
        tf.optimizers.Adagrad(learning_rate=1.0),
        tf.optimizers.Ftrl(learning_rate=1.0),
    ]

    def step(params, indices):
        with tf.GradientTape() as tape:
            embeddings = sok.lookup_sparse(params, indices, combiners=combiners)
            loss = 0
            for i in range(len(embeddings)):
                loss = loss + tf.reduce_sum(embeddings[i])
        grads = tape.gradient(loss, params)
        sok_optimizer.apply_gradients(zip(grads, params))
        loss = hvd.allreduce(loss, op=hvd.Sum)
        return loss

    for optimizer_id, optimizer in enumerate(optimizers):
        sok_optimizer = sok.OptimizerWrapper(optimizer)
        # sok variables
        sok_vars = []
        if len(gpus) >= 2:
            for i in range(len(cols)):
                v = sok.DynamicVariable(
                    dimension=cols[i],
                    initializer=str(initial_vals[i]),
                    mode="localized:%d" % gpus[i],
                )
                sok_vars.append(v)
        else:
            for i in range(len(cols)):
                v = sok.DynamicVariable(
                    dimension=cols[i], initializer=str(initial_vals[i]), mode="localized:0"
                )
                sok_vars.append(v)
        local_indices = []
        for row in rows:
            local_size = row // hvd.size()
            if hvd.rank() < row % hvd.size():
                local_size += 1
            indices = np.arange(local_size) * hvd.size() + hvd.rank()
            indices = tf.convert_to_tensor(indices, dtype=tf.int64)
            local_indices.append(indices)

        # indices
        total_indices = []
        for i in range(len(rows)):
            offsets = np.random.randint(1, hotness[i] + 1, iters * batch_size)
            offsets = tf.convert_to_tensor(offsets, dtype=tf.int64)
            offsets = hvd.broadcast(offsets, root_rank=0)
            values = np.random.randint(0, rows[i], tf.reduce_sum(offsets))
            values = tf.convert_to_tensor(values, dtype=tf.int64)
            values = hvd.broadcast(values, root_rank=0)
            total_indices.append(tf.RaggedTensor.from_row_lengths(values, offsets))
        left = batch_size // hvd.size() * hvd.rank()
        right = batch_size // hvd.size() * (hvd.rank() + 1)
        indices = []
        for j in range(len(total_indices)):
            indices.append(total_indices[j][batch_size + left : batch_size + right])
        _ = step(sok_vars, indices)

        vars_unique_ids = []
        for sok_var in sok_vars:
            vars_unique_ids.append(sok_var._unique_id)
        have_state = True
        for vars_unique_id in vars_unique_ids:
            tmp_slot = optimizer._slots.get(vars_unique_id)
            if tmp_slot == None:
                have_state = False
                break
        slot_names = optimizer.get_slot_names()
        slot_states_list_raw = []
        slot_states_index_list_raw = []
        slot_vars_list = []
        if have_state:
            for slot_name in slot_names:
                slot_vars_np_list_raw = []
                slot_vars_index_np_list_raw = []
                tmp_slot_var_list = []
                for sok_var in sok_vars:
                    slot_var = optimizer.get_slot(sok_var, slot_name)
                    ex_indices, ex_values = sok.export(slot_var)
                    slot_vars_np_list_raw.append(ex_values.numpy())
                    slot_vars_index_np_list_raw.append(ex_indices.numpy())
                    tmp_slot_var_list.append(slot_var)
                slot_states_list_raw.append(slot_vars_np_list_raw)
                slot_states_index_list_raw.append(slot_vars_index_np_list_raw)
                slot_vars_list.append(tmp_slot_var_list)

        sok_var_nps_raw = []
        sok_var_index_nps_raw = []
        sok_var_nps_new = []
        sok_var_index_nps_new = []
        for sok_var in sok_vars:
            ex_indices, ex_values = sok.export(sok_var)

            sok_var_nps_raw.append(ex_values.numpy())
            sok_var_index_nps_raw.append(ex_indices.numpy())
        sok.dump("./weight", sok_vars, sok_optimizer)

        for sok_var in sok_vars:
            ex_indices, ex_values = sok.export(sok_var)
            zeros_values = tf.zeros(ex_values.shape)
            sok.assign(sok_var, ex_indices, zeros_values)

        for tmp_slot_list in slot_vars_list:
            for tmp_slot_var in tmp_slot_list:
                ex_indices, ex_values = sok.export(tmp_slot_var)
                zeros_values = tf.zeros(ex_values.shape)
                sok.assign(tmp_slot_var, ex_indices, zeros_values)
        sok.load("./weight", sok_vars, sok_optimizer)

        for sok_var in sok_vars:
            ex_indices, ex_values = sok.export(sok_var)
            sok_var_nps_new.append(ex_values.numpy())
            sok_var_index_nps_new.append(ex_indices.numpy())
        slot_states_list_new = []
        slot_states_index_list_new = []
        if have_state:
            for slot_name in slot_names:
                slot_vars_np_list_new = []
                slot_vars_index_np_list_new = []
                for sok_var in sok_vars:
                    slot_var = optimizer.get_slot(sok_var, slot_name)
                    ex_indices, ex_values = sok.export(slot_var)
                    slot_vars_np_list_new.append(ex_values.numpy())
                    slot_vars_index_np_list_new.append(ex_indices.numpy())
                slot_states_list_new.append(slot_vars_np_list_new)
                slot_states_index_list_new.append(slot_vars_index_np_list_new)

        # check var value before dump and var value after load
        for i in range(len(sok_vars)):
            var_sorted = np.argsort(sok_var_index_nps_raw[i])
            var_pos = np.searchsorted(
                sok_var_index_nps_raw[i][var_sorted], sok_var_index_nps_new[i]
            )
            remap_indices = var_sorted[var_pos]
            tmp_sok_var_nps_raw = sok_var_nps_raw[i][remap_indices, :]

            if hvd.rank() == sok_vars[i].target_gpu:
                assert ((sok_var_nps_new[i] - tmp_sok_var_nps_raw) < 1e-5).all()

        if have_state:
            for i, tmp_slot_states_list in enumerate(slot_states_list_new):
                for j, tmp_array in enumerate(tmp_slot_states_list):
                    index_raw = slot_states_index_list_raw[i][j]
                    index_new = slot_states_index_list_new[i][j]
                    var_sorted = np.argsort(index_raw)
                    var_pos = np.searchsorted(index_raw[var_sorted], index_new)
                    remap_indices = var_sorted[var_pos]
                    tmp_var_raw = slot_states_list_raw[i][j][remap_indices, :]
                    assert ((slot_states_list_new[i][j] - tmp_var_raw) < 1e-5).all()
        print(
            "[SOK INFO] dump load localized dynamic test %dth optimizer successfully" % optimizer_id
        )
