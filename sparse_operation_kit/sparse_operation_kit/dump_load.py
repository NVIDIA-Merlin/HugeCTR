#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import sys
import string
import numpy as np
from enum import Enum
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.resource_variable_ops import ResourceVariable, variable_accessed
from tensorflow.python.eager import context

from sparse_operation_kit.communication import rank
from sparse_operation_kit.communication import num_ranks
from sparse_operation_kit.communication import id_in_rank
from sparse_operation_kit.communication import num_gpus
from sparse_operation_kit.communication import alltoall
from sparse_operation_kit.communication import allreduce
from sparse_operation_kit.communication import allgather
from sparse_operation_kit.communication import broadcast
from sparse_operation_kit.communication import global_gpu_id

from sparse_operation_kit.distributed_variable import DistributedVariable
from sparse_operation_kit.distributed_variable import LocalizedVariable

from sparse_operation_kit.dynamic_variable import DynamicVariable, export, assign
from dataclasses import dataclass

# length:byte
integer_length = 4
long_long_length = 8
opt_name_max_length = 32
opt_var_name_max_length = 32
table_name_max_length = 256
file_head_length = 296
save_buffer_size_bytes = 1024 * 1024 * 64  # 1Gb
optimizer_names = ["SGD", "Adamax", "Adadelta", "Adagrad", "Ftrl", "Adam"]


class data_type_convert:
    # value :(dtypeindex , dtype data size in byte,numpy dtype)
    tf_dtype_info = {
        tf.int32: (0, 4, np.int32),
        tf.int64: (1, 8, np.int64),
        tf.uint32: (2, 4, np.uint64),
        tf.uint64: (3, 8, np.uint64),
        tf.float16: (4, 2, np.float16),
        tf.float32: (5, 4, np.float32),
        tf.float64: (6, 8, np.float64),
    }

    integer_to_dtype = {
        0: (tf.int32, np.int32),
        1: (tf.int64, np.int64),
        2: (tf.uint32, np.uint32),
        3: (tf.uint64, np.uint64),
        4: (tf.float16, np.float16),
        5: (tf.float32, np.float32),
        6: (tf.float64, np.float64),
    }

    @classmethod
    def convert_to_int(cls, tf_data_type):
        type_info = cls.tf_dtype_info.get(tf_data_type)
        if type_info == None:
            raise Exception(
                "SOK only support int32,int64,uint32,uint64 in key type, and float32,float64 in embedding var type"
            )
        return type_info[0]

    @classmethod
    def convert_to_np_dtype(cls, tf_data_type):
        type_info = cls.tf_dtype_info.get(tf_data_type)
        if type_info == None:
            raise Exception(
                "SOK only support int32,int64,uint32,uint64 in key type, and float32,float64 in embedding var type"
            )
        return type_info[2]

    @classmethod
    def get_dtype_size(cls, tf_data_type):
        type_info = cls.tf_dtype_info.get(tf_data_type)
        if type_info == None:
            raise Exception(
                "SOK only support int32,int64,uint32,uint64 in key type, and float32,float64 in embedding var type"
            )
        return type_info[1]

    @classmethod
    def get_tf_dtype_by_index(cls, index):
        type_info = cls.integer_to_dtype.get(index)
        if type_info == None:
            raise Exception(
                "SOK only support int32,int64,uint32,uint64 in key type, and float32,float64 in embedding var type"
            )
        return type_info[0]

    @classmethod
    def get_np_dtype_by_index(cls, index):
        type_info = cls.integer_to_dtype.get(index)
        if type_info == None:
            raise Exception(
                "SOK only support int32,int64,uint32,uint64 in key type, and float32,float64 in embedding var type"
            )
        return type_info[1]


def get_sok_optimizer_name(optimizer):
    if optimizer == None:
        return "None"
    if hasattr(optimizer, "_optimizer"):
        optimizer_full_name = str(type(optimizer._optimizer))
    else:
        optimizer_full_name = str(type(optimizer))
    return_name = ""
    for i, optimizer_name in enumerate(optimizer_names):
        if optimizer_name in optimizer_full_name:
            return_name = optimizer_name
    if len(return_name) == 0:
        raise Exception(
            "optimizer is %s,Now only support SGD,Adamax,Adadelta,Adagrad,Ftrl"
            % optimizer_full_name
        )
    return return_name


@dataclass
class SOK_var_info:
    opt_name: str  # optimizer name
    key_type: int  # key_data_type
    emb_type: int  # emb_data_type
    emb_num: int  # number of embedding vector
    emb_length: int  # length of embedding vector
    emb_name: str  #

    def __init__(
        self,
        opt_name_input: str = "",
        key_type_input: int = "",
        emb_type_input: int = "",
        emb_num_input: int = 0,
        emb_length_input: int = 0,
        emb_name_input: str = "",
    ):
        self.opt_name = opt_name_input
        self.key_type = key_type_input
        self.emb_type = emb_type_input
        self.emb_num = emb_num_input
        self.emb_length = emb_length_input
        self.emb_name = emb_name_input
        return


class MetaVarType(Enum):
    KeyType = 0
    EmbType = 1
    EmbLength = 2
    EmbNum = 3
    TableName = 4
    OptName = 5


class FileType(Enum):
    Key = 0
    Emb = 1
    OptState = 2


MetaVarOffsetDict = {
    MetaVarType.KeyType: 0,
    MetaVarType.EmbType: 4,
    MetaVarType.EmbLength: 8,
    MetaVarType.EmbNum: 12,
    MetaVarType.TableName: 20,
    MetaVarType.OptName: 20 + table_name_max_length,
}


def get_meta_info_offset(variable_type, table_num):
    assert isinstance(variable_type, MetaVarType)
    offset = integer_length  # metainfo have embedding table num at begin of it , type is a integer
    offset += MetaVarOffsetDict[variable_type] * table_num
    return offset


def save_meta_file(path: str, sok_var_info_list: list):
    meta_path = path + "/meta_info"
    num_tables = len(sok_var_info_list)

    table_name_list = []
    opt_name_list = []
    emb_length_list = []
    key_type_list = []
    embed_type_list = []
    embed_num_list = []
    gpu_id = global_gpu_id()

    if gpu_id == 0:
        for i in range(num_tables):
            tmp_sok_var_info = sok_var_info_list[i]
            table_name_list.append(tmp_sok_var_info.emb_name.rjust(table_name_max_length, " "))
            opt_name_list.append(tmp_sok_var_info.opt_name.rjust(opt_name_max_length, " "))
            emb_length_list.append(tmp_sok_var_info.emb_length)
            key_type_list.append(tmp_sok_var_info.key_type)
            embed_type_list.append(tmp_sok_var_info.emb_type)
            embed_num_list.append(tmp_sok_var_info.emb_num)
        f = open(meta_path, "wb")
        f.write(
            num_tables.to_bytes(integer_length, "big", signed=False)
        )  # use big-endien and unsigned save all the int data
        for key_type in key_type_list:
            f.write(key_type.to_bytes(integer_length, "big", signed=False))
        for embed_type in embed_type_list:
            f.write(embed_type.to_bytes(integer_length, "big", signed=False))
        for emb_length in emb_length_list:
            f.write(emb_length.to_bytes(integer_length, "big", signed=False))
        for embed_num in embed_num_list:
            f.write(embed_num.to_bytes(long_long_length, "big", signed=False))
        table_names_str = ""
        for table_name in table_name_list:
            table_names_str += table_name
        f.write(table_names_str.encode())  # str save in binaray is utf-8
        opt_names_str = ""

        for opt_name in opt_name_list:
            opt_names_str += opt_name
        f.write(opt_names_str.encode())
        f.close()

    # still need a barrier between gpus
    return


def convert_bytes_to_int_list(bytes_list, data_length, signed=False):
    tmp_list = [
        int.from_bytes(
            bytes_list[i * data_length : i * data_length + data_length], "big", signed=signed
        )
        for i in range(int(len(bytes_list) / data_length))
    ]
    return tmp_list


def convert_bytes_to_string_list(bytes_list, data_length, signed=False):
    tmp_list = [
        (bytes_list[i * data_length : i * data_length + data_length]).decode().strip()
        for i in range(int(len(bytes_list) / data_length))
    ]
    return tmp_list


def load_meta_file(path: str):
    meta_path = path + "/meta_info"

    if not os.path.exists(meta_path) or not os.path.exists(meta_path):
        raise Exception(
            "can't find meta_info data from path = %s ,please ensure the integrity of weight file"
            % path
        )
    f = open(meta_path, "rb")
    table_nums = int.from_bytes(f.read(integer_length), byteorder="big", signed=False)
    sok_var_info_list = [SOK_var_info()] * table_nums

    # read KeyType
    tmp_offset = get_meta_info_offset(MetaVarType.KeyType, table_nums)
    f.seek(tmp_offset, 0)
    tmp_bytes = f.read(table_nums * integer_length)
    key_type_list = convert_bytes_to_int_list(tmp_bytes, integer_length)
    for i in range(table_nums):
        sok_var_info_list[i].key_type = key_type_list[i]

    # read EmbType
    tmp_offset = get_meta_info_offset(MetaVarType.EmbType, table_nums)
    f.seek(tmp_offset, 0)
    tmp_bytes = f.read(table_nums * integer_length)
    emb_type_list = convert_bytes_to_int_list(tmp_bytes, integer_length)
    for i in range(table_nums):
        sok_var_info_list[i].emb_type = emb_type_list[i]

    # read EmbLength
    tmp_offset = get_meta_info_offset(MetaVarType.EmbLength, table_nums)
    f.seek(tmp_offset, 0)
    tmp_bytes = f.read(table_nums * integer_length)
    emb_length_list = convert_bytes_to_int_list(tmp_bytes, integer_length)
    for i in range(table_nums):
        sok_var_info_list[i].emb_length = emb_length_list[i]

    # read EmbNum
    tmp_offset = get_meta_info_offset(MetaVarType.EmbNum, table_nums)
    f.seek(tmp_offset, 0)
    tmp_bytes = f.read(table_nums * long_long_length)
    emb_num_list = convert_bytes_to_int_list(tmp_bytes, long_long_length)
    for i in range(table_nums):
        sok_var_info_list[i].emb_num = emb_num_list[i]

    # read TableName
    tmp_offset = get_meta_info_offset(MetaVarType.TableName, table_nums)
    f.seek(tmp_offset, 0)
    tmp_bytes = f.read(table_nums * table_name_max_length)
    emb_name_list = convert_bytes_to_string_list(tmp_bytes, table_name_max_length)
    for i in range(table_nums):
        sok_var_info_list[i].emb_name = emb_name_list[i]

    # read OptName
    tmp_offset = get_meta_info_offset(MetaVarType.OptName, table_nums)
    f.seek(tmp_offset, 0)
    tmp_bytes = f.read(table_nums * opt_name_max_length)
    opt_name_list = convert_bytes_to_string_list(tmp_bytes, opt_name_max_length)
    for i in range(table_nums):
        sok_var_info_list[i].opt_name = opt_name_list[i]

    sok_var_info_dict = {}
    for i in range(table_nums):
        sok_var_info_dict[sok_var_info_list[i].emb_name] = sok_var_info_list[i]

    return sok_var_info_dict


def write_file_head(path, sok_var_info, var_type, var_name, data_index):
    f = open(path, "wb")
    f.write(
        sok_var_info.emb_name.rjust(table_name_max_length, " ").encode()
    )  # str save in binaray is utf-8
    f.write(var_type.to_bytes(integer_length, "big", signed=False))
    f.write(var_name.rjust(opt_var_name_max_length, " ").encode())  # str save in binaray is utf-8
    f.write(data_index.to_bytes(integer_length, "big", signed=False))
    f.close()
    return


def read_file_head(path):
    f = open(path, "rb")
    emb_name = f.read(table_name_max_length).decode().strip()
    f.seek(table_name_max_length, 0)
    file_type = FileType(int.from_bytes(f.read(integer_length), "big", signed=False))
    f.seek(table_name_max_length + integer_length, 0)
    var_name = f.read(opt_var_name_max_length).decode().strip()
    f.seek(table_name_max_length + integer_length + opt_var_name_max_length, 0)
    type_index = int.from_bytes(f.read(integer_length), "big", signed=False)

    f.close()
    return emb_name, file_type, var_name, type_index


def check_optimizer_is_valid(optimizer, dump_vars):
    vars_unique_ids = []
    all_var_have_state = True
    all_var_not_have_state = True
    if optimizer is not None:
        for dump_var in dump_vars:
            vars_unique_ids.append(dump_var._unique_id)
        for vars_unique_id in vars_unique_ids:
            tmp_slot = optimizer._slots.get(vars_unique_id)
            if tmp_slot == None:
                all_var_have_state = False
            else:
                all_var_not_have_state = False
        if not (all_var_have_state or all_var_not_have_state):
            raise (
                "Now only support :1.all input sok vars have states in optimizer 2. all input sok vars  don't have states in optimizer"
            )
    return all_var_have_state, all_var_not_have_state


def activate_optimizer_state(optimizer, load_vars):
    have_dynamic = False
    for load_var in load_vars:
        if isinstance(load_var, DynamicVariable):
            have_dynamic = True
            break
    if have_dynamic and (
        "sparse_operation_kit.optimizer.OptimizerWrapper" not in str(type(optimizer))
    ):
        raise ("load a dynamic variable but optimize is not sok OptimizerWrapper")

    optimizer._create_slots(load_vars)


def get_save_rounds(tensor_shape, tensor_data_size):
    num_vector = tensor_shape[0]
    embedding_length = tensor_shape[1]
    total_data_size = tensor_shape.num_elements() * tensor_data_size
    if total_data_size <= save_buffer_size_bytes:
        num_rounds = 1
        tmp_num_vector = num_vector
    else:
        tmp_num_vector = int(
            np.floor(save_buffer_size_bytes / (embedding_length * tensor_data_size))
        )
        num_rounds = np.ceil(num_vector / tmp_num_vector).astype(np.int64)
    num_rounds_tensor = tf.convert_to_tensor([num_rounds])
    # num_rounds_tensor = allreduce(num_rounds_tensor, "max")
    num_rounds_tensor_all = allgather(num_rounds_tensor)
    num_rounds_tensor_all = tf.math.reduce_max(num_rounds_tensor_all)
    return num_rounds_tensor_all.numpy(), tmp_num_vector


def save_optimizer_to_filesysyem_static(optimizer, var, path, sok_var_info):
    optimizer_name = get_sok_optimizer_name(optimizer)
    slot_names = optimizer.get_slot_names()
    global_gpu_num = num_gpus()
    gpu_id = global_gpu_id()
    table_name = var.name
    for i in string.punctuation:
        table_name = table_name.replace(i, "_")
    for slot_name in slot_names:
        slot_path = path + "/" + table_name + "-" + optimizer_name + "-" + slot_name
        target_gpu = var.target_gpu
        slot_var = optimizer.get_slot(var, slot_name)
        state_tensor = tf.convert_to_tensor(slot_var, dtype=slot_var.dtype)
        # is distribute
        if target_gpu == -1:
            tmp_dtype = state_tensor.dtype
            dtype_size = tmp_dtype.size
            num_rounds, num_vector_per_round = get_save_rounds(state_tensor.shape, dtype_size)
            if gpu_id == 0:
                write_file_head(
                    slot_path,
                    sok_var_info,
                    FileType.OptState.value,
                    slot_name,
                    data_type_convert.convert_to_int(state_tensor.dtype),
                )
            save_offset = 0
            for i in range(num_rounds):
                start_offset = save_offset
                end_offset = save_offset + num_vector_per_round
                save_offset += num_vector_per_round
                if start_offset > state_tensor.shape[0]:
                    start_offset = state_tensor.shape[0]
                if end_offset > state_tensor.shape[0]:
                    end_offset = state_tensor.shape[0]
                tmp_state_tensor = state_tensor[start_offset:end_offset, :]
                if global_gpu_num > 1:
                    tmp_state_tensor = allgather(tmp_state_tensor)
                if gpu_id == 0:
                    tmp_state_tensor_np = tmp_state_tensor.numpy()
                    with open(slot_path, mode="ba+") as fstate:
                        tmp_state_tensor_np.tofile(fstate)
                del tmp_state_tensor

        else:
            if gpu_id == target_gpu:
                write_file_head(
                    slot_path,
                    sok_var_info,
                    FileType.OptState.value,
                    slot_name,
                    data_type_convert.convert_to_int(state_tensor.dtype),
                )
                state_tensor_np = state_tensor.numpy()

                with open(slot_path, mode="ba+") as fstate:
                    state_tensor_np.tofile(fstate)


def save_optimizer_to_filesysyem_dynamic(optimizer, var, path, sok_var_info):
    optimizer_name = get_sok_optimizer_name(optimizer)
    slot_names = optimizer.get_slot_names()
    global_gpu_num = num_gpus()
    gpu_id = global_gpu_id()
    table_name = var.name
    for i in string.punctuation:
        table_name = table_name.replace(i, "_")

    # indices_var, _ = export(var)
    # indices_var_np = indices_var.numpy()
    for slot_name in slot_names:
        slot_path = path + "/" + table_name + "-" + optimizer_name + "-" + slot_name
        target_gpu = var.target_gpu
        slot_var = optimizer.get_slot(var, slot_name)
        indices_tensor, state_tensor = export(slot_var)

        sort_indice_tensor = tf.cast(tf.argsort(indices_tensor), tf.int64)
        state_tensor = tf.gather(state_tensor, sort_indice_tensor)
        indices_tensor = tf.gather(indices_tensor, sort_indice_tensor)

        # is distribute
        if target_gpu == -1:
            tmp_dtype = state_tensor.dtype
            dtype_size = tmp_dtype.size
            num_rounds, num_vector_per_round = get_save_rounds(state_tensor.shape, dtype_size)
            if gpu_id == 0:
                write_file_head(
                    slot_path,
                    sok_var_info,
                    FileType.OptState.value,
                    slot_name,
                    data_type_convert.convert_to_int(state_tensor.dtype),
                )
            save_offset = 0
            for i in range(num_rounds):
                start_offset = save_offset
                end_offset = save_offset + num_vector_per_round
                save_offset += num_vector_per_round
                if start_offset > state_tensor.shape[0]:
                    start_offset = state_tensor.shape[0]
                if end_offset > state_tensor.shape[0]:
                    end_offset = state_tensor.shape[0]
                tmp_state_tensor = state_tensor[start_offset:end_offset, :]
                if global_gpu_num > 1:
                    tmp_state_tensor = allgather(tmp_state_tensor)
                if gpu_id == 0:
                    tmp_state_np = tmp_state_tensor.numpy()
                    with open(slot_path, mode="ba+") as fstate:
                        tmp_state_np.tofile(fstate)
                del tmp_state_tensor

        else:
            if gpu_id == target_gpu:
                write_file_head(
                    slot_path,
                    sok_var_info,
                    FileType.OptState.value,
                    slot_name,
                    data_type_convert.convert_to_int(state_tensor.dtype),
                )
                state_np = state_tensor.numpy()

                with open(slot_path, mode="ba+") as fstate:
                    state_np.tofile(fstate)


def save_table_to_filesystem_static(var, optimizer, path, have_states):
    global_gpu_num = num_gpus()
    gpu_id = global_gpu_id()
    table_name = var.name
    for i in string.punctuation:
        table_name = table_name.replace(i, "_")
    key_path = path + "/" + table_name + "-key"
    weight_path = path + "/" + table_name + "-weight"
    target_gpu = var.target_gpu
    sok_var_info = SOK_var_info()
    sok_var_info.opt_name = get_sok_optimizer_name(optimizer)

    # is distribute
    if target_gpu == -1:
        num_ev = var.shape[0]
        ev_length = var.shape[1]
        num_np = np.zeros(global_gpu_num, dtype=np.uint64)
        num_np[gpu_id] = num_ev
        num_evs = tf.convert_to_tensor(num_np, dtype=tf.uint64)
        indice_np = np.arange(int(num_ev), dtype=np.int64)
        indice_np = indice_np * global_gpu_num + gpu_id
        indice = tf.convert_to_tensor(indice_np, dtype=tf.int64)
        weight = tf.convert_to_tensor(var, var.dtype)

        tmp_dtype = weight.dtype
        dtype_size = tmp_dtype.size
        num_rounds, num_vector_per_round = get_save_rounds(weight.shape, dtype_size)

        total_indice = allgather(indice)
        if gpu_id == 0:
            sok_var_info.key_type = data_type_convert.convert_to_int(indice.dtype)
            sok_var_info.emb_type = data_type_convert.convert_to_int(weight.dtype)
            sok_var_info.emb_num = total_indice.shape[0]
            sok_var_info.emb_length = var.shape[1]

            write_file_head(
                key_path,
                sok_var_info,
                FileType.Key.value,
                "",
                data_type_convert.convert_to_int(indice.dtype),
            )
            write_file_head(
                weight_path,
                sok_var_info,
                FileType.Emb.value,
                "",
                data_type_convert.convert_to_int(weight.dtype),
            )

        save_offset = 0
        for i in range(num_rounds):
            start_offset = save_offset
            end_offset = save_offset + num_vector_per_round
            save_offset += num_vector_per_round
            if start_offset > weight.shape[0]:
                start_offset = weight.shape[0]
            if end_offset > weight.shape[0]:
                end_offset = weight.shape[0]
            tmp_indice = indice[start_offset:end_offset]
            tmp_weight = weight[start_offset:end_offset, :]
            if global_gpu_num > 1:
                tmp_indice = allgather(tmp_indice)
                tmp_weight = allgather(tmp_weight)
            if gpu_id == 0:
                indice_np = tmp_indice.numpy()
                weight_np = tmp_weight.numpy()
                with open(key_path, mode="ba+") as fkey:
                    indice_np.tofile(fkey)

                with open(weight_path, mode="ba+") as femb:
                    weight_np.tofile(femb)
                del tmp_indice
                del tmp_weight

    else:
        if gpu_id == target_gpu:
            num_ev = var.shape[0]
            ev_length = var.shape[1]
            num_np = np.zeros(global_gpu_num, dtype=np.uint64)
            num_np[gpu_id] = num_ev
            num_evs = tf.convert_to_tensor(num_np, dtype=tf.uint64)
            indice_np = np.arange(num_ev, dtype=np.uint64)
            indice_np = indice_np * global_gpu_num + gpu_id
            indice = tf.convert_to_tensor(indice_np, dtype=tf.uint64)
            weight = tf.convert_to_tensor(var, var.dtype)

            num_ev = var.shape[0]
            ev_length = var.shape[1]
            sok_var_info.key_type = data_type_convert.convert_to_int(indice.dtype)
            sok_var_info.emb_type = data_type_convert.convert_to_int(weight.dtype)
            sok_var_info.emb_num = indice.shape[0]
            sok_var_info.emb_length = var.shape[1]
            indice_np = np.arange(num_ev, dtype=np.uint64)
            weight_np = var.numpy()
            write_file_head(
                key_path,
                sok_var_info,
                FileType.Key.value,
                "",
                data_type_convert.convert_to_int(tf.uint64),
            )
            write_file_head(
                weight_path,
                sok_var_info,
                FileType.Emb.value,
                "",
                data_type_convert.convert_to_int(var.dtype),
            )
            with open(key_path, mode="ba+") as fkey:
                indice_np.tofile(fkey)
            with open(weight_path, mode="ba+") as femb:
                weight_np.tofile(femb)

    if optimizer != None and have_states[0]:
        save_optimizer_to_filesysyem_static(optimizer, var, path, sok_var_info)


def save_table_to_filesystem_dynamic(var, optimizer, path, have_states):
    global_gpu_num = num_gpus()
    gpu_id = global_gpu_id()
    table_name = var.name
    for i in string.punctuation:
        table_name = table_name.replace(i, "_")
    key_path = path + "/" + table_name + "-key"
    weight_path = path + "/" + table_name + "-weight"
    target_gpu = var.target_gpu
    sok_var_info = SOK_var_info()
    sok_var_info.opt_name = get_sok_optimizer_name(optimizer)
    # is distribute
    if target_gpu == -1:
        indice, weight = export(var)
        sort_indice_tensor = tf.cast(tf.argsort(indice), tf.int64)
        weight = tf.gather(weight, sort_indice_tensor)
        indice = tf.gather(indice, sort_indice_tensor)

        ev_length = weight.shape[1]
        tmp_dtype = weight.dtype
        dtype_size = tmp_dtype.size
        num_rounds, num_vector_per_round = get_save_rounds(weight.shape, dtype_size)

        total_indice = allgather(indice)
        if gpu_id == 0:
            if total_indice.shape[0] == 0:
                raise Exception("dynamic table don't have value in it , table_name:", table_name)
            sok_var_info.key_type = data_type_convert.convert_to_int(indice.dtype)
            sok_var_info.emb_type = data_type_convert.convert_to_int(weight.dtype)
            sok_var_info.emb_num = total_indice.shape[0]
            sok_var_info.emb_length = ev_length

            write_file_head(
                key_path,
                sok_var_info,
                FileType.Key.value,
                "",
                data_type_convert.convert_to_int(indice.dtype),
            )
            write_file_head(
                weight_path,
                sok_var_info,
                FileType.Emb.value,
                "",
                data_type_convert.convert_to_int(weight.dtype),
            )
        save_offset = 0
        for i in range(num_rounds):
            start_offset = save_offset
            end_offset = save_offset + num_vector_per_round
            save_offset += num_vector_per_round
            if start_offset > weight.shape[0]:
                start_offset = weight.shape[0]
            if end_offset > weight.shape[0]:
                end_offset = weight.shape[0]

            tmp_indice = indice[start_offset:end_offset]
            tmp_weight = weight[start_offset:end_offset, :]
            if global_gpu_num > 1:
                tmp_indice = allgather(tmp_indice)
                tmp_weight = allgather(tmp_weight)

            if gpu_id == 0:
                indice_np = tmp_indice.numpy()
                weight_np = tmp_weight.numpy()

                with open(key_path, mode="ba+") as fkey:
                    indice_np.tofile(fkey)
                with open(weight_path, mode="ba+") as femb:
                    weight_np.tofile(femb)
                del tmp_indice
                del tmp_weight

    else:
        if gpu_id == target_gpu:
            ex_indices, ex_values = export(var)
            ev_length = ex_values.shape[1]
            num_ev = ex_values.shape[0]
            if num_ev == 0:
                raise Exception("dynamic table don't have value in it , table_name:", table_name)

            sok_var_info.key_type = data_type_convert.convert_to_int(ex_indices.dtype)
            sok_var_info.emb_type = data_type_convert.convert_to_int(ex_values.dtype)
            sok_var_info.emb_num = ex_indices.shape[0]
            sok_var_info.emb_length = ex_values.shape[1]

            ex_indices_np = ex_indices.numpy()
            ex_values_np = ex_values.numpy()

            sort_indice_index = np.argsort(ex_indices_np)
            ex_indices_np = ex_indices_np[sort_indice_index]
            ex_values_np = ex_values_np[sort_indice_index]
            write_file_head(
                key_path,
                sok_var_info,
                FileType.Key.value,
                "",
                data_type_convert.convert_to_int(ex_indices.dtype),
            )
            write_file_head(
                weight_path,
                sok_var_info,
                FileType.Emb.value,
                "",
                data_type_convert.convert_to_int(ex_values.dtype),
            )
            with open(key_path, mode="ba+") as fkey:
                ex_indices_np.tofile(fkey)
            with open(weight_path, mode="ba+") as femb:
                ex_values_np.tofile(femb)

    if optimizer != None and have_states[0]:
        save_optimizer_to_filesysyem_dynamic(optimizer, var, path, sok_var_info)


def check_weight_file_valid(key_path, weight_path, optimizer_state_paths):
    if not os.path.exists(key_path):
        return False, "key file %s is not exist" % key_path
    if not os.path.exists(weight_path):
        return False, "weight file %s is not exist" % weight_path

    _, _, _, key_data_index = read_file_head(key_path)
    _, _, _, emb_data_index = read_file_head(weight_path)

    key_data_length = data_type_convert.get_dtype_size(
        data_type_convert.get_tf_dtype_by_index(key_data_index)
    )
    emb_data_length = data_type_convert.get_dtype_size(
        data_type_convert.get_tf_dtype_by_index(emb_data_index)
    )
    key_file_size = os.stat(key_path).st_size - file_head_length
    weight_file_size = os.stat(weight_path).st_size - file_head_length

    if key_file_size % key_data_length != 0:
        return (
            False,
            "The effective length of the key file(%s) is not divisible by the length of the key type "
            % key_path,
        )
    if weight_file_size % emb_data_length != 0:
        return (
            False,
            "The effective length of the weight file(%s) is not divisible by the length of theweight type "
            % weight_path,
        )
    key_num = key_file_size / key_data_length

    if (weight_file_size / emb_data_length) % key_num != 0:
        return (
            False,
            "The effective length of the weight file(%s) is not divisible by key_num "
            % weight_path,
        )

    ev_length = weight_file_size / emb_data_length / key_num

    if len(optimizer_state_paths) > 0:
        for optimizer_state_path in optimizer_state_paths:
            if not os.path.exists(optimizer_state_path):
                return False, "optimizer state file %s is not exist" % optimizer_state_path
            _, _, _, tmp_data_index = read_file_head(optimizer_state_path)
            tmp_data_length = data_type_convert.get_dtype_size(
                data_type_convert.get_tf_dtype_by_index(tmp_data_index)
            )
            tmp_file_size = os.stat(optimizer_state_path).st_size - file_head_length
            if tmp_file_size % tmp_data_length != 0:
                return (
                    False,
                    "The effective length of the optimizer state file(%s) is not divisible by the length of the weight type "
                    % optimizer_state_path,
                )
            if (tmp_file_size / tmp_data_length) % key_num != 0:
                return (
                    False,
                    "The effective length of the optimizer state file(%s) is not divisible by key_num "
                    % optimizer_state_path,
                )
            if tmp_file_size / emb_data_length / key_num != ev_length:
                return (
                    False,
                    "the optimizer state file(%s)'s ev_length is not equal to weight file"
                    % optimizer_state_path,
                )

    return True, ""


def load_table_to_filesystem_static(var, optimizer, path):
    global_gpu_num = num_gpus()
    gpu_id = global_gpu_id()
    table_name = var.name
    for i in string.punctuation:
        table_name = table_name.replace(i, "_")
    key_path = path + "/" + table_name + "-key"
    weight_path = path + "/" + table_name + "-weight"
    if not os.path.exists(key_path) or not os.path.exists(weight_path):
        raise Exception("can't find key or weight to load with table name:", table_name)
    target_gpu = var.target_gpu

    optimizer_state_paths = []
    optimizer_state_names = []
    optimizer_name = ""
    if optimizer is not None:
        optimizer_name = get_sok_optimizer_name(optimizer)
        slot_names = optimizer.get_slot_names()
        for slot_name in slot_names:
            optimizer_state_names.append(slot_name)
            optimizer_state_paths.append(
                path + "/" + table_name + "-" + optimizer_name + "-" + slot_name
            )

    file_valid, error_msg = check_weight_file_valid(key_path, weight_path, optimizer_state_paths)
    if not file_valid:
        raise Exception(error_msg)

    _, _, _, key_data_index = read_file_head(key_path)
    _, _, _, emb_data_index = read_file_head(weight_path)
    key_tf_dtype = data_type_convert.get_tf_dtype_by_index(key_data_index)
    key_np_dtype = data_type_convert.get_np_dtype_by_index(key_data_index)
    emb_tf_dtype = data_type_convert.get_tf_dtype_by_index(emb_data_index)
    emb_np_dtype = data_type_convert.get_np_dtype_by_index(emb_data_index)

    # is distribute
    if target_gpu == -1:
        num_ev = var.shape[0]
        ev_length = var.shape[1]

        with open(key_path, "rb") as fkey:
            fkey.seek(file_head_length, os.SEEK_SET)
            indice_np = np.fromfile(fkey, dtype=key_np_dtype)

        with open(weight_path, "rb") as femb:
            femb.seek(file_head_length, os.SEEK_SET)
            weight_np = np.fromfile(femb, dtype=emb_np_dtype).reshape((-1, ev_length))
        mask_np = indice_np % global_gpu_num == gpu_id
        indice_np = indice_np[mask_np]
        weight_np = weight_np[mask_np]

        indice = tf.convert_to_tensor(indice_np, dtype=key_tf_dtype)
        weight = tf.convert_to_tensor(weight_np, dtype=emb_tf_dtype)
        try:
            var.assign(weight)
        except:
            raise Exception("weight file(%s)'s weight is not same with sok variable" % weight_path)
        if len(optimizer_state_paths) > 0:
            # activate_optimizer_state(optimizer, [var])
            for i, optimizer_state_path in enumerate(optimizer_state_paths):
                _, _, _, tmp_data_index = read_file_head(optimizer_state_path)
                tmp_tf_dtype = data_type_convert.get_tf_dtype_by_index(tmp_data_index)
                tmp_np_dtype = data_type_convert.get_np_dtype_by_index(tmp_data_index)

                slot_var = optimizer.get_slot(var, optimizer_state_names[i])
                with open(optimizer_state_path, "rb") as ftmp:
                    ftmp.seek(file_head_length, os.SEEK_SET)
                    tmp_state_np = np.fromfile(ftmp, dtype=tmp_np_dtype).reshape((-1, ev_length))
                tmp_state_np = tmp_state_np[mask_np]
                tmp_state = tf.convert_to_tensor(tmp_state_np, dtype=tmp_tf_dtype)
                try:
                    slot_var.assign(tmp_state)
                except:
                    raise Exception(
                        "state file(%s)'s weight is not same with sok variable"
                        % optimizer_state_path
                    )
    else:
        if gpu_id == target_gpu:
            num_ev = var.shape[0]
            ev_length = var.shape[1]

            with open(key_path, "rb") as fkey:
                fkey.seek(file_head_length, os.SEEK_SET)
                indice_np = np.fromfile(fkey, dtype=key_np_dtype)

            with open(weight_path, "rb") as femb:
                femb.seek(file_head_length, os.SEEK_SET)
                weight_np = np.fromfile(femb, dtype=emb_np_dtype).reshape((-1, ev_length))

            indice = tf.convert_to_tensor(indice_np, dtype=key_tf_dtype)
            weight = tf.convert_to_tensor(weight_np, dtype=emb_tf_dtype)
            try:
                var.assign(weight)
            except:
                raise Exception(
                    "weight file(%s)'s weight is not same with sok variable" % weight_path
                )
            if len(optimizer_state_paths) > 0:
                # activate_optimizer_state(optimizer, [var])
                for i, optimizer_state_path in enumerate(optimizer_state_paths):
                    _, _, _, tmp_data_index = read_file_head(optimizer_state_path)
                    tmp_tf_dtype = data_type_convert.get_tf_dtype_by_index(tmp_data_index)
                    tmp_np_dtype = data_type_convert.get_np_dtype_by_index(tmp_data_index)

                    slot_var = optimizer.get_slot(var, optimizer_state_names[i])
                    with open(optimizer_state_path, "rb") as ftmp:
                        ftmp.seek(file_head_length, os.SEEK_SET)
                        tmp_state_np = np.fromfile(ftmp, dtype=tmp_np_dtype).reshape(
                            (-1, ev_length)
                        )
                    tmp_state = tf.convert_to_tensor(tmp_state_np, dtype=tmp_tf_dtype)
                    try:
                        slot_var.assign(tmp_state)
                    except:
                        raise Exception(
                            "state file(%s)'s weight is not same with sok variable"
                            % optimizer_state_path
                        )

    return


def load_table_to_filesystem_dynamic(var, optimizer, path):
    global_gpu_num = num_gpus()
    gpu_id = global_gpu_id()
    table_name = var.name
    for i in string.punctuation:
        table_name = table_name.replace(i, "_")
    key_path = path + "/" + table_name + "-key"
    weight_path = path + "/" + table_name + "-weight"
    if not os.path.exists(key_path) or not os.path.exists(weight_path):
        raise Exception("can't find key or weight to load with table name:", table_name)
    target_gpu = var.target_gpu

    optimizer_state_paths = []
    optimizer_state_names = []
    optimizer_name = ""
    if optimizer is not None:
        optimizer_name = get_sok_optimizer_name(optimizer)
        slot_names = optimizer.get_slot_names()
        for slot_name in slot_names:
            optimizer_state_names.append(slot_name)
            optimizer_state_paths.append(
                path + "/" + table_name + "-" + optimizer_name + "-" + slot_name
            )

    file_valid, error_msg = check_weight_file_valid(key_path, weight_path, optimizer_state_paths)
    if not file_valid:
        raise Exception(error_msg)

    _, _, _, key_data_index = read_file_head(key_path)
    _, _, _, emb_data_index = read_file_head(weight_path)
    key_tf_dtype = data_type_convert.get_tf_dtype_by_index(key_data_index)
    key_np_dtype = data_type_convert.get_np_dtype_by_index(key_data_index)
    emb_tf_dtype = data_type_convert.get_tf_dtype_by_index(emb_data_index)
    emb_np_dtype = data_type_convert.get_np_dtype_by_index(emb_data_index)

    # is distribute
    if target_gpu == -1:
        ev_length = var.shape[1]
        with open(key_path, "rb") as fkey:
            fkey.seek(file_head_length, os.SEEK_SET)
            indice_np = np.fromfile(fkey, dtype=key_np_dtype)

        with open(weight_path, "rb") as femb:
            femb.seek(file_head_length, os.SEEK_SET)
            weight_np = np.fromfile(femb, dtype=emb_np_dtype).reshape((-1, ev_length))

        mask_np = indice_np % global_gpu_num == gpu_id
        indice_np = indice_np[mask_np]
        weight_np = weight_np[mask_np]

        indice = tf.convert_to_tensor(indice_np, dtype=key_tf_dtype)
        weight = tf.convert_to_tensor(weight_np, dtype=emb_tf_dtype)
        assign(var, indice, weight)
        if len(optimizer_state_paths) > 0:
            # activate_optimizer_state(optimizer, [var])
            for i, optimizer_state_path in enumerate(optimizer_state_paths):
                _, _, _, tmp_data_index = read_file_head(optimizer_state_path)
                tmp_tf_dtype = data_type_convert.get_tf_dtype_by_index(tmp_data_index)
                tmp_np_dtype = data_type_convert.get_np_dtype_by_index(tmp_data_index)

                slot_var = optimizer.get_slot(var, optimizer_state_names[i])
                with open(optimizer_state_path, "rb") as ftmp:
                    ftmp.seek(file_head_length, os.SEEK_SET)
                    tmp_state_np = np.fromfile(ftmp, dtype=tmp_np_dtype).reshape((-1, ev_length))
                tmp_state_np = tmp_state_np[mask_np]
                tmp_state = tf.convert_to_tensor(tmp_state_np, dtype=tmp_tf_dtype)
                assign(slot_var, indice, tmp_state)

    else:
        if gpu_id == target_gpu:
            ev_length = var.dimension
            with open(key_path, "rb") as fkey:
                fkey.seek(file_head_length, os.SEEK_SET)
                indice_np = np.fromfile(fkey, dtype=key_np_dtype)

            with open(weight_path, "rb") as femb:
                femb.seek(file_head_length, os.SEEK_SET)
                weight_np = np.fromfile(femb, dtype=emb_np_dtype).reshape((-1, ev_length))

            indice = tf.convert_to_tensor(indice_np, dtype=key_tf_dtype)
            weight = tf.convert_to_tensor(weight_np, dtype=emb_tf_dtype)
            assign(var, indice, weight)
            if len(optimizer_state_paths) > 0:
                # activate_optimizer_state(optimizer, [var])
                for i, optimizer_state_path in enumerate(optimizer_state_paths):
                    _, _, _, tmp_data_index = read_file_head(optimizer_state_path)
                    tmp_tf_dtype = data_type_convert.get_tf_dtype_by_index(tmp_data_index)
                    tmp_np_dtype = data_type_convert.get_np_dtype_by_index(tmp_data_index)

                    slot_var = optimizer.get_slot(var, optimizer_state_names[i])
                    with open(optimizer_state_path, "rb") as ftmp:
                        ftmp.seek(file_head_length, os.SEEK_SET)
                        tmp_state_np = np.fromfile(ftmp, dtype=tmp_np_dtype).reshape(
                            (-1, ev_length)
                        )
                    tmp_state = tf.convert_to_tensor(tmp_state_np, dtype=tmp_tf_dtype)
                    assign(slot_var, indice, tmp_state)


def dump_per_table(var, optimizer, path, have_states):
    if isinstance(var, DynamicVariable):
        save_table_to_filesystem_dynamic(var, optimizer, path, have_states)

    elif isinstance(var, DistributedVariable):
        save_table_to_filesystem_static(var, optimizer, path, have_states)

    elif isinstance(var, LocalizedVariable):
        save_table_to_filesystem_static(var, optimizer, path, have_states)

    else:
        raise Exception("dump table type should be sok.DynamicVariable or sok.Variable")


def load_per_table(var, optimizer, path):
    if isinstance(var, DynamicVariable):
        load_table_to_filesystem_dynamic(var, optimizer, path)
    elif isinstance(var, DistributedVariable):
        load_table_to_filesystem_static(var, optimizer, path)

    elif isinstance(var, LocalizedVariable):
        load_table_to_filesystem_static(var, optimizer, path)
    else:
        raise Exception("load table type should be sok.DynamicVariable or sok.Variable")


def load_table(path, load_vars, optimizer):
    if optimizer is not None:
        if (
            (not isinstance(optimizer, tf.keras.optimizers.Optimizer))
            and (not isinstance(optimizer, tf.optimizers.Optimizer))
            and ("sparse_operation_kit.optimizer.OptimizerWrapper" not in str(type(optimizer)))
        ):
            raise Exception(
                "the type of your input optimizer is not tf.optimizers.Optimizer or sok.optimizer.OptimizerWrapper, please checkout your dump optimizer input"
            )
        activate_optimizer_state(optimizer, load_vars)
    for var in load_vars:
        load_per_table(var, optimizer, path)

    ar_flag_np = np.arange(1)
    ar_flag = tf.convert_to_tensor(ar_flag_np, dtype=tf.int32)
    _ = allreduce(ar_flag, op="sum")
    return


def dump_table(path, dump_vars, optimizer):
    if optimizer is not None:
        if (
            (not isinstance(optimizer, tf.keras.optimizers.Optimizer))
            and (not isinstance(optimizer, tf.optimizers.Optimizer))
            and ("sparse_operation_kit.optimizer.OptimizerWrapper" not in str(type(optimizer)))
        ):
            raise Exception(
                "the type of your input optimizer is not tf.optimizers.Optimizer or sok.optimizer.OptimizerWrapper, please checkout your dump optimizer input"
            )
    # have_states: first element is all_var_have_state, second element is all_var_not_have_state
    have_states = check_optimizer_is_valid(optimizer, dump_vars)
    for var in dump_vars:
        dump_per_table(var, optimizer, path, have_states)
    ar_flag_np = np.arange(1)
    ar_flag = tf.convert_to_tensor(ar_flag_np, dtype=tf.int32)
    _ = allreduce(ar_flag, op="sum")
    return


def dump(path, dump_vars, optimizer=None):
    """
    Abbreviated as ``sok.dump``.

    Dump the embedding tables into one folder , will generate key and value file per table.

    Table name prefix is same as variable name in tensorflow.

    Now is only support ``SGD,Adamax,Adadelta,Adagrad,Ftrl,Adam`` optimizers.

    Note: for ``Adam`` optimizers , it functions in the same way as ``LazyAdam`` in SOK, rather than TensorFlow's ``Adam``

    Parameters
    ----------
    path: string
          weight file folder
    dump_vars: List,Tuple,SOK Variable
               Can be a single or list of sok.Variable and sok.DynamicVariable
    optimizer: SOK.OptimizerWrapper,optional,default is None
               when model train , need to dump optimizer state,input ``sok.OptimizerWrapper``

    Returns
    -------
    None

    Example
    -------
    .. code-block:: python

        import numpy as np
        import tensorflow as tf
        import horovod.tensorflow as hvd
        import sparse_operation_kit as sok

        v = sok.DynamicVariable(dimension=3, initializer="13")

        optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
        optimizer = sok.OptimizerWrapper(optimizer)
        path = "your weight path , weight is dumped by sok.dump"


        indices = tf.convert_to_tensor([0, 1, 2**40], dtype=tf.int64)

        with tf.GradientTape() as tape:
            embedding = tf.nn.embedding_lookup(v, indices)
            print("embedding:", embedding)
            loss = tf.reduce_sum(embedding)

        grads = tape.gradient(loss, [v])
        optimizer.apply_gradients(zip(grads, [v]))

        sok.dump(path,v,optimizer)
    """
    try:
        os.makedirs(path, exist_ok=True)
    except:
        raise Exception("can't build path:", path)
    is_list = isinstance(dump_vars, list) or isinstance(dump_vars, tuple)
    if not is_list:
        dump_vars = [dump_vars]
    assert isinstance(dump_vars, list) or isinstance(dump_vars, tuple)

    have_none = False
    not_sok_variable = False
    for dump_var in dump_vars:
        if dump_vars == None:
            have_none = True
    if have_none:
        raise Exception("the input of your sok variables have none ,please")

    if isinstance(optimizer, list) or isinstance(optimizer, tuple):
        if len(optimizer) > 1:
            raise Exception("Only support dump one optmizer state")
        if len(optimizer) == 0:
            optimizer = None
        optimizer = optimizer[0]

    dump_table(path, dump_vars, optimizer)
    print("[SOK INFO] SOK dump weight in path:", path, " success!")
    return


def load(path, load_vars, optimizer=None):
    """
    Abbreviated as ``sok.load``.

    Load the embedding tables from sok weight folder.
    The sok table name must be the same with the weight file prefix.

    Now is only support ``SGD,Adamax,Adadelta,Adagrad,Ftrl,Adam`` optimizers.

    Note: for ``Adam`` optimizers , it functions in the same way as ``LazyAdam`` in SOK, rather than TensorFlow's ``Adam``

    Parameters
    ----------
    path: string
          weight file folder
    load_vars: List,Tuple,SOK Variable
               Can be a single or list of sok.Variable and sok.DynamicVariable
    optimizer: SOK.OptimizerWrapper,optional,default is None
               when model train , need to load optimizer state,input ``sok.OptimizerWrapper``

    Returns
    -------
    None

    Example
    -------
    .. code-block:: python

        import numpy as np
        import tensorflow as tf
        import horovod.tensorflow as hvd
        import sparse_operation_kit as sok

        v = sok.DynamicVariable(dimension=3, initializer="13")

        optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
        optimizer = sok.OptimizerWrapper(optimizer)
        path = "your weight path , weight is dumped by sok.dump"

        sok.load(path,v,optimizer)

        indices = tf.convert_to_tensor([0, 1, 2**40], dtype=tf.int64)

        with tf.GradientTape() as tape:
            embedding = tf.nn.embedding_lookup(v, indices)
            print("embedding:", embedding)
            loss = tf.reduce_sum(embedding)

        grads = tape.gradient(loss, [v])
        optimizer.apply_gradients(zip(grads, [v]))
    """
    if not os.path.exists(path):
        raise Exception("can't find path to load, path:", path)

    is_list = isinstance(load_vars, list) or isinstance(load_vars, tuple)
    if not is_list:
        load_vars = [load_vars]
    assert isinstance(load_vars, list) or isinstance(load_vars, tuple)

    have_none = False
    not_sok_variable = False
    for load_var in load_vars:
        if load_vars == None:
            have_none = True
    if have_none:
        raise Exception("the input of your sok variables have none ,please")

    if isinstance(optimizer, list) or isinstance(optimizer, tuple):
        if len(optimizer) > 1:
            raise Exception("Only support load one optmizer state")
        if len(optimizer) == 0:
            optimizer = None
        optimizer = optimizer[0]

    load_table(path, load_vars, optimizer)
    print("[SOK INFO] SOK load weight from path:", path, " success!")
    return
