"""
 Copyright (c) 2023, NVIDIA CORPORATION.
 
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

import struct
import numpy as np
import pandas as pd


def compare_array_approx(results, reference, model_name, abs_th, rel_th):
    mean_relative_error = np.mean(np.abs(results - reference) / (np.abs(reference)))
    mean_absolute_error = np.mean(np.abs(results - reference))
    if mean_absolute_error > abs_th and mean_relative_error > rel_th:
        raise RuntimeError(
            "Too large inference error between ONNX and HugeCTR for {}".format(model_name)
        )
        sys.exit(1)
    else:
        print("Successfully convert HugeCTR model to ONNX for {}".format(model_name))
        print("Total number of samples for inference: {}".format(results.shape[0]))
        print(
            "Mean absosulte error between HugeCTR and ONNX Runtime inference: {}".format(
                mean_absolute_error
            )
        )
        print(
            "Mean relative error between HugeCTR and ONNX Runtime inference: {}".format(
                mean_relative_error
            )
        )


def read_samples_for_dcn(data_file, num_samples=64, key_type="I32", slot_num=26):
    key_type_map = {"I32": ["I", 4], "I64": ["q", 8]}
    with open(data_file, "rb") as file:
        # skip data_header
        file.seek(4 + 64 + 1, 0)
        batch_label = []
        batch_dense = []
        batch_keys = []
        for _ in range(num_samples):
            # one sample
            length_buffer = file.read(4)  # int
            length = struct.unpack("i", length_buffer)
            label_buffer = file.read(4)  # int
            label = struct.unpack("i", label_buffer)[0]
            dense_buffer = file.read(4 * 13)  # dense_dim * float
            dense = struct.unpack("13f", dense_buffer)
            keys = []
            for _ in range(slot_num):
                nnz_buffer = file.read(4)  # int
                nnz = struct.unpack("i", nnz_buffer)[0]
                key_buffer = file.read(key_type_map[key_type][1] * nnz)  # nnz * sizeof(key_type)
                key = struct.unpack(str(nnz) + key_type_map[key_type][0], key_buffer)
                keys += list(key)
            check_bit_buffer = file.read(1)  # char
            check_bit = struct.unpack("c", check_bit_buffer)[0]
            batch_label.append(label)
            batch_dense.append(dense)
            batch_keys.append(keys)
    batch_label = np.reshape(np.array(batch_label, dtype=np.float32), newshape=(num_samples, 1))
    batch_dense = np.reshape(np.array(batch_dense, dtype=np.float32), newshape=(num_samples, 13))
    batch_keys = np.reshape(np.array(batch_keys, dtype=np.int64), newshape=(num_samples, 26, 1))
    return batch_label, batch_dense, batch_keys


def read_samples_for_wdl(data_file, num_samples=64, key_type="I32", slot_num=26):
    key_type_map = {"I32": ["I", 4], "I64": ["q", 8]}
    with open(data_file, "rb") as file:
        # skip data_header
        file.seek(4 + 64 + 1, 0)
        batch_label = []
        batch_dense = []
        batch_wide_data = []
        batch_deep_data = []
        for _ in range(num_samples):
            # one sample
            length_buffer = file.read(4)  # int
            length = struct.unpack("i", length_buffer)
            label_buffer = file.read(4)  # int
            label = struct.unpack("i", label_buffer)[0]
            dense_buffer = file.read(4 * 13)  # dense_dim * float
            dense = struct.unpack("13f", dense_buffer)
            keys = []
            for _ in range(slot_num):
                nnz_buffer = file.read(4)  # int
                nnz = struct.unpack("i", nnz_buffer)[0]
                key_buffer = file.read(key_type_map[key_type][1] * nnz)  # nnz * sizeof(key_type)
                key = struct.unpack(str(nnz) + key_type_map[key_type][0], key_buffer)
                keys += list(key)
            check_bit_buffer = file.read(1)  # char
            check_bit = struct.unpack("c", check_bit_buffer)[0]
            batch_label.append(label)
            batch_dense.append(dense)
            batch_wide_data.append(keys[0:2])
            batch_deep_data.append(keys[2:28])
    batch_label = np.reshape(np.array(batch_label, dtype=np.float32), newshape=(num_samples, 1))
    batch_dense = np.reshape(np.array(batch_dense, dtype=np.float32), newshape=(num_samples, 13))
    batch_wide_data = np.reshape(
        np.array(batch_wide_data, dtype=np.int64), newshape=(num_samples, 1, 2)
    )
    batch_deep_data = np.reshape(
        np.array(batch_deep_data, dtype=np.int64), newshape=(num_samples, 26, 1)
    )
    return batch_label, batch_dense, batch_wide_data, batch_deep_data


def read_samples_for_ncf(data_file, num_samples=64, key_type="I32", slot_num=2):
    key_type_map = {"I32": ["I", 4], "I64": ["q", 8]}
    with open(data_file, "rb") as file:
        # skip data_header
        file.seek(64, 0)
        batch_label = []
        batch_dense = []
        batch_keys = []
        for _ in range(num_samples):
            # one sample
            label_buffer = file.read(4)  # int
            label = struct.unpack("f", label_buffer)[0]
            dense_buffer = file.read(4)  # dense_dim * float
            dense = struct.unpack("f", dense_buffer)
            keys = []
            for _ in range(slot_num):
                nnz_buffer = file.read(4)  # int
                nnz = struct.unpack("i", nnz_buffer)[0]
                key_buffer = file.read(key_type_map[key_type][1] * nnz)  # nnz * sizeof(key_type)
                key = struct.unpack(str(nnz) + key_type_map[key_type][0], key_buffer)
                keys += list(key)
            batch_label.append(label)
            batch_dense.append(dense)
            batch_keys.append(keys)
    batch_label = np.reshape(np.array(batch_label, dtype=np.float32), newshape=(num_samples, 1))
    batch_dense = np.reshape(np.array(batch_dense, dtype=np.float32), newshape=(num_samples, 1))
    batch_keys = np.reshape(np.array(batch_keys, dtype=np.int64), newshape=(num_samples, 2, 1))
    return batch_label, batch_dense, batch_keys


slot_size_array = [192403, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 801]
slot_shift = np.insert(np.cumsum(slot_size_array), 0, 0)[:-1]


def read_samples_for_din(
    data_file, num_samples=64, key_type="I64", slot_num=23, slot_shift=slot_shift
):
    df = pd.read_parquet(data_file, engine="pyarrow")
    columns = df.columns
    dense = np.reshape(np.array([], dtype=np.float32), newshape=(num_samples, 0))
    cat_data = df[columns[0:23]].loc[0 : num_samples - 1].values + slot_shift
    user = np.reshape(cat_data[:, 0:1], newshape=(num_samples, 1, 1))
    good = np.reshape(cat_data[:, 1:12], newshape=(num_samples, 11, 1))
    cate = np.reshape(cat_data[:, 12:23], newshape=(num_samples, 11, 1))
    return dense, user, good, cate


def read_samples_for_mmoe(data_file, key_slot_shift, num_samples=64, slot_num=32):
    df = pd.read_parquet(data_file, engine="pyarrow")
    columns = df.columns

    batch_dense = np.zeros((num_samples, 0)).astype(np.float32)

    batch_keys = np.array(df[columns[2:34]].loc[0 : num_samples - 1].values + key_slot_shift)
    batch_keys = np.reshape(batch_keys, (num_samples, slot_num, 1))

    batch_label = np.array(df[columns[0:2]].loc[0 : num_samples - 1].values)
    batch_label = np.reshape(batch_label, (num_samples, 2))

    return batch_label, batch_dense, batch_keys


def read_samples_for_bst(
    data_file, num_samples=64, key_type="I64", slot_num=23, slot_shift=slot_shift
):
    df = pd.read_parquet(data_file, engine="pyarrow")
    columns = df.columns
    dense = np.reshape(
        df[columns[24:25]].loc[0 : num_samples - 1].values, newshape=(num_samples, 1)
    )
    cat_data = df[columns[0:24]].loc[0 : num_samples - 1].values + slot_shift
    cat_data = cat_data.astype("int64")
    label = np.reshape(cat_data[:, 0:1].astype("float64"), newshape=(num_samples, 1))
    user = np.reshape(cat_data[:, 1:2], newshape=(num_samples, 1, 1))
    good = np.reshape(cat_data[:, 2:12], newshape=(num_samples, 10, 1))
    target_good = np.reshape(cat_data[:, 12:13], newshape=(num_samples, 1, 1))
    cate = np.reshape(cat_data[:, 13:23], newshape=(num_samples, 10, 1))
    target_cate = np.reshape(cat_data[:, 23:24], newshape=(num_samples, 1, 1))
    return label, dense, user, good, target_good, cate, target_cate
