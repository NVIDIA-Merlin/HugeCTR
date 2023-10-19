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
import pdb

wdl_slot_size = [
    203750,
    18573,
    14082,
    7020,
    18966,
    4,
    6382,
    1246,
    49,
    185920,
    71354,
    67346,
    11,
    2166,
    7340,
    60,
    4,
    934,
    15,
    204208,
    141572,
    199066,
    60940,
    9115,
    72,
    34,
    278899,
    355877,
]
wdl_offset = np.insert(np.cumsum(wdl_slot_size), 0, 0)[:-1]
dcn_slot_size = [
    203931,
    18598,
    14092,
    7012,
    18977,
    4,
    6385,
    1245,
    49,
    186213,
    71328,
    67288,
    11,
    2168,
    7338,
    61,
    4,
    932,
    15,
    204515,
    141526,
    199433,
    60919,
    9137,
    71,
    34,
]
dcn_offset = np.insert(np.cumsum(dcn_slot_size), 0, 0)[:-1]


def compare_array_approx(results, reference, model_name, abs_th, rel_th):
    mean_relative_error = np.mean(np.abs(results - reference) / (np.abs(reference) + 1e-10))
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


def read_samples_for_dcn(
    data_file, num_samples=64, key_type="I64", slot_num=26, slot_shift=dcn_offset
):
    key_type_map = {"I32": ["I", 4], "I64": ["q", 8]}
    df = pd.read_parquet(data_file, engine="pyarrow")
    columns = df.columns
    batch_label = np.reshape(
        df[columns[0]].loc[0 : num_samples - 1].to_numpy(), newshape=(num_samples, 1)
    )
    batch_dense = np.reshape(
        df[columns[1:14]].loc[0 : num_samples - 1].to_numpy(),
        newshape=(num_samples, 13),
    )
    batch_keys = np.reshape(
        (df[columns[14:40]].loc[0 : num_samples - 1] + slot_shift).to_numpy(),
        newshape=(num_samples, 26, 1),
    )
    # print("label",batch_label)
    # print("dense",batch_dense)
    # print("keys",batch_keys)
    return batch_label, batch_dense, batch_keys


def read_samples_for_wdl(
    data_file, num_samples=64, key_type="I64", slot_num=26, slot_shift=wdl_offset
):
    key_type_map = {"I32": ["I", 4], "I64": ["q", 8]}
    df = pd.read_parquet(data_file, engine="pyarrow")
    columns = df.columns
    batch_label = np.reshape(
        df[columns[0]].loc[0 : num_samples - 1].to_numpy(), newshape=(num_samples, 1)
    )
    batch_dense = np.reshape(
        df[columns[1:14]].loc[0 : num_samples - 1].to_numpy(),
        newshape=(num_samples, 13),
    )
    batch_deep_data = np.reshape(
        (df[columns[14:40]].loc[0 : num_samples - 1] + slot_shift[0:26]).to_numpy(),
        newshape=(num_samples, 26, 1),
    )
    batch_wide_data = np.reshape(
        (df[columns[40:42]].loc[0 : num_samples - 1] + slot_shift[26:28]).to_numpy(),
        newshape=(num_samples, 2, 1),
    )
    return batch_label, batch_dense, batch_wide_data, batch_deep_data


ncf_slot_size = [162543, 56573]
ncf_offset = np.insert(np.cumsum(ncf_slot_size), 0, 0)[:-1]


def read_samples_for_ncf(data_file, num_samples=64, key_type="I32", slot_num=2):
    key_type_map = {"I32": ["I", 4], "I64": ["q", 8]}
    df = pd.read_parquet(data_file, engine="pyarrow")
    columns = df.columns
    batch_label = np.reshape(
        df[columns[3]].loc[0 : num_samples - 1].to_numpy(), newshape=(num_samples, 1)
    )
    batch_keys = np.reshape(
        (df[columns[0:2]].loc[0 : num_samples - 1] + ncf_offset[0:2]).to_numpy(),
        newshape=(num_samples, 2, 1),
    )
    return batch_label, batch_keys


din_slot_size = [
    192403,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    63001,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    801,
]
slot_shift = np.insert(np.cumsum(din_slot_size), 0, 0)[:-1]


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
