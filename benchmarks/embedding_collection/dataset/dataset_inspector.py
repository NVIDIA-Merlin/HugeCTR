import os
import numpy as np
import struct
import random

alpha = 1.1
output_filename = "/workdir/dataset/dcnv2_synthetic_alpha{}.bin".format(alpha)

dataset_info = [
    (7, 1, 100),
    (2, 1, 1000),
    (2, 1, 10000),
    (2, 10, 1000),
    (5, 10, 10000),
    (2, 10, 500000),
    (1, 5, 3000000),
    (4, 10, 40000000),
    (1, 100, 40000000),
]

# output_filename = '/data/train_data.bin'
#
# dataset_info = [
#     (1, 3, 40000000),
#     (1, 2, 39060),
#     (1, 1, 17295),
#     (1, 2, 7424),
#     (1, 6, 20265),
#     (1, 1, 3),
#     (1, 1, 7122),
#     (1, 1, 1543),
#     (1, 1, 63),
#     (1, 7, 40000000),
#     (1, 3, 3067956),
#     (1, 8, 405282),
#     (1, 1, 10),
#     (1, 6, 2209),
#     (1, 9, 11938),
#     (1, 5, 155),
#     (1, 1, 4),
#     (1, 1, 976),
#     (1, 1, 14),
#     (1, 12, 40000000),
#     (1, 100, 40000000),
#     (1, 27, 40000000),
#     (1, 10, 590152),
#     (1, 3, 12973),
#     (1, 1, 108),
#     (1, 1, 36),
# ]
num_dense_features = 13
num_label = 1

n = 1024  # 100 * 65536

num_cate_features = sum([num_table * hotness for (num_table, hotness, _) in dataset_info])
max_vocabulary_size = sum([v for _, _, v in dataset_info])
item_num_per_sample = 1 + num_dense_features + num_cate_features
sample_format = r"1I" + str(num_dense_features) + "f" + str(num_cate_features) + "I"
sample_size_in_bytes = 1 * 4 + num_dense_features * 4 + num_cate_features * 4

print("num_dense_features", num_dense_features)
print("num_cate_features", num_cate_features)
print("item_num_per_sample", item_num_per_sample)
print("sample_format", sample_format)

with open(output_filename, "rb") as file:
    file_stats = os.stat(output_filename)
    file_size = file_stats.st_size
    assert file_size % sample_size_in_bytes == 0
    sample_start_id = random.randint(0, file_size / sample_size_in_bytes - n - 1)
    file.seek(sample_start_id * sample_size_in_bytes)
    data_buffer = file.read(n * sample_size_in_bytes)
    data = struct.unpack(sample_format * n, data_buffer)
    data = [list(data[i * item_num_per_sample : (i + 1) * item_num_per_sample]) for i in range(n)]

cate_features = [v[1 + num_dense_features :] for v in data]
cate_features = np.asarray(cate_features)
print(cate_features.shape)

hotness_offset = 0
cate_offset = 0
table_id = 0
all_cate_features = []
for num_table, hotness, vocabulary_size in dataset_info:
    for t in range(num_table):
        current_table_keys = cate_features[:, hotness_offset : hotness_offset + hotness]
        current_table_keys = current_table_keys + cate_offset
        all_cate_features.append(current_table_keys)
        print(
            "table id",
            table_id,
            "unique ratio",
            np.unique(current_table_keys).size / current_table_keys.size,
        )
        hotness_offset += hotness
        cate_offset += vocabulary_size
        table_id += 1
        assert np.max(current_table_keys) < cate_offset
        assert np.max(current_table_keys) >= np.min(current_table_keys)
        assert np.max(current_table_keys) - np.min(current_table_keys) < vocabulary_size

all_cate_features = np.concatenate(all_cate_features, axis=1)
assert all_cate_features.size == n * num_cate_features
print(
    "bz:",
    n,
    "alpha:",
    alpha,
    ", total unique ratio ",
    np.unique(all_cate_features).size / all_cate_features.size,
)
