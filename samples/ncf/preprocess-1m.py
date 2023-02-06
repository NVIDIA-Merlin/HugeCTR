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

import pandas as pd
import torch
import tqdm
import sys

write_dir = "data/ml-1m/"

MIN_RATINGS = 20
USER_COLUMN = "userId"
ITEM_COLUMN = "movieId"

df = pd.read_csv("data/ml-1m/ratings.csv")
print("Filtering out users with less than {} ratings".format(MIN_RATINGS))
grouped = df.groupby(USER_COLUMN)
df = grouped.filter(lambda x: len(x) >= MIN_RATINGS)

print("Mapping original user and item IDs to new sequential IDs")
df[USER_COLUMN], unique_users = pd.factorize(df[USER_COLUMN])
df[ITEM_COLUMN], unique_items = pd.factorize(df[ITEM_COLUMN])

nb_users = len(unique_users)
nb_items = len(unique_items)

print("Number of users: %d\nNumber of items: %d" % (len(unique_users), len(unique_items)))

# Save the mapping to do the inference later on
import pickle

with open("./mappings.pickle", "wb") as handle:
    pickle.dump(
        {"users": unique_users, "items": unique_items}, handle, protocol=pickle.HIGHEST_PROTOCOL
    )

# Need to sort before popping to get the last item
df.sort_values(by="timestamp", inplace=True)

# clean up data
del df["rating"], df["timestamp"]
df = df.drop_duplicates()  # assuming it keeps order

# now we have filtered and sorted by time data, we can split test data out
grouped_sorted = df.groupby(USER_COLUMN, group_keys=False)
test_data = grouped_sorted.tail(1).sort_values(by=USER_COLUMN)
# need to pop for each group
train_data = grouped_sorted.apply(lambda x: x.iloc[:-1])

train_data["target"] = 1
test_data["target"] = 1
print(train_data.head())


class _TestNegSampler:
    def __init__(self, train_ratings, nb_users, nb_items, nb_neg):
        self.nb_neg = nb_neg
        self.nb_users = nb_users
        self.nb_items = nb_items

        # compute unique ids for quickly created hash set and fast lookup
        ids = (train_ratings[:, 0] * self.nb_items) + train_ratings[:, 1]
        self.set = set(ids)

    def generate(self, batch_size=128 * 1024):
        users = (
            torch.arange(0, self.nb_users)
            .reshape([1, -1])
            .repeat([self.nb_neg, 1])
            .transpose(0, 1)
            .reshape(-1)
        )

        items = [-1] * len(users)

        random_items = torch.LongTensor(batch_size).random_(0, self.nb_items).tolist()
        print("Generating validation negatives...")
        for idx, u in enumerate(tqdm.tqdm(users.tolist())):
            if not random_items:
                random_items = torch.LongTensor(batch_size).random_(0, self.nb_items).tolist()
            j = random_items.pop()
            while u * self.nb_items + j in self.set:
                if not random_items:
                    random_items = torch.LongTensor(batch_size).random_(0, self.nb_items).tolist()
                j = random_items.pop()

            items[idx] = j
        items = torch.LongTensor(items)
        return items


sampler = _TestNegSampler(
    df.values, nb_users, nb_items, 102
)  # using 102 negative samples for even batches
train_negs = sampler.generate()
train_negs = train_negs.reshape(-1, 102)

sampler = _TestNegSampler(df.values, nb_users, nb_items, 1)  # using 1 negative test sample
test_negs = sampler.generate()
test_negs = test_negs.reshape(-1, 1)


import numpy as np

# generating negative samples for training
train_data_neg = np.zeros((train_negs.shape[0] * train_negs.shape[1], 3), dtype=int)
idx = 0
for i in tqdm.tqdm(range(train_negs.shape[0])):
    for j in range(train_negs.shape[1]):
        train_data_neg[idx, 0] = i  # user ID
        train_data_neg[idx, 1] = train_negs[i, j]  # negative item ID
        idx += 1

# generating negative samples for testing
test_data_neg = np.zeros((test_negs.shape[0] * test_negs.shape[1], 3), dtype=int)
idx = 0
for i in tqdm.tqdm(range(test_negs.shape[0])):
    for j in range(test_negs.shape[1]):
        test_data_neg[idx, 0] = i
        test_data_neg[idx, 1] = test_negs[i, j]
        idx += 1

train_data_np = np.concatenate([train_data_neg, train_data.values])
np.random.shuffle(train_data_np)

test_data_np = np.concatenate([test_data_neg, test_data.values])
np.random.shuffle(test_data_np)

# HugeCTR expect user ID and item ID to be different, so we use 0 -> nb_users for user IDs and
# nb_users -> nb_users+nb_items for item IDs.
train_data_np[:, 1] += nb_users
test_data_np[:, 1] += nb_users

print(np.max(train_data_np[:, 1]))


from ctypes import c_longlong as ll
from ctypes import c_uint
from ctypes import c_float
from ctypes import c_int


def write_hugeCTR_data(huge_ctr_data, filename="huge_ctr_data.dat"):
    print("Writing %d samples" % huge_ctr_data.shape[0])
    with open(filename, "wb") as f:
        # write header
        f.write(ll(0))  # 0: no error check; 1: check_num
        f.write(ll(huge_ctr_data.shape[0]))  # the number of samples in this data file
        f.write(ll(1))  # dimension of label
        f.write(ll(1))  # dimension of dense feature
        f.write(ll(2))  # long long slot_num
        for _ in range(3):
            f.write(ll(0))  # reserved for future use

        for i in tqdm.tqdm(range(huge_ctr_data.shape[0])):
            f.write(c_float(huge_ctr_data[i, 2]))  # float label[label_dim];
            f.write(c_float(0))  # dummy dense feature
            f.write(c_int(1))  # slot 1 nnz: user ID
            f.write(c_uint(huge_ctr_data[i, 0]))
            f.write(c_int(1))  # slot 2 nnz: item ID
            f.write(c_uint(huge_ctr_data[i, 1]))


def generate_filelist(filelist_name, num_files, filename_prefix):
    with open(filelist_name, "wt") as f:
        f.write("{0}\n".format(num_files))
        for i in range(num_files):
            f.write("{0}_{1}.dat\n".format(filename_prefix, i))


import os

if not os.path.exists(write_dir):
    os.makedirs(write_dir)

for i, data_arr in enumerate(np.array_split(train_data_np, 10)):
    write_hugeCTR_data(data_arr, filename=write_dir + "train_huge_ctr_data_%d.dat" % i)

generate_filelist(write_dir + "train_filelist.txt", 10, write_dir + "train_huge_ctr_data")


for i, data_arr in enumerate(np.array_split(test_data_np, 10)):
    write_hugeCTR_data(data_arr, filename=write_dir + "test_huge_ctr_data_%d.dat" % i)

generate_filelist(write_dir + "test_filelist.txt", 10, write_dir + "test_huge_ctr_data")
