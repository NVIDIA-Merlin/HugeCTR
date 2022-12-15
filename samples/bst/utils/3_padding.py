import random
import pickle
import pandas as pd
import numpy as np

random.seed(1234)
MAX_SIZE = 10
with open("../raw_data/remap.pkl", "rb") as f:
    reviews_df = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count, example_count = pickle.load(f)

train_set = []
test_set = []
i = 0
for reviewerID, hist in reviews_df.groupby("reviewerID"):
    pos_list = hist["asin"].tolist()
    if len(pos_list) < 2:
        continue

    def gen_neg():
        neg = pos_list[0]
        while neg in pos_list:
            neg = random.randint(0, item_count - 1)
        return neg

    neg_list = [gen_neg() for i in range(len(pos_list))]
    train_hist = pos_list[:-2]
    test_hist = pos_list[:-1]
    train_seq_len = len(train_hist)
    if len(train_hist) >= MAX_SIZE:
        train_hist = train_hist[:MAX_SIZE]
        train_seq_len = MAX_SIZE
    else:
        train_hist = train_hist + (MAX_SIZE - len(train_hist)) * [-1]
    test_seq_len = len(test_hist)
    if len(test_hist) >= MAX_SIZE:
        test_hist = test_hist[:MAX_SIZE]
        test_seq_len = MAX_SIZE
    else:
        test_hist = test_hist + (MAX_SIZE - len(test_hist)) * [-1]
    cate_list = np.append(cate_list, -1)
    cat_train_hist_pos = [cate_list[i] for i in (train_hist + [pos_list[-2]])]
    cat_train_hist_neg = [cate_list[i] for i in (train_hist + [neg_list[-2]])]
    cat_test_hist_pos = [cate_list[i] for i in (test_hist + [pos_list[-1]])]
    cat_test_hist_neg = [cate_list[i] for i in (test_hist + [neg_list[-1]])]

    train_set.append(
        [train_seq_len] + [reviewerID] + train_hist + [pos_list[-2]] + cat_train_hist_pos + [1]
    )
    train_set.append(
        [train_seq_len] + [reviewerID] + train_hist + [neg_list[-2]] + cat_train_hist_neg + [0]
    )
    test_set.append(
        [test_seq_len] + [reviewerID] + test_hist + [pos_list[-1]] + cat_test_hist_pos + [1]
    )
    test_set.append(
        [test_seq_len] + [reviewerID] + test_hist + [neg_list[-1]] + cat_test_hist_neg + [0]
    )

assert len(test_set) == user_count * 2

train_df = pd.DataFrame(
    train_set,
    columns=["SLEN"]
    + ["UID"]
    + ["MID" + str(i) for i in range(MAX_SIZE + 1)]
    + ["CID" + str(i) for i in range(MAX_SIZE + 1)]
    + ["LABEL"],
)
test_df = pd.DataFrame(
    test_set,
    columns=["SLEN"]
    + ["UID"]
    + ["MID" + str(i) for i in range(MAX_SIZE + 1)]
    + ["CID" + str(i) for i in range(MAX_SIZE + 1)]
    + ["LABEL"],
)
train_df.to_parquet("../bst_data/train_temp.parquet")
test_df.to_parquet("../bst_data/test_temp.parquet")
