"""
 Copyright (c) 2021, NVIDIA CORPORATION.
 
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
import argparse
from gen_data import save_to_file, restore_from_file
import numpy as np
from multiprocessing import Process


def main(filename, split_num, save_prefix):
    def split_func(split_id, samples, labels):
        each_split_sample_num = samples.shape[0] // split_num
        my_samples = samples[
            split_id * each_split_sample_num : (split_id + 1) * each_split_sample_num
        ]
        my_labels = labels[
            split_id * each_split_sample_num : (split_id + 1) * each_split_sample_num
        ]
        save_to_file(save_prefix + str(split_id) + r".file", my_samples, my_labels)

    if 1 == split_num:
        print("[WARNING]: There is no need to split this file into 1 shards.")
        return

    samples, labels = restore_from_file(filename)
    if samples.shape[0] % split_num != 0:
        raise RuntimeError(
            "The number of samples: %d is not divisible by "
            + "split_num: %d" % (samples.shape[0], split_num)
        )

    process_list = list()
    for i in range(split_num):
        p = Process(target=split_func, args=(i, samples, labels))
        process_list.append(p)
        p.start()

    for p in process_list:
        if p.is_alive():
            p.join()

    print("[INFO]: Split dataset finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--filename", type=str, required=True, help="the filename of the whole dataset"
    )
    parser.add_argument(
        "--split_num", type=int, required=True, help="the number of shards to be splited."
    )
    parser.add_argument(
        "--save_prefix", type=str, required=True, help="the prefix string used to save shards."
    )

    args = parser.parse_args()

    main(args.filename, args.split_num, args.save_prefix)
