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
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "../../../sparse_operation_kit/unit_test/test_scripts/")))
from utils import generate_random_samples, save_to_file, restore_from_file, tf_dataset

def main(args):
    sparse_keys = True if 1 == args.sparse_keys else False

    counts = args.iter_num // 10

    total_samples, total_labels = None, None

    for _ in range(counts):
        random_samples, random_labels = generate_random_samples(
                                num_of_samples=args.global_batch_size * 10,
                                vocabulary_size=args.vocabulary_size,
                                slot_num=args.slot_num,
                                max_nnz=args.nnz_per_slot,
                                use_sparse_mask=sparse_keys)

        if total_samples is None:
            total_samples = random_samples
            total_labels = random_labels
        else:
            total_samples = np.concatenate([total_samples, random_samples], axis=0)
            total_labels = np.concatenate([total_labels, random_labels], axis=0)

    left = args.iter_num - (counts * 10)
    if (left > 0):
        random_samples, random_labels = generate_random_samples(
                                num_of_samples=args.global_batch_size * left,
                                vocabulary_size=args.vocabulary_size,
                                slot_num=args.slot_num,
                                max_nnz=args.nnz_per_slot,
                                use_sparse_mask=sparse_keys)

        if total_samples is None:
            total_samples = random_samples
            total_labels = random_labels
        else:
            total_samples = np.concatenate([total_samples, random_samples], axis=0)
            total_labels = np.concatenate([total_labels, random_labels], axis=0)
    
    save_to_file(args.filename, total_samples, total_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--global_batch_size", type=int,
                        required=True,
                        help="the global batch-size used in each iteration.")
    parser.add_argument("--slot_num", type=int,
                        required=True,
                        help="the number of feature fields.")
    parser.add_argument("--nnz_per_slot", type=int,
                        required=True,
                        help="the number of valid keys in each slot")
    parser.add_argument("--vocabulary_size", type=int,
                        required=False, default=4096)
    parser.add_argument("--iter_num", type=int,
                        required=True, 
                        help="the number of training iterations.")
    parser.add_argument("--filename", type=str,
                        required=False, default=r"./data.file",
                        help="the filename of saved datas.")
    parser.add_argument("--sparse_keys", type=int, choices=[0, 1],
                       required=False, default=0,
                       help="whether to generate sparse keys, where -1 is used"+\
                            " to denote invalid keys.")

    args = parser.parse_args()

    main(args)


    
    

