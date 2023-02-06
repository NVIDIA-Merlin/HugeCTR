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

from trainer import Trainer
from model import DLRM
from dataset import BinaryDataset, SyntheticDataset
import tensorflow as tf
import numpy as np
import time
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--global_batch_size", type=int)
parser.add_argument("--xla", action="store_true", help="enable xla of tensorflow")
parser.add_argument(
    "--compress",
    action="store_true",
    help="use tf.unique/tf.gather to compress/decompress embedding keys",
)
parser.add_argument(
    "--eval_in_last", action="store_true", help="evaluate only after the last iteration"
)
parser.add_argument(
    "--use_synthetic_dataset", action="store_true", help="use synthetic dataset for profiling"
)
parser.add_argument("--use_tf_embedding", action="store_true")
parser.add_argument("--use_tf_optimizer", action="store_true")
parser.add_argument("--early_stop", type=int, default=-1)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=24.0)
args = parser.parse_args()
args.lr_schedule_steps = [
    int(2750 * 55296 / args.global_batch_size),
    int(49315 * 55296 / args.global_batch_size),
    int(27772 * 55296 / args.global_batch_size),
]
print("[Info] args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if args.xla:
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=fusible"

start_time = time.time()


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices("GPU")
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    global_batch_size = args.global_batch_size

    if args.use_synthetic_dataset or args.data_dir is None:
        metadata = {"vocab_sizes": [1000] * 26}
        print("[Info] Using synthetic dataset")
        dataset = SyntheticDataset(
            batch_size=global_batch_size,
            num_iterations=args.early_stop if args.early_stop > 0 else 4000,
            vocab_sizes=metadata["vocab_sizes"],
            prefetch=20,
        )
        test_dataset = SyntheticDataset(
            batch_size=global_batch_size,
            num_iterations=args.early_stop if args.early_stop > 0 else 128,
            vocab_sizes=metadata["vocab_sizes"],
            prefetch=20,
        )
    else:
        with open(os.path.join(args.data_dir, "train/metadata.json"), "r") as f:
            metadata = json.load(f)
        print(metadata)
        print("[Info] Using dataset in %s" % args.data_dir)
        dtype = {"int32": np.int32, "float32": np.float32}
        dataset_dir = args.data_dir
        dataset = BinaryDataset(
            os.path.join(dataset_dir, "train/label.bin"),
            os.path.join(dataset_dir, "train/dense.bin"),
            os.path.join(dataset_dir, "train/category.bin"),
            batch_size=global_batch_size,
            drop_last=True,
            global_rank=0,
            global_size=1,
            prefetch=20,
            label_raw_type=dtype[metadata["label_raw_type"]],
            dense_raw_type=dtype[metadata["dense_raw_type"]],
            category_raw_type=dtype[metadata["category_raw_type"]],
            log=metadata["dense_log"],
        )
        test_dataset = BinaryDataset(
            os.path.join(dataset_dir, "test/label.bin"),
            os.path.join(dataset_dir, "test/dense.bin"),
            os.path.join(dataset_dir, "test/category.bin"),
            batch_size=global_batch_size,
            drop_last=False,
            global_rank=0,
            global_size=1,
            prefetch=20,
            label_raw_type=dtype[metadata["label_raw_type"]],
            dense_raw_type=dtype[metadata["dense_raw_type"]],
            category_raw_type=dtype[metadata["category_raw_type"]],
            log=metadata["dense_log"],
        )

    model = DLRM(
        metadata["vocab_sizes"],
        num_dense_features=13,
        embedding_vec_size=128,
        bottom_stack_units=[512, 256, 128],
        top_stack_units=[1024, 1024, 512, 256, 1],
        num_gpus=1,
        compress=args.compress,
        use_tf=args.use_tf_embedding,
    )

    trainer = Trainer(
        model,
        dataset,
        test_dataset,
        auc_thresholds=8000,
        base_lr=args.lr,
        warmup_steps=args.lr_schedule_steps[0],
        decay_start_step=args.lr_schedule_steps[1],
        decay_steps=args.lr_schedule_steps[2],
        use_tf_optimizer=args.use_tf_optimizer,
    )

    if args.eval_in_last:
        trainer.train(
            eval_interval=None, eval_in_last=True, early_stop=args.early_stop, epochs=args.epochs
        )
    else:
        trainer.train(eval_in_last=False, early_stop=args.early_stop, epochs=args.epochs)

    print("main time: %.2fs" % (time.time() - start_time))
