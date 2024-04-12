#!/usr/bin/env python3

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import json
import logging
import os
import pathlib

import tensorflow as tf
import numpy as np

# method from PEP-366 to support relative import in executed modules
if __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

import dataloading.feature_spec
from dataloading.dataloader import create_input_pipelines

LOGGER = logging.getLogger("run_performance_on_triton")

def create_input_data_hps(batch_sizes, dataset_path, dataset_type, result_path, feature_spec,
                          total_benchmark_samples, fused_embedding):

    input_data = {}
    for batch_size in batch_sizes:
        filename = os.path.join(result_path, str(batch_size) + ".json")
        print("generating input data: ", filename)
        shapes = create_input_data_hps_batch(batch_size=batch_size, dst_path=filename, dataset_path=dataset_path,
                                             dataset_type=dataset_type, feature_spec=feature_spec,
                                             total_benchmark_samples=total_benchmark_samples,
                                             fused_embedding=fused_embedding)
        input_data[batch_size] = (filename, shapes)
    return input_data


def create_input_data_hps_batch(batch_size, dst_path, dataset_path, dataset_type, feature_spec,
                      total_benchmark_samples, fused_embedding):

    fspec = dataloading.feature_spec.FeatureSpec.from_yaml(
        os.path.join(dataset_path, feature_spec)
    )
    num_tables = len(fspec.get_categorical_sizes())
    table_ids = list(range(num_tables))

    _, dataloader = create_input_pipelines(dataset_type=dataset_type, dataset_path=dataset_path,
                                           train_batch_size=batch_size, test_batch_size=batch_size,
                                           table_ids=table_ids, feature_spec=feature_spec, rank=0, world_size=1)

    generated = 0
    batches = []

    categorical_cardinalities = fspec.get_categorical_sizes()
    categorical_cardinalities = np.roll(np.cumsum(np.array(categorical_cardinalities)), 1)
    categorical_cardinalities[0] = 0

    for batch in dataloader.op():
        features, labels = batch
        numerical_features, cat_features = features
        cat_features = tf.concat(cat_features, axis=1).numpy().astype(np.int32)
        cat_features = np.add(cat_features, categorical_cardinalities).flatten()
        numerical_features = numerical_features.numpy().astype(np.float32).flatten()

        batch = {
            "categorical_features": cat_features.tolist(),
            "numerical_features": numerical_features.tolist(),
        }
        batches.append(batch)
        generated += batch_size
        if generated >= total_benchmark_samples:
            break

    with open(dst_path, "w") as f:
        json.dump(obj={"data": batches}, fp=f, indent=4)

    shapes = [
        f"categorical_features:{cat_features.shape[0]}",
        f"numerical_features:{numerical_features.shape[0]}",
    ]
    return shapes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-path",
        type=pathlib.Path,
        required=True,
        help="Path where processed data is stored.",
    )
    parser.add_argument(
        "--fused-embedding",
        action="store_true",
        help="Use the fused embedding API for HPS",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        default=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
        help="List of batch sizes to test.",
        nargs="*",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose logs",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--dataset_path", default=None, required=True, help="Path to dataset directory"
    )
    parser.add_argument(
        "--feature_spec",
        default="feature_spec.yaml",
        help="Name of the feature spec file in the dataset directory",
    )
    parser.add_argument(
        "--dataset_type",
        default="tf_raw",
        choices=["tf_raw", "synthetic", "split_tfrecords"],
        help="The type of the dataset to use",
    )

    parser.add_argument(
        "--num-benchmark-samples",
        default=2**18,
        type=int,
        help="The type of the dataset to use",
    )

    args = parser.parse_args()

    log_level = logging.INFO if not args.verbose else logging.DEBUG
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    input_data = create_input_data_hps(batch_sizes=args.batch_sizes, dataset_path=args.dataset_path, result_path=args.result_path,
                                   dataset_type=args.dataset_type, feature_spec=args.feature_spec,
                                   total_benchmark_samples=args.num_benchmark_samples,
                                   fused_embedding=args.fused_embedding)

if __name__ == "__main__":
    main()
