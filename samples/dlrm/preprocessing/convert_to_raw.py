# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
import logging
import os
import time

import numpy as np

"""
Script to convert the reference TorchRec NumPy dataset to a binary raw format for HugeCTR training.

The script requires a machine with about 200GB RAM as it reads all three
day_*_labels.npy, day_*_dense.npy and day_*_sparse_multi_hot.npz files into memory.
It should complete in about 5h hours (depending on I/O bandwidth).

For the MLPerf Training v3.0 the expected md5sum of the output files are:

| file           | md5sum                           |
|:---------------|:---------------------------------|
| test_data.bin  | cf636876d8baf0776287be23b31c2f14 |
| train_data.bin | 4d48daf07cc244f6fa933b832d7fe5a3 |
| val_data.bin   | c7ca591ad3fd2b09b75d99fa4fc210e2 |
"""

INPUT_LABELS_FILE = "day_{day}_labels.npy"
INPUT_DENSE_FILE = "day_{day}_dense.npy"
INPUT_SPARSE_FILE = "day_{day}_sparse_multi_hot.npz"
OUTPUT_FILE = "{stage}_data.bin"
NUM_DAYS = 24
NUM_SPARSE = 26
TRAIN, VAL, TEST = "train", "val", "test"
STAGES = (TRAIN, VAL, TEST)
LAST_DAY_TEST_VAL_SPLIT_POINT = 89_137_319


class DataConverter:
    def __init__(
        self,
        input_dir_labels_and_dense: str,
        input_dir_sparse_multihot: str,
        output_dir: str,
        stage: str,
        buffer_size: int,
        chunk_size: int,
        logger: logging.Logger,
        logging_interval: int,
    ):
        self.input_dir_labels_and_dense = input_dir_labels_and_dense
        self.input_dir_sparse_multihot = input_dir_sparse_multihot
        self.output_file = os.path.join(output_dir, OUTPUT_FILE.format(stage=stage))
        self.logger = logger
        self.logging_interval = logging_interval
        self.stage = stage
        self.buffer_size = buffer_size
        self.chunk_size = chunk_size
        self.days = self._get_days_for_stage()
        self.slice_ = self._get_slice_for_stage()

    def _get_days_for_stage(self):
        if self.stage == TRAIN:
            return list(range(NUM_DAYS - 1))
        else:
            return [NUM_DAYS - 1]

    def _get_slice_for_stage(self):
        slice_ = None
        if self.stage == VAL:
            slice_ = slice(None, LAST_DAY_TEST_VAL_SPLIT_POINT)
        elif self.stage == TEST:
            slice_ = slice(LAST_DAY_TEST_VAL_SPLIT_POINT, None)
        self.logger.debug(f"stage = {self.stage}, slice_ = {slice_}")
        return slice_

    def _read_metadata(self, f):
        np.lib.format.read_magic(f)
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
        assert not fortran_order, "C-like index order expected"
        self.logger.debug(f"Data shape = {shape}")
        return shape, dtype

    def _load_data_for_day(self, day):
        labels_file = INPUT_LABELS_FILE.format(day=day)
        dense_file = INPUT_DENSE_FILE.format(day=day)
        sparse_file = INPUT_SPARSE_FILE.format(day=day)

        self.logger.debug(f"Loading {labels_file}...")
        with open(os.path.join(self.input_dir_labels_and_dense, labels_file), "rb") as f:
            _, dtype = self._read_metadata(f)
            label = np.fromfile(f, dtype=dtype)
        self.logger.debug("Loading done")

        self.logger.debug(f"Loading {dense_file}...")
        with open(os.path.join(self.input_dir_labels_and_dense, dense_file), "rb") as f:
            shape, dtype = self._read_metadata(f)
            dense = np.fromfile(f, dtype=dtype).reshape(shape, order="C")
        self.logger.debug("Loading done")

        self.logger.debug(f"Loading {sparse_file}...")
        sparse_dict = np.load(os.path.join(self.input_dir_sparse_multihot, sparse_file))
        sparse_list = [sparse_dict[str(i)] for i in range(NUM_SPARSE)]
        self.logger.debug("Loading done")

        if self.slice_ is not None:
            self.logger.debug("Slicing data...")
            label = label[self.slice_]
            dense = dense[self.slice_]
            sparse_list = [sparse[self.slice_] for sparse in sparse_list]
            self.logger.debug("Slicing done")

        return label, dense, sparse_list

    def save(self):
        self.logger.info(f"Writing data to {self.output_file}...")
        samples_total = 0
        start_time = time.perf_counter()
        with open(self.output_file, "wb", buffering=self.buffer_size) as out:
            write = out.write
            for day in self.days:
                self.logger.info(f"Processing data for day = {day}...")
                label, dense, sparse_list = self._load_data_for_day(day)
                # We concatenate sparse features as it saves time on writing
                # data below. It is done in chunks to save memory.
                start = 0
                end = self.chunk_size
                while start < len(label):
                    self.logger.debug("Concatenating sparse features...")
                    sparse = np.concatenate(
                        [sparse_feat[start:end] for sparse_feat in sparse_list], axis=1
                    )
                    self.logger.debug("Concatenating done")
                    for samples_total, (label_row, dense_row, sparse_row) in enumerate(
                        zip(label[start:end], dense[start:end], sparse), samples_total + 1
                    ):
                        write(label_row.tobytes())
                        write(dense_row.tobytes())
                        write(sparse_row.tobytes())
                        if samples_total % self.logging_interval == 0:
                            self.logger.info(f"Number of samples done: {samples_total:,}")
                    start = end
                    end += self.chunk_size
        end_time = time.perf_counter()
        self.logger.info(f"Creating {self.output_file} done.")
        self.logger.info(
            f"Total number of samples done for stage = {self.stage}: {samples_total:,}"
        )
        self.logger.info(f"Throughput: {samples_total / (end_time - start_time):.2f} [samples/sec]")


def get_logger(level):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    s_handler = logging.StreamHandler()
    log_format = logging.Formatter("[%(asctime)s][%(levelname)s]: %(message)s")
    s_handler.setFormatter(log_format)
    logger.addHandler(s_handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NumPy to Raw format conversion script.")
    parser.add_argument(
        "--input_dir_labels_and_dense",
        type=str,
        required=True,
        help="Input directory with labels and dense data",
    )
    parser.add_argument(
        "--input_dir_sparse_multihot",
        type=str,
        required=True,
        help="Input directory with sparse multi-hot data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the raw binary dataset",
    )
    parser.add_argument(
        "--stages",
        type=str,
        choices=STAGES,
        default=STAGES,
        nargs="+",
        help="Stages to process",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=2_147_483_647,
        help="Buffer size for writing data",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2_000_000,
        help="Chunk size for concatenating sparse features before saving",
    )
    parser.add_argument(
        "--logging_level",
        type=int,
        default=logging.INFO,
        help="Logging level",
    )
    parser.add_argument(
        "--logging_interval",
        type=int,
        default=10_000_000,
        help="Logging interval for the number of samples done",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger(level=args.logging_level)
    logger.info("NumPy to Raw format conversion script")
    logger.info(f"args are = {vars(args)}")

    os.makedirs(args.output_dir, exist_ok=True)
    for stage in args.stages:
        converter = DataConverter(
            input_dir_labels_and_dense=args.input_dir_labels_and_dense,
            input_dir_sparse_multihot=args.input_dir_sparse_multihot,
            output_dir=args.output_dir,
            stage=stage,
            buffer_size=args.buffer_size,
            chunk_size=args.chunk_size,
            logger=logger,
            logging_interval=args.logging_interval,
        )
        converter.save()

    logger.info("Done.")


if __name__ == "__main__":
    main()
