import os
import queue
import concurrent

import numpy as np
import tensorflow as tf
from typing import List, Optional


class BinaryDataset:
    def __init__(
        self,
        label_bin,
        dense_bin,
        category_bin,
        batch_size=1,
        drop_last=True,
        global_rank=0,
        global_size=1,
        prefetch=1,
        label_raw_type=np.int32,
        dense_raw_type=np.int32,
        category_raw_type=np.int32,
        hotness_per_table: List[int] = None,
        log=True,
    ):
        """
        * batch_size : The batch size of local rank, which means the total batch size of all ranks should be (batch_size * global_size).
        * prefetch   : If prefetch > 1, it can only be read sequentially, otherwise it will return incorrect samples.
        """
        self._hotness_per_table = hotness_per_table
        self._hotness_per_sample = np.sum(hotness_per_table)
        self._num_category = len(hotness_per_table)
        self._check_file(label_bin, dense_bin, category_bin, 4, self._hotness_per_sample)

        self._batch_size = batch_size
        self._drop_last = drop_last
        self._global_rank = global_rank
        self._global_size = global_size

        # actual number of samples in the binary file
        self._samples_in_all_ranks = os.path.getsize(label_bin) // 4

        # self._num_entries represents there are how many batches
        self._num_entries = self._samples_in_all_ranks // (batch_size * global_size)

        # number of samples in current rank
        self._num_samples = self._num_entries * batch_size

        if not self._drop_last:
            if (self._samples_in_all_ranks % (batch_size * global_size)) < global_size:
                print(
                    "The number of samples in last batch is less than global_size, so the drop_last=False will be ignored."
                )
                self._drop_last = True
            else:
                # assign the samples in the last batch to different local ranks
                samples_in_last_batch = [
                    (self._samples_in_all_ranks % (batch_size * global_size)) // global_size
                ] * global_size
                for i in range(global_size):
                    if i < (self._samples_in_all_ranks % (batch_size * global_size)) % global_size:
                        samples_in_last_batch[i] += 1
                assert sum(samples_in_last_batch) == (
                    self._samples_in_all_ranks % (batch_size * global_size)
                )
                self._samples_in_last_batch = samples_in_last_batch[global_rank]

                # the offset of last batch
                self._last_batch_offset = []
                offset = 0
                for i in range(global_size):
                    self._last_batch_offset.append(offset)
                    offset += samples_in_last_batch[i]

                # correct the values when drop_last=False
                self._num_entries += 1
                self._num_samples = (
                    self._num_entries - 1
                ) * batch_size + self._samples_in_last_batch

        self._prefetch = min(prefetch, self._num_entries)
        self._prefetch_queue = queue.Queue()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self._label_file = os.open(label_bin, os.O_RDONLY)
        self._dense_file = os.open(dense_bin, os.O_RDONLY)
        self._category_file = os.open(category_bin, os.O_RDONLY)

        self._label_raw_type = label_raw_type
        self._dense_raw_type = dense_raw_type
        self._category_raw_type = category_raw_type

        self._log = log

        self._row_lengths = []
        for i in range(len(self._hotness_per_table)):
            self._row_lengths.append(
                tf.repeat(self._hotness_per_table[i], repeats=self._batch_size)
            )

    def _check_file(
        self, label_bin, dense_bin, category_bin, label_type_byte, sparse_hotness_per_sample
    ):
        # num_samples represents the actual number of samples in the dataset

        num_samples = os.path.getsize(label_bin) // label_type_byte
        if num_samples <= 0:
            raise RuntimeError("There must be at least one sample in %s" % label_bin)

        if num_samples <= 0:
            raise RuntimeError("There must be at least one sample in %s" % label_bin)
        # check file size
        for file, bytes_per_sample in [
            [label_bin, 4],
            [dense_bin, 52],
            [category_bin, 4 * sparse_hotness_per_sample],
        ]:
            file_size = os.path.getsize(file)
            if file_size % bytes_per_sample != 0:
                raise RuntimeError(
                    "The file size of %s should be an integer multiple of %d."
                    % (file, bytes_per_sample)
                )
            if (file_size // bytes_per_sample) != num_samples:
                raise RuntimeError(
                    "The number of samples in %s is not equeal to %s" % (dense_bin, label_bin)
                )

    def __del__(self):
        for file in [self._label_file, self._dense_file, self._category_file]:
            if file is not None:
                os.close(file)

    def __len__(self):
        return self._num_entries

    def __getitem__(self, idx):
        if idx >= self._num_entries:
            raise IndexError()

        if self._prefetch <= 1:
            return self._get(idx)

        if idx == 0:
            for i in range(self._prefetch):
                self._prefetch_queue.put(self._executor.submit(self._get, (i)))

        if idx < (self._num_entries - self._prefetch):
            self._prefetch_queue.put(self._executor.submit(self._get, (idx + self._prefetch)))

        return self._prefetch_queue.get().result()

    def _get(self, idx):
        # calculate the offset & number of the samples to be read
        if not self._drop_last and idx == self._num_entries - 1:
            sample_offset = (
                idx * (self._batch_size * self._global_size)
                + self._last_batch_offset[self._global_rank]
            )
            batch = self._samples_in_last_batch
        else:
            sample_offset = idx * (self._batch_size * self._global_size) + (
                self._batch_size * self._global_rank
            )
            batch = self._batch_size
        row_lengths = self._row_lengths
        if batch != self._batch_size:
            row_lengths = []
            for i in range(len(self._hotness_per_table)):
                row_lengths.append(tf.repeat(self._hotness_per_table[i], repeats=batch))

        # read the data from binary file
        label_raw_data = os.pread(self._label_file, 4 * batch, 4 * sample_offset)
        label = np.frombuffer(label_raw_data, dtype=self._label_raw_type).reshape([batch, 1])

        dense_raw_data = os.pread(self._dense_file, 52 * batch, 52 * sample_offset)
        dense = np.frombuffer(dense_raw_data, dtype=self._dense_raw_type).reshape([batch, 13])
        category_raw_data = os.pread(
            self._category_file,
            self._hotness_per_sample * 4 * batch,
            self._hotness_per_sample * 4 * sample_offset,
        )
        category = np.frombuffer(category_raw_data, dtype=self._category_raw_type).reshape(
            [batch, self._hotness_per_sample]
        )
        indices = np.cumsum(self._hotness_per_table)[:-1]

        sub_arrays = np.split(category, indices, axis=1)

        if (
            self._label_raw_type == self._dense_raw_type
            and self._label_raw_type == self._category_raw_type
        ):
            data = np.concatenate([label, dense, category], axis=1)
            data = tf.convert_to_tensor(data)
            label = tf.cast(data[:, 0:1], dtype=tf.float32)
            dense = tf.cast(data[:, 1:14], dtype=tf.float32)
            category_np = data[:, 14:]
            flat_values = tf.reshape(category_np, [-1])
            category_ragged_tensors = []
            for i in range(len(sub_arrays)):
                flat_values = tf.reshape(sub_arrays[i], [-1])
                category_ragged_tensors.append(
                    tf.RaggedTensor.from_row_lengths(flat_values, row_lengths[i])
                )

        else:
            label = tf.convert_to_tensor(label, dtype=tf.float32)
            dense = tf.convert_to_tensor(dense, dtype=tf.float32)
            category = tf.convert_to_tensor(category, dtype=tf.int64)

            category_ragged_tensors = []
            for i in range(len(sub_arrays)):
                flat_values = tf.reshape(sub_arrays[i], [-1])
                category_ragged_tensors.append(
                    tf.RaggedTensor.from_row_lengths(flat_values, row_lengths[i])
                )

        # preprocess
        if self._log:
            dense = tf.math.log(dense + 3.0)

        return (dense, category_ragged_tensors), label
