import os
import argparse
import time
import json
import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--slot_size_array", type=str)
    parser.add_argument("--dense_type", type=str, default="int32")
    parser.add_argument("--label_type", type=str, default="int32")
    parser.add_argument("--category_type", type=str, default="int32")
    parser.add_argument("--dense_log", type=str, default="True")
    args = parser.parse_args()

    args.slot_size_array = eval(args.slot_size_array)
    assert isinstance(args.slot_size_array, list)

    if args.dense_log == "False":
        args.dense_log = False
    else:
        args.dense_log = True

    MULTI_HOT_SIZES = [
        3,
        2,
        1,
        2,
        6,
        1,
        1,
        1,
        1,
        7,
        3,
        8,
        1,
        6,
        9,
        5,
        1,
        1,
        1,
        12,
        100,
        27,
        10,
        3,
        1,
        1,
    ]

    sparse_hotness_per_sample = np.sum(MULTI_HOT_SIZES)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    size = os.path.getsize(args.input)
    NKEYS_PER_SAMPLE = sum(MULTI_HOT_SIZES)
    NUM_TABLE = 26
    NUM_DENSE = 13
    NUM_LABEL = 1
    BYTES_PER_SAMPLE = (
        NKEYS_PER_SAMPLE + NUM_DENSE + NUM_LABEL
    ) * 4  # 4 bytes per data , float32 , uint32

    print("BYTES_PER_SAMPLE = ", BYTES_PER_SAMPLE)
    assert size % BYTES_PER_SAMPLE == 0
    num_samples = size // BYTES_PER_SAMPLE

    chunk_size = 1024 * 1024

    inp_f = open(args.input, "rb")

    label_f = open(os.path.join(args.output, "label.bin"), "wb")
    dense_f = open(os.path.join(args.output, "dense.bin"), "wb")
    category_f = open(os.path.join(args.output, "category.bin"), "wb")

    num_loops = num_samples // chunk_size + 1

    start_time = time.time()
    for i in range(num_loops):
        t = time.time()

        if i == (num_loops - 1):
            batch = min(chunk_size, num_samples % chunk_size)
            if batch == 0:
                break
        else:
            batch = chunk_size

        raw_buffer = inp_f.read(BYTES_PER_SAMPLE * batch)

        for j in range(batch):
            label_buffer = raw_buffer[j * BYTES_PER_SAMPLE : j * BYTES_PER_SAMPLE + 4]
            dense_buffer = raw_buffer[j * BYTES_PER_SAMPLE + 4 : j * BYTES_PER_SAMPLE + 56]
            category_buffer = raw_buffer[
                j * BYTES_PER_SAMPLE + 56 : j * BYTES_PER_SAMPLE + BYTES_PER_SAMPLE
            ]

            label_f.write(label_buffer)
            dense_f.write(dense_buffer)
            category_f.write(category_buffer)

        print(
            "%d/%d batch finished. write %d samples, time: %.2fms, remaining time: %.2f min"
            % (
                i + 1,
                num_loops,
                batch,
                (time.time() - t) * 1000,
                ((time.time() - start_time) / 60) * (num_loops / (i + 1) - 1),
            )
        )

    inp_f.close()
    label_f.close()
    dense_f.close()
    category_f.close()

    metadata = {
        "vocab_sizes": args.slot_size_array,
        "label_raw_type": args.label_type,
        "dense_raw_type": args.dense_type,
        "category_raw_type": args.category_type,
        "dense_log": args.dense_log,
        "hotness_per_table": MULTI_HOT_SIZES,
    }
    with open(os.path.join(args.output, "metadata.json"), "w") as f:
        json.dump(metadata, f)
