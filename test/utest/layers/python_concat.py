from __future__ import print_function
import numpy as np
import argparse
import sys
import numpy as np
import os

parser = argparse.ArgumentParser(description="np array concat")
parser.add_argument("num_inputs", help="Number of inputs", type=int)
parser.add_argument("array_ndim", help="Dimension of the input arrays", type=int)
parser.add_argument("axis", help="Bytes per float in the file", type=int)
parser.add_argument("input_file", help="Input filename")
args = parser.parse_args()


def iter_arrays(fname, array_ndim=3, dim_dtype=np.int32, array_dtype=np.float32):
    with open(fname, "rb") as f:
        fsize = os.fstat(f.fileno()).st_size

        # while we haven't yet reached the end of the file...
        while f.tell() < fsize:
            # get the dimensions for this array
            dims = np.fromfile(f, dim_dtype, array_ndim)

            # get the array contents
            yield np.fromfile(f, array_dtype, np.prod(dims)).reshape(dims)


input_arrays = iter_arrays(args.input_file, args.array_ndim)

output = np.concatenate([array for array in input_arrays], axis=args.axis)
print(output.flatten().tolist())
