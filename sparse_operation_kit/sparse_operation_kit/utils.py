import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from sparse_operation_kit import tf_version


def SOK_IndexedSlices():
    if tf_version[0] == 2 and tf_version[1] >= 13:
        return tf.IndexedSlices
    return ops.IndexedSlices


def get_nano_time(datetime_str):
    # datetime_str should be %Y-%m-%d %H:%M:%S
    nanotime_obj = np.datetime64(datetime_str, "ns")
    return nanotime_obj.astype("uint64")
