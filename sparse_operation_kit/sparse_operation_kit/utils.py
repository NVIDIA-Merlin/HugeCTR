import tensorflow as tf
from tensorflow.python.framework import ops
from sparse_operation_kit import tf_version


def SOK_IndexedSlices():
    if tf_version[0] == 2 and tf_version[1] >= 13:
        return tf.IndexedSlices
    return ops.IndexedSlices
