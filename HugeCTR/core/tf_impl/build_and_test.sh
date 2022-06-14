#!/bin/bash

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
echo $TF_CFLAGS
echo $TF_LFLAGS
g++ -std=c++17 -shared tf_impl_ops_test.cpp ../device.cpp -DGOOGLE_CUDA -DTF_IMPL_UT -I./../../../ -I./../ -I./../../include/ -I/usr/local/cuda-11.6/targets/x86_64-linux/include/ -o tf_impl_ops_test.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2

python tf_impl_ops_test.py
