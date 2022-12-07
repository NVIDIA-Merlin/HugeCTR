#!/bin/bash
set -e


task_num=${1:-1}

tf_version=`python -c "import tensorflow as tf;print(tf.__version__[0])"`

if [[ ${tf_version} -eq 1 ]];then
   cd tf1 
elif [[ ${tf_version} -eq 2 ]];then
   cd tf2 
else
   exit 1
fi

# -------- lookup -------------- #
cd lookup
horovodrun -np ${task_num} python lookup_sparse_distributed_test.py
horovodrun -np ${task_num} python lookup_sparse_distributed_dynamic_test.py
horovodrun -np ${task_num} python lookup_sparse_localized_test.py
horovodrun -np ${task_num} python lookup_sparse_localized_dynamic_test.py
horovodrun -np ${task_num} python all2all_dense_embedding_test.py
horovodrun -np ${task_num} python all2all_dense_embedding_dynamic_test.py
