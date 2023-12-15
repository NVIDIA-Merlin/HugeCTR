#!/bin/bash
set -e

task_num=${1:-1}

tf_version=`python -c "import tensorflow as tf;print(tf.__version__[0])"`

if [[ ${tf_version} -eq 1 ]];then
   echo "Don't support tf1"
   exit 1
elif [[ ${tf_version} -eq 2 ]];then
   cd tf2 
else
   exit 1
fi

# -------- lookup -------------- #
cd dump_load
horovodrun -np ${task_num} python dump_load_distribute_static_big_table.py


