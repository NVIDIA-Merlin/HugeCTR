#!/bin/bash
set -e


tf_version=`python -c "import tensorflow as tf;print(tf.__version__[0])"`

if [[ ${tf_version} -eq 1 ]];then
   cd tf1 
elif [[ ${tf_version} -eq 2 ]];then
   cd tf2 
else
   exit 1
fi

# -------- variable -------------- #
cd variable
python dynamic_variable_test.py
python sok_sgd_test.py
python assign_and_export_test.py
horovodrun -np 1 python filter_variables_test.py
