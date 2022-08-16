set -e

# TODO: check tf version
cd tf2

# -------- variable -------------- #
cd variable
python dynamic_variable_test.py
python sok_sgd_test.py
python assign_and_export_test.py
python filter_variables_test.py
cd ..

# -------- lookup -------------- #
cd lookup
num_gpu=`nvidia-smi  -L | wc -l`
for ((i=1; i<=${num_gpu}; i++))
do
    horovodrun -np ${i} python lookup_sparse_distributed_test.py
    horovodrun -np ${i} python lookup_sparse_distributed_dynamic_test.py
    horovodrun -np ${i} python lookup_sparse_localized_test.py
    horovodrun -np ${i} python lookup_sparse_localized_dynamic_test.py
    horovodrun -np ${i} python all2all_dense_embedding_test.py
    horovodrun -np ${i} python all2all_dense_embedding_dynamic_test.py
done
