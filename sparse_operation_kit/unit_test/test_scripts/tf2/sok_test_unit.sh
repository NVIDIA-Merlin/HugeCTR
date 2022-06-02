set -e

# ---------- operation unit test------------- #
python3 test_sparse_emb_demo_model_single_worker.py \
        --gpu_num=1 --iter_num=100 \
        --max_vocabulary_size_per_gpu=1024 \
        --slot_num=10 --max_nnz=4 \
        --embedding_vec_size=4 \
        --combiner='mean' --global_batch_size=65536 \
        --optimizer='adam' \
        --save_params=1 \
        --generate_new_datas=1 \
        --use_hashtable=0 \
        --use_tf_initializer=1

python3 test_dense_emb_demo_model_single_worker.py \
        --gpu_num=1 --iter_num=100 \
        --max_vocabulary_size_per_gpu=1024 \
        --slot_num=10 --nnz_per_slot=4 \
        --embedding_vec_size=4 \
        --global_batch_size=65536 \
        --optimizer='adam' \
        --save_params=1 \
        --generate_new_datas=1 \
        --use_hashtable=0 \
        --use_tf_initializer=1

python3 prepare_dataset.py \
        --global_batch_size=65536 \
        --slot_num=10 \
        --nnz_per_slot=5 \
        --iter_num=30 \
        --vocabulary_size=1024 \
        --filename="datas.file" \
        --split_num=1 \
        --save_prefix="data_"

pip install mpi4py

mpiexec -np 1 --allow-run-as-root \
        --oversubscribe \
        python3 test_multi_dense_emb_demo_model_mpi.py \
        --file_prefix="./data_" \
        --global_batch_size=65536 \
        --max_vocabulary_size_per_gpu=8192 \
        --slot_num_list 3 3 4 \
        --nnz_per_slot=5 \
        --num_dense_layers=4 \
        --embedding_vec_size_list 2 4 8 \
        --dataset_iter_num=30 \
        --optimizer="adam"

horovodrun --mpi-args="--oversubscribe" -np 1 -H localhost:8 \
        python3 test_multi_dense_emb_demo_model_hvd.py \
        --file_prefix="./data_" \
        --global_batch_size=65536 \
        --max_vocabulary_size_per_gpu=8192 \
        --slot_num_list 3 3 4 \
        --nnz_per_slot=5 \
        --num_dense_layers=4 \
        --embedding_vec_size_list 2 4 8 \
        --dataset_iter_num=30 \
        --optimizer="adam"

echo "Test merlin-sok passed!"
# ----- clean intermediate files ------ #
rm *.file && rm -rf embedding_variables/
