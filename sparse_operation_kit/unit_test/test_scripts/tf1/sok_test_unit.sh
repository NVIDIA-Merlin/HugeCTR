set -e

# ---------- operation unit test------------- #
# dense embedding + adam + save_param + hashtable
python3 test_dense_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --nnz_per_slot=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1

# sparse embedding + adam + save_params + no-hashtable
python3 test_sparse_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --max_nnz=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=0

# dense embedding + adam + save_params + hashtable
pip install mpi4py

mpiexec --allow-run-as-root -np 8 --oversubscribe \
    python3 test_dense_emb_demo.py \
    --distributed_tool="horovod" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --nnz_per_slot=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1

# sparse_embedding + adam + save_params + hashtable
mpiexec --allow-run-as-root -np 8 --oversubscribe \
    python3 test_sparse_emb_demo.py \
    --distributed_tool="horovod" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --max_nnz=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1

echo "Test merlin-sok passed!"
# ----- clean intermediate files ------ #
rm *.file && rm -rf embedding_variables/
