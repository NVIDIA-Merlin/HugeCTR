set -e 
export PS4='\n\033[0;33m+[${BASH_SOURCE}:${LINENO}]: \033[0m'
set -x

# ----- single GPU -------- #
python3 test_dense_emb_demo.py \
    --gpu_num=1 \
    --distributed_tool="onedevice" \
    --iter_num=30 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=30 \
    --nnz_per_slot=10 \
    --embedding_vec_size=4 \
    --global_batch_size=16384 \
    --optimizer="plugin_adam" \
    --generate_new_datas=1 \
    --save_params=1 \
    --use_hashtable=1

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
    --save_params=0 \
    --restore_params=1 \
    --use_hashtable=1

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
    --save_params=0 \
    --restore_params=0 \
    --use_hashtable=0

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
    --save_params=0 \
    --restore_params=0 \
    --use_hashtable=0 \
    --dynamic_input=1

# --------- horovod -------------- #
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


# ----- clean intermediate files ------ #
rm *.file && rm -rf embedding_variables/