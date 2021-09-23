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
    --optimizer="adam" \
    --generate_new_datas=0 \
    --save_params=1 \
    --use_hashtable=1


# --------- tf.distribute.Strategy -------- #



# --------- horovod -------------- #