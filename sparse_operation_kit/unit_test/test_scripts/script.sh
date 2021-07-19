set -e
export PS4='\n\033[0;33m+[${BASH_SOURCE}:${LINENO}]: \033[0m'
set -x

# ---------- operation unit test------------- #
python3 test_all_gather_dispatcher.py
python3 test_csr_conversion_distributed.py
python3 test_reduce_scatter_dispatcher.py

# ---------------------------- Sparse Embedding Layers testing ------------------- #
# ---------- single node save testing ------- #
python3 test_sparse_emb_demo_model_single_worker.py \
        --gpu_num=8 --iter_num=100 \
        --max_vocabulary_size_per_gpu=1024 \
        --slot_num=10 --max_nnz=4 \
        --embedding_vec_size=4 \
        --combiner='mean' --global_batch_size=65536 \
        --optimizer='plugin_adam' \
        --save_params=1 \
        --generate_new_datas=1

# ------------ single node restore testing ------- #
python3 test_sparse_emb_demo_model_single_worker.py \
        --gpu_num=8 --iter_num=100 \
        --max_vocabulary_size_per_gpu=1024 \
        --slot_num=10 --max_nnz=4 \
        --embedding_vec_size=4 \
        --combiner='mean' --global_batch_size=65536 \
        --optimizer='adam' \
        --restore_params=1 \
        --generate_new_datas=1

# ----------- multi worker test with ips set mannually, save testing ------ #
# python3 test_sparse_emb_demo_model_multi_worker.py \
#         --local_gpu_num=8 --iter_num=100 \
#         --max_vocabulary_size_per_gpu=1024 \
#         --slot_num=10 --max_nnz=4 \
#         --embedding_vec_size=4 \
#         --combiner='mean' --global_batch_size=65536 \
#         --optimizer='plugin_adam' \
#         --save_params=1 \
#         --generate_new_datas=1 \
#         --ips "10.33.12.11" "10.33.12.29"

# # ----------- multi worker test with ips set mannually, restore testing ------ #
# python3 test_sparse_emb_demo_model_multi_worker.py \
#         --local_gpu_num=8 --iter_num=100 \
#         --max_vocabulary_size_per_gpu=1024 \
#         --slot_num=10 --max_nnz=4 \
#         --embedding_vec_size=4 \
#         --combiner='mean' --global_batch_size=65536 \
#         --optimizer='plugin_adam' \
#         --restore_params=1 \
#         --generate_new_datas=1 \
#         --ips "10.33.12.11" "10.33.12.29"

# ------ multi worker test within single worker but using different GPUs. save
python3 test_sparse_emb_demo_model_multi_worker.py \
        --local_gpu_num=8 --iter_num=100 \
        --max_vocabulary_size_per_gpu=1024 \
        --slot_num=10 --max_nnz=4 \
        --embedding_vec_size=4 \
        --combiner='mean' --global_batch_size=65536 \
        --optimizer='plugin_adam' \
        --generate_new_datas=1 \
        --save_params=1 \
        --ips "localhost" "localhost"

# ------ multi worker test within single worker but using different GPUs. restore
python3 test_sparse_emb_demo_model_multi_worker.py \
        --local_gpu_num=8 --iter_num=100 \
        --max_vocabulary_size_per_gpu=1024 \
        --slot_num=10 --max_nnz=4 \
        --embedding_vec_size=4 \
        --combiner='mean' --global_batch_size=65536 \
        --optimizer='adam' \
        --generate_new_datas=1 \
        --restore_params=1 \
        --ips "localhost" "localhost"


# ---------------------------- Dense Embedding Layers testing ------------------- #
# ---------- single node save testing ------- #
python3 test_dense_emb_demo_model_single_worker.py \
        --gpu_num=8 --iter_num=100 \
        --max_vocabulary_size_per_gpu=1024 \
        --slot_num=10 --nnz_per_slot=4 \
        --embedding_vec_size=4 \
        --global_batch_size=65536 \
        --optimizer='plugin_adam' \
        --save_params=1 \
        --generate_new_datas=1

# ---------- single node restore testing ------- #
python3 test_dense_emb_demo_model_single_worker.py \
        --gpu_num=8 --iter_num=100 \
        --max_vocabulary_size_per_gpu=1024 \
        --slot_num=10 --nnz_per_slot=4 \
        --embedding_vec_size=4 \
        --global_batch_size=65536 \
        --optimizer='adam' \
        --restore_params=1 \
        --generate_new_datas=1

# ----------- multi worker test with ips set mannually, save testing ------ #
# python3 test_dense_emb_demo_model_multi_worker.py \
#         --local_gpu_num=8 --iter_num=100 \
#         --max_vocabulary_size_per_gpu=1024 \
#         --slot_num=10 --nnz_per_slot=4 \
#         --embedding_vec_size=4 \
#         --global_batch_size=65536 \
#         --optimizer='plugin_adam' \
#         --save_params=1 \
#         --generate_new_datas=1 \
#         --ips "10.33.12.22" "10.33.12.16"

# ----------- multi worker test with ips set mannually, restore testing ------ #
# python3 test_dense_emb_demo_model_multi_worker.py \
#         --local_gpu_num=8 --iter_num=100 \
#         --max_vocabulary_size_per_gpu=1024 \
#         --slot_num=10 --nnz_per_slot=4 \
#         --embedding_vec_size=4 \
#         --global_batch_size=65536 \
#         --optimizer='adam' \
#         --restore_params=1 \
#         --generate_new_datas=1 \
#         --ips "10.33.12.22" "10.33.12.16"

# ------ multi worker test within single worker but using different GPUs. save
python3 test_dense_emb_demo_model_multi_worker.py \
        --local_gpu_num=8 --iter_num=100 \
        --max_vocabulary_size_per_gpu=1024 \
        --slot_num=10 --nnz_per_slot=4 \
        --embedding_vec_size=4 \
        --global_batch_size=65536 \
        --optimizer='plugin_adam' \
        --save_params=1 \
        --generate_new_datas=1 \
        --ips "localhost" "localhost"

# ------ multi worker test within single worker but using different GPUs. restore
python3 test_dense_emb_demo_model_multi_worker.py \
        --local_gpu_num=8 --iter_num=100 \
        --max_vocabulary_size_per_gpu=1024 \
        --slot_num=10 --nnz_per_slot=4 \
        --embedding_vec_size=4 \
        --global_batch_size=65536 \
        --optimizer='adam' \
        --restore_params=1 \
        --generate_new_datas=1 \
        --ips "localhost" "localhost"
