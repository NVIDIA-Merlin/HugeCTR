set -e

# ---------- unit test------------- #
python3 test_all_gather_dispatcher.py
python3 test_csr_conversion_distributed.py
python3 test_reduce_scatter_dispatcher.py

# ------ demo model test ---------- #
python3 test_demo_model_single_worker.py \
        --gpu_num=8 --iter_num=100 \
        --max_vocabulary_size_per_gpu=1024 \
        --slot_num=10 --max_nnz=4 \
        --embedding_vec_size=4 \
        --combiner='mean' --global_batch_size=65536 \
        --optimizer='plugin_adam' \
        --generate_new_datas=1

# ----------- multi worker test with ips set mannually ------ #
# python3 test_demo_model_multi_worker.py \
#         --local_gpu_num=8 --iter_num=100 \
#         --max_vocabulary_size_per_gpu=1024 \
#         --slot_num=10 --max_nnz=4 \
#         --embedding_vec_size=4 \
#         --combiner='mean' --global_batch_size=65536 \
#         --optimizer='plugin_adam' \
#         --generate_new_datas=1 \
#         --ips "10.33.12.35" "10.33.12.42"


# ------ multi worker test within single worker but using different GPUs.
python3 test_demo_model_multi_worker.py \
        --local_gpu_num=8 --iter_num=100 \
        --max_vocabulary_size_per_gpu=1024 \
        --slot_num=10 --max_nnz=4 \
        --embedding_vec_size=4 \
        --combiner='mean' --global_batch_size=65536 \
        --optimizer='plugin_adam' \
        --generate_new_datas=1 \
        --ips "localhost" "localhost"