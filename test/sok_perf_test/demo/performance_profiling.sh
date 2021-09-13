set -e
export PS4='\n\033[0;33m+[${BASH_SOURCE}:${LINENO}]: \033[0m'
set -x

# install specified TensorFlow and other tools
pip install tensorflow-gpu==2.5
pip install nvtx

# recompile sok and install it to system
current_dir=$(pwd)
cd ../../sparse_operation_kit/ && mkdir -p build && cd build && \
rm -rf * && cmake -DSM="70;80" -DUSE_NVTX=ON .. && make -j && make install 
cd $current_dir

# generate synthetic dataset
python3 gen_data.py \
    --global_batch_size=65536 \
    --slot_num=100 \
    --nnz_per_slot=10 \
    --iter_num=30 

# split whole dataset into multiple shards
python3 split_data.py \
    --filename="./data.file" \
    --split_num=8 \
    --save_prefix="data_"

# profiling tf on single GPU
nsys profile --sample=none --backtrace=none --cudabacktrace=none \
    --cpuctxsw=none -f true -o tf_1gpu_perf \
    python3 run_tf.py \
    --global_batch_size=8192 \
    --slot_num=100 \
    --nnz_per_slot=10 \
    --embedding_vec_size=16 \
    --num_dense_layers=10 \
    --vocabulary_size=8192 \
    --early_stop_iter=-1 \
    --filename="./data_0.file" \
    --data_splited=0 \
    --sparse_keys=0

# profiling sok on single GPU
nsys profile --sample=none --backtrace=none --cudabacktrace=none \
    --cpuctxsw=none -f true -o sok_1gpu_perf \
    python3 run_sok.py \
    --global_batch_size=8192 \
    --slot_num=100 \
    --nnz_per_slot=10 \
    --embedding_vec_size=16 \
    --num_dense_layers=10 \
    --vocabulary_size=8192 \
    --early_stop_iter=-1 \
    --filename="./data_0.file" \
    --data_splited=0 \
    --sparse_keys=0 \
    --whether_single_gpu=1

# profiling sok on 2 GPU
nsys profile --sample=none --backtrace=none --cudabacktrace=none \
    --cpuctxsw=none -f true -o sok_2gpu_perf --trace-fork-before-exec=true \
    mpiexec --allow-run-as-root -np 2 \
    python3 run_sok.py \
    --global_batch_size=16384 \
    --slot_num=100 \
    --nnz_per_slot=10 \
    --embedding_vec_size=16 \
    --num_dense_layers=10 \
    --vocabulary_size=8192 \
    --early_stop_iter=-1 \
    --filename="./data_" \
    --data_splited=1 \
    --sparse_keys=0 \
    --whether_single_gpu=0

# profiling sok on 4 GPU
nsys profile --sample=none --backtrace=none --cudabacktrace=none \
    --cpuctxsw=none -f true -o sok_4gpu_perf --trace-fork-before-exec=true \
    mpiexec --allow-run-as-root -np 4 \
    python3 run_sok.py \
    --global_batch_size=32768 \
    --slot_num=100 \
    --nnz_per_slot=10 \
    --embedding_vec_size=16 \
    --num_dense_layers=10 \
    --vocabulary_size=8192 \
    --early_stop_iter=-1 \
    --filename="./data_" \
    --data_splited=1 \
    --sparse_keys=0 \
    --whether_single_gpu=0

# profiling sok on 8 GPU
nsys profile --sample=none --backtrace=none --cudabacktrace=none \
    --cpuctxsw=none -f true -o sok_8gpu_perf --trace-fork-before-exec=true \
    mpiexec --allow-run-as-root -np 8 \
    python3 run_sok.py \
    --global_batch_size=65536 \
    --slot_num=100 \
    --nnz_per_slot=10 \
    --embedding_vec_size=16 \
    --num_dense_layers=10 \
    --vocabulary_size=8192 \
    --early_stop_iter=-1 \
    --filename="./data_" \
    --data_splited=1 \
    --sparse_keys=0 \
    --whether_single_gpu=0

# clean intermediate files
rm *.file
echo "[INFO]: Cleaned intermediate files."