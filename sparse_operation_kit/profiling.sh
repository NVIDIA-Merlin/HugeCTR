set -e

# nsys profile --trace=cublas,cuda,cudnn,nvtx --sample=cpu --backtrace=dwarf --cudabacktrace=all -f true \
#     -o profiling_65536_100_10_no_print_32threads \
#     python test.py


nsys profile --trace=cublas,cuda,cudnn,nvtx --sample=cpu --backtrace=dwarf --cudabacktrace=all -f true \
    -o no_input_no_print \
    python test.py      

TF_GPU_THREAD_MODE=gpu_private nsys profile --trace=cublas,cuda,cudnn,nvtx --sample=cpu --backtrace=dwarf --cudabacktrace=all -f true \
    -o no_input_no_print_gpu_private \
    python test.py      

TF_GPU_THREAD_MODE=gpu_private TF_GPU_THREAD_COUNT=8 nsys profile --trace=cublas,cuda,cudnn,nvtx --sample=cpu --backtrace=dwarf --cudabacktrace=all -f true \
    -o no_input_no_print_gpu_private_8threads \
    python test.py     

TF_GPU_THREAD_MODE=gpu_private TF_GPU_THREAD_COUNT=16 nsys profile --trace=cublas,cuda,cudnn,nvtx --sample=cpu --backtrace=dwarf --cudabacktrace=all -f true \
    -o no_input_no_print_gpu_private_16threads \
    python test.py  

TF_GPU_THREAD_MODE=gpu_private TF_GPU_THREAD_COUNT=32 nsys profile --trace=cublas,cuda,cudnn,nvtx --sample=cpu --backtrace=dwarf --cudabacktrace=all -f true \
    -o no_input_no_print_gpu_private_32threads \
    python test.py

TF_GPU_THREAD_MODE=gpu_shared nsys profile --trace=cublas,cuda,cudnn,nvtx --sample=cpu --backtrace=dwarf --cudabacktrace=all -f true \
    -o no_input_no_print_gpu_shared \
    python test.py

TF_GPU_THREAD_MODE=gpu_shared TF_GPU_THREAD_COUNT=8 nsys profile --trace=cublas,cuda,cudnn,nvtx --sample=cpu --backtrace=dwarf --cudabacktrace=all -f true \
    -o no_input_no_print_gpu_shared_8threads \
    python test.py

TF_GPU_THREAD_MODE=gpu_shared TF_GPU_THREAD_COUNT=16 nsys profile --trace=cublas,cuda,cudnn,nvtx --sample=cpu --backtrace=dwarf --cudabacktrace=all -f true \
    -o no_input_no_print_gpu_shared_16threads \
    python test.py

TF_GPU_THREAD_MODE=gpu_shared TF_GPU_THREAD_COUNT=32 nsys profile --trace=cublas,cuda,cudnn,nvtx --sample=cpu --backtrace=dwarf --cudabacktrace=all -f true \
    -o no_input_no_print_gpu_shared_32threads \
    python test.py