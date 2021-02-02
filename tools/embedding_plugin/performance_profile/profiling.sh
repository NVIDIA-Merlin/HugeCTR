set -e

fprop_version='v3'
embedding_type='localized'

python3 embedding_plugin_profile.py --gpus 0 --which='plugin' --batch_size=65536 --prepare_datas=1 \
          --embedding_type=$embedding_type --fprop_version=$fprop_version

nsys profile --trace=cublas,cuda,cudnn,nvtx,openmp --sample=none --backtrace=dwarf --cudabacktrace=all -f true \
     -o plugin_1gpu_$fprop_version"_"$embedding_type python3 embedding_plugin_profile.py --which=plugin \
     --gpus 0 --num_layers=7 --batch_size=65536 \
     --embedding_type=$embedding_type --fprop_version=$fprop_version
nsys profile --trace=cublas,cuda,cudnn,nvtx,openmp --sample=none --backtrace=dwarf --cudabacktrace=all -f true \
     -o plugin_2gpu_$fprop_version"_"$embedding_type python3 embedding_plugin_profile.py --which=plugin \
     --gpus 0 1 --num_layers=7 --batch_size=65536 \
     --embedding_type=$embedding_type --fprop_version=$fprop_version
nsys profile --trace=cublas,cuda,cudnn,nvtx,openmp --sample=none --backtrace=dwarf --cudabacktrace=all -f true \
     -o plugin_4gpu_$fprop_version"_"$embedding_type python3 embedding_plugin_profile.py --which=plugin \
     --gpus 0 1 2 3 --num_layers=7 --batch_size=65536 \
     --embedding_type=$embedding_type --fprop_version=$fprop_version
nsys profile --trace=cublas,cuda,cudnn,nvtx,openmp --sample=none --backtrace=dwarf --cudabacktrace=all -f true \
     -o plugin_8gpu_$fprop_version"_"$embedding_type python3 embedding_plugin_profile.py --which=plugin \
     --gpus 0 1 2 3 4 5 6 7 --num_layers=7 --batch_size=65536 \
     --embedding_type=$embedding_type --fprop_version=$fprop_version
# nsys profile --trace=cublas,cuda,cudnn,nvtx,openmp --sample=none --backtrace=dwarf --cudabacktrace=all -f true \
#      -o plugin_16gpu_$fprop_version"_"$embedding_type python3 embedding_plugin_profile.py --which=plugin \
#      --gpus 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
#      --num_layers=7 --batch_size=65536 --embedding_type=$embedding_type --fprop_version=$fprop_version

# nsys profile --trace=cublas,cuda,cudnn,nvtx,openmp --sample=none --backtrace=dwarf --cudabacktrace=all -f true \
     # -o origin_1gpu python3 embedding_plugin_profile.py --which=origin --gpus 0 --num_layers=7 --batch_size=65536
# nsys profile --trace=cublas,cuda,cudnn,nvtx,openmp --sample=none --backtrace=dwarf --cudabacktrace=all -f true \
     # -o origin_2gpu python3 embedding_plugin_profile.py --which=origin --gpus 0 1 --num_layers=7 --batch_size=65536
# nsys profile --trace=cublas,cuda,cudnn,nvtx,openmp --sample=none --backtrace=dwarf --cudabacktrace=all -f true \
     # -o origin_4gpu python3 embedding_plugin_profile.py --which=origin --gpus 0 1 2 3 --num_layers=7 --batch_size=65536
# nsys profile --trace=cublas,cuda,cudnn,nvtx,openmp --sample=none --backtrace=dwarf --cudabacktrace=all -f true \
     # -o origin_8gpu python3 embedding_plugin_profile.py --which=origin --gpus 0 1 2 3 4 5 6 7 --num_layers=7 --batch_size=65536
# nsys profile --trace=cublas,cuda,cudnn,nvtx,openmp --sample=none --backtrace=dwarf --cudabacktrace=all -f true \
#      -o origin_16gpu python3 embedding_plugin_profile.py --which=origin \
#      --gpus 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
#      --num_layers=7 --batch_size=65536
