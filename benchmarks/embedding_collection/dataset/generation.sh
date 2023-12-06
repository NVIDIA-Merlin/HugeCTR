#!/usr/bin/env bash

output_dir="/lustre/fsw/coreai_devtech_all/aleliu/hugectr/benchmarks/embedding_collection/dataset/"
dataset_name="7table_470B_hotness20"
# dataset_name="180table_70B_hotness80"
# dataset_name="200table_100B_hotness20"
# dataset_name="510table_110B_hotness5"
result_name=${output_dir}/${dataset_name}_synthetic_alpha1.1.bin
for ((i = 0; i < 100; i++)); do
  python3 ${dataset_name}.py $i ${output_dir}
done

#for ((i = 10; i < 100; i++)); do
#  cp ${result_name}.part$((i % 10)) ${result_name}.part$i
#done
cat ${result_name}.part* > ${result_name}
rm ${result_name}.part*
