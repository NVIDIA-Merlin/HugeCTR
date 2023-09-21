#!/usr/bin/env bash

mkdir -p /hps_torch_fuse_table_benchmark

cd /hps_torch_fuse_table_benchmark

cp -r /model_repo ./

cp -r /model_repo/8_table.json ./

cp -r /model_repo/embeddings ./

LD_PRELOAD=/usr/local/lib/python${PYTHON_VERSION}/dist-packages/merlin_hps-0.0.0-py${PYTHON_VERSION}-linux-x86_64.egg/hps_torch/lib/libhps_torch.so tritonserver --model-repository=model_repo --load-model=8_static_table_autofused --load-model=8_static_table_unfused --load-model=8_dynamic_table_autofused --load-model=8_dynamic_table_unfused --model-control-mode=explicit &

while [[ $(curl -v localhost:8000/v2/health/ready 2>&1 | grep "OK" | wc -l) -eq 0 ]]; do
        sleep 10;
done

echo "Successfully launching the Triton server for all models"

batch_size=(256 1024 4096 16384)

model_name=("8_static_table_unfused" "8_static_table_autofused" "8_dynamic_table_unfused" "8_dynamic_table_autofused")

for b in ${batch_size[*]};
do
  for m in ${model_name[*]};
  do
    echo $b $m
    perf_analyzer -m ${m} -u localhost:8000 --input-data /perf_data/${b}.json --shape input_1:8,${b},10
  done
done
