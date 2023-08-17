#!/usr/bin/env bash

mkdir -p /3fc_147gb_model_benchmark

cd /3fc_147gb_model_benchmark

cp /model_repo/light.json ./

LD_PRELOAD=/usr/local/hps_trt/lib/libhps_plugin.so tritonserver --model-repository=/model_repo --load-model=dynamic_3fc_lite_hps_trt --model-control-mode=explicit &

while [[ $(curl -v localhost:8000/v2/health/ready 2>&1 | grep "OK" | wc -l) -eq 0 ]]; do
        sleep 10;
done

echo "Successfully launching the Triton server for all models"

batch_size=(256 1024 4096 16384)

model_name=("dynamic_3fc_lite_hps_trt")

for b in ${batch_size[*]};
do
  for m in ${model_name[*]};
  do
    echo $b $m
    perf_analyzer -m ${m} -u localhost:8000 --input-data /model_repo/perf_data/${b}.json --shape categorical_features:${b},26 --shape numerical_features:${b},13
  done
done
