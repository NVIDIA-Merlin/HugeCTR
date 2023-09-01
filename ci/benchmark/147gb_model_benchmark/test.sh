#!/usr/bin/env bash

mkdir -p /147gb_model_benchmark/model_repo

cd /147gb_model_benchmark

cp /model_repo/light.json ./

cp /model_repo/dynamic_build.py ./

cp /model_repo/*.onnx ./

cp -r /model_repo/dynamic*trt ./model_repo

python3 dynamic_build.py

mv dynamic_1fc_lite.trt model_repo/dynamic_1fc_lite_hps_trt/1

mv dynamic_3fc_lite.trt model_repo/dynamic_3fc_lite_hps_trt/1

mv dynamic_dlrm.trt model_repo/dynamic_dlrm_hps_trt/1

LD_PRELOAD=/usr/local/hps_trt/lib/libhps_plugin.so tritonserver --model-repository=model_repo --load-model=dynamic_1fc_lite_hps_trt --load-model=dynamic_3fc_lite_hps_trt --load-model=dynamic_dlrm_hps_trt --model-control-mode=explicit &

while [[ $(curl -v localhost:8000/v2/health/ready 2>&1 | grep "OK" | wc -l) -eq 0 ]]; do
        sleep 10;
done

echo "Successfully launching the Triton server for all models"

batch_size=(256 1024 4096 16384)

model_name=("dynamic_1fc_lite_hps_trt" "dynamic_3fc_lite_hps_trt" "dynamic_dlrm_hps_trt")

for b in ${batch_size[*]};
do
  for m in ${model_name[*]};
  do
    echo $b $m
    perf_analyzer -m ${m} -u localhost:8000 --input-data /model_repo/perf_data/${b}.json --shape categorical_features:${b},26 --shape numerical_features:${b},13
  done
done
