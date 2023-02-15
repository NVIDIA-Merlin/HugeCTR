#!/usr/bin/env bash

mkdir -p /hps_plugin_benchmark

cd /hps_plugin_benchmark

python3 /workdir/docs/source/hierarchical_parameter_server/hps_dlrm_benchmark_scripts/create_tf_models.py

python3 /workdir/docs/source/hierarchical_parameter_server/hps_dlrm_benchmark_scripts/create_trt_engines.py

cp -r /model_repo ./

mv dlrm_tf_saved_model model_repo/native_tf/1/model.savedmodel

mv hps_plugin_dlrm_tf_saved_model model_repo/tf_with_hps/1/model.savedmodel

mv fp32_hps_plugin_dlrm.trt model_repo/fp32_trt_with_hps/1

mv fp16_hps_plugin_dlrm.trt model_repo/fp16_trt_with_hps/1

LD_PRELOAD=/usr/local/hps_trt/lib/libhps_plugin.so:/usr/local/lib/python3.8/dist-packages/merlin_hps-1.0.0-py3.8-linux-x86_64.egg/hierarchical_parameter_server/lib/libhierarchical_parameter_server.so tritonserver --model-repository=model_repo --load-model=native_tf --load-model=tf_with_hps --load-model=fp32_trt_with_hps --load-model=fp16_trt_with_hps --model-control-mode=explicit &

while [[ $(curl -v localhost:8000/v2/health/ready 2>&1 | grep "OK" | wc -l) -eq 0 ]]; do
        sleep 10;
done

echo "Successfully launching the Triton server for all models"

batch_size=(32 1024 16384)

model_name=("native_tf" "tf_with_hps" "fp32_trt_with_hps" "fp16_trt_with_hps")

for b in ${batch_size[*]};
do
  for m in ${model_name[*]};
  do
    echo $b $m
    perf_analyzer -m ${m} -u localhost:8000 --input-data /perf_data/${b}.json --shape categorical_features:${b},26 --shape numerical_features:${b},13
  done
done