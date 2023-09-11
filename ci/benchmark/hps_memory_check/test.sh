#!/usr/bin/env bash

python3 ${WORKDIR}/ci/common/generate_inference_config.py --config_template ${WORKDIR}/ci/common/config_pbtxt_template.txt --ps_template ${WORKDIR}/ci/common/ps_template.json --batchsize 64 --mixed_precision false --ec_type dynamic --config_output /model/dlrm/config.pbtxt --ps_output /model/ps.json

tritonserver --model-repository=/model/ --load-model=dlrm --log-verbose=1 --model-control-mode=explicit --backend-directory=/usr/local/hugectr/backends --backend-config=hps,ps=/model/ps.json > /dev/null 2> /dev/null &
echo > /logs/cpu_dynamic_mem.log
while [[ $(curl -v localhost:8000/v2/health/ready 2>&1 | grep "OK" | wc -l) -eq 0 ]]; do
    (top -d 1 -n 10 -b | grep triton) >> /logs/cpu_dynamic_mem.log
done
kill -s 9 `pgrep tritonserver`
sleep 10;

python3 ${WORKDIR}/ci/common/generate_inference_config.py --config_template ${WORKDIR}/ci/common/config_pbtxt_template.txt --ps_template ${WORKDIR}/ci/common/ps_template.json --batchsize 64 --mixed_precision false --ec_type uvm --config_output /model/dlrm/config.pbtxt --ps_output /model/ps.json
tritonserver --model-repository=/model/ --load-model=dlrm --log-verbose=1 --model-control-mode=explicit --backend-directory=/usr/local/hugectr/backends --backend-config=hps,ps=/model/ps.json > /dev/null 2> /dev/null &
echo > /logs/cpu_uvm_mem.log
while [[ $(curl -v localhost:8000/v2/health/ready 2>&1 | grep "OK" | wc -l) -eq 0 ]]; do
        (top -d 1 -n 10 -b | grep triton) >> /logs/cpu_uvm_mem.log
done
kill -s 9 `pgrep tritonserver`
sleep 10;

python3 ${WORKDIR}/ci/common/generate_inference_config.py --config_template ${WORKDIR}/ci/common/config_pbtxt_template.txt --ps_template ${WORKDIR}/ci/common/ps_template.json --batchsize 64 --mixed_precision false --ec_type static --config_output /model/dlrm/config.pbtxt --ps_output /model/ps.json
tritonserver --model-repository=/model/ --load-model=dlrm --log-verbose=1 --model-control-mode=explicit --backend-directory=/usr/local/hugectr/backends --backend-config=hps,ps=/model/ps.json > /dev/null 2> /dev/null &
echo > /logs/cpu_static_mem.log
while [[ $(curl -v localhost:8000/v2/health/ready 2>&1 | grep "OK" | wc -l) -eq 0 ]]; do
        (top -d 1 -n 10 -b | grep triton) >> /logs/cpu_static_mem.log  
done
kill -s 9 `pgrep tritonserver`




