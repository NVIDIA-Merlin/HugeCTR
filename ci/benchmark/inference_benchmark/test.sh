#!/usr/bin/env bash

python3 /workdir/ci/common/generate_inference_config.py --config_template /workdir/ci/common/config_pbtxt_template.txt --ps_template /workdir/ci/common/ps_template.json --batchsize ${BZ} --mixed_precision ${MIXED_PRECISION} --config_output /model/dlrm/config.pbtxt --ps_output /model/ps.json

tritonserver --model-repository=/model/ --load-model=dlrm --log-verbose=1 --model-control-mode=explicit --backend-directory=/usr/local/hugectr/backends --backend-config=hugectr,ps=/model/ps.json > /dev/null 2> /dev/null &
#tritonserver --model-repository=/model/ --load-model=dlrm --log-verbose=1 --model-control-mode=explicit --backend-directory=/usr/local/hugectr/backends --backend-config=hugectr,ps=/model/ps.json &

while [[ $(curl -v localhost:8000/v2/health/ready 2>&1 | grep "OK" | wc -l) -eq 0 ]]; do
        sleep 10;
done

perf_analyzer -m dlrm -u localhost:8000 --input-data /perf_data/${BZ}.json --shape CATCOLUMN:${CATCOLUMN} --shape DES:${DES} --shape ROWINDEX:${ROWINDEX}
