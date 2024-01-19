#!/bin/bash

# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# runs benchmark and reports time to convergence

# default value for DLRM_BIND only if it is not already defined
#: ${DLRM_BIND:="numactl --membind=1,3,5,7"}
: ${DLRM_BIND:=}

set -ex

ARGS=""
[ -n "${OPTIMIZER:-}" ] && ARGS+=" --optimizer ${OPTIMIZER}"
[ -n "${BATCHSIZE:-}" ] && ARGS+=" --batchsize ${BATCHSIZE}"
[ -n "${BATCHSIZE_EVAL:-}" ] && ARGS+=" --batchsize_eval ${BATCHSIZE_EVAL}"
[ -n "${LEARNING_RATE:-}" ] && ARGS+=" --lr ${LEARNING_RATE}"
[ -n "${WARMUP_STEPS:-}" ] && ARGS+=" --warmup_steps ${WARMUP_STEPS}"
[ -n "${DECAY_START:-}" ] && ARGS+=" --decay_start ${DECAY_START}"
[ -n "${DECAY_STEPS:-}" ] && ARGS+=" --decay_steps ${DECAY_STEPS}"
[ "$ENABLE_TF32_COMPUTE" = true ] && ARGS+=" --enable_tf32_compute"
[ "$USE_MIXED_PRECISION" = true ] && ARGS+=" --use_mixed_precision"
[ -n "${SCALER:-}" ] && ARGS+=" --scaler ${SCALER}"
[ "$GEN_LOSS_SUMMARY" = true ] && ARGS+=" --gen_loss_summary"
[ "$USE_ALGORITHM_SEARCH" = false ] && ARGS+=" --disable_algorithm_search"
[ -n "${SHARDING_PLAN:-}" ] && ARGS+=" --sharding_plan ${SHARDING_PLAN}"
[ -n "${DP_SHARDING_THRESHOLD:-}" ] && ARGS+=" --dp_sharding_threshold ${DP_SHARDING_THRESHOLD}"
[ -n "${MAX_ITER:-}" ] && ARGS+=" --max_iter ${MAX_ITER}"
[ -n "${DISPLAY_INTERVAL:-}" ] && ARGS+=" --display_interval ${DISPLAY_INTERVAL}"
[ -n "${EVAL_INTERVAL:-}" ] && ARGS+=" --eval_interval ${EVAL_INTERVAL}"
[ -n "${MAX_EVAL_BATCHES:-}" ] && ARGS+=" --max_eval_batches ${MAX_EVAL_BATCHES}"
[ -n "${AUC_THRESHOLD:-}" ] && ARGS+=" --auc_threshold ${AUC_THRESHOLD}"
[ -n "${DGXNGPU:-}" ] && ARGS+=" --num_gpus_per_node ${DGXNGPU}"
[ -n "${MEM_COMM_BW_RATIO:-}" ] && ARGS+=" --mem_comm_bw_ratio ${MEM_COMM_BW_RATIO}"
[ -n "${SEED:-}" ] && ARGS+=" --seed ${SEED}"
[ -n "${MLPERF_POWER_TRAIN_AFTER_RUN_STOP:-}" ] && ARGS+=" --minimum_training_time ${MINIMUM_TRAINING_TIME:-0}"

readonly node_rank="${SLURM_NODEID:-0}"
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"

if [ "$LOGGER" = "apiLog.sh" ];
then
  LOGGER="${LOGGER} -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
  if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ];
  then
    LOGGER=$LOGGER
  else
    LOGGER=""
  fi
fi

echo "DLRM_BIND is set to \"${DLRM_BIND}\""
${LOGGER} ${DLRM_BIND} python3 ${RUN_SCRIPT} ${ARGS} | tee /tmp/dlrm_hugectr.log

 ret_code=${PIPESTATUS[0]}
 if [[ $ret_code != 0 ]]; then exit $ret_code; fi
