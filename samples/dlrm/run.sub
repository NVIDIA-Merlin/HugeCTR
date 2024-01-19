#!/bin/bash

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#SBATCH --job-name mlperf-dlrm:hugectr
#SBATCH -t 00:30:00

set -euxo pipefail

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"

# Vars with defaults
: "${MLPERF_RULESET:=3.1.0}"
: "${MLPERF_CLUSTER_NAME:='unknown'}"
: "${NEXP:=10}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${CHECK_COMPLIANCE:=1}"
: "${API_LOG_DIR:=./api_logs}" # apiLog.sh output dir
: "${ABSLOGDIR:=${PWD}/results}"
: "${POWERCMDDIR:=' '}"
: "${DATADIR_VAL:=${DATADIR}}"
: "${MOUNTS:=${DATADIR}:/data,${DATADIR_VAL}:/data_val}"
: "${LOGDIR:=./results}"

export MODEL_NAME="recommendation"
export MODEL_FRAMEWORK="pytorch"
LOG_BASE="${DATESTAMP}"
SPREFIX="${MODEL_NAME}_${MODEL_FRAMEWORK}_${DGXNNODES}x${DGXNGPU}x${BATCHSIZE}_${DATESTAMP}"


readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name="${MODEL_NAME}_${SLURM_JOB_ID}"
_cont_mounts=${MOUNTS}

if [ "${API_LOGGING:-}" -eq 1 ]; then
    API_LOG_DIR=${API_LOG_DIR}/${MODEL_FRAMEWORK}/${MODEL_NAME}/${DGXSYSTEM}
    mkdir -p ${API_LOG_DIR}
    _cont_mounts="${_cont_mounts},${API_LOG_DIR}:/logs"

    # Create JSON file for cuDNN
    JSON_MODEL_NAME="MLPERF_${MODEL_NAME}_${MODEL_FRAMEWORK}_train"
    JSON_README_LINK="${README_PREFIX}/${MODEL_NAME}/${MODEL_FRAMEWORK}/README.md"
    JSON_FMT='{model_name: $mn, readme_link: $rl, configs: {($dt): [$bs]}, sweep: {($dt): [$bs]}}'
    JSON_OUTPUT="${JSON_MODEL_NAME}.cudnn.json"
    jq -n --indent 4 --arg mn $JSON_MODEL_NAME --arg rl $JSON_README_LINK --arg dt $APILOG_PRECISION --arg bs $BATCHSIZE "$JSON_FMT" > ${API_LOG_DIR}/$JSON_OUTPUT
fi
if [ "${JET:-0}" -eq 1 ]; then
    _cont_mounts="${_cont_mounts},${JET_DIR}:/root/.jet,${LOGDIR}:/results"
fi

# make sure the results directory exists on the host
( umask 0002; mkdir -p "${LOGDIR}" )

# Setup container
echo MELLANOX_VISIBLE_DEVICES="${MELLANOX_VISIBLE_DEVICES:-}"
srun --mpi="${SLURM_MPI_TYPE:-pmix}" --ntasks="${SLURM_JOB_NUM_NODES}" --container-image="${CONT}" --container-name="${_cont_name}" true
srun -N1 -n1 --container-name="${_cont_name}" ibv_devinfo --list
srun -N1 -n1 --container-name="${_cont_name}" nvidia-smi topo -m

#ssh to nodes for power measurements
NODELIST=$(scontrol show hostnames ${SLURM_JOB_NODELIST})
NODELIST=(${NODELIST[*]})
if [ -f "$POWERCMDDIR/power_monitor.sh"  ]; then
    ( umask 0002; mkdir -p "${ABSLOGDIR}" )
    for i in "${NODELIST[@]}"
    do
        ssh $i 'export NODENAME='"'$i'"';export ABSLOGDIR='"'$ABSLOGDIR'"';export SLURM_JOB_NODELIST='"'$SLURM_JOB_NODELIST'"';export SLURM_JOB_ID='"'$SLURM_JOB_ID'"';POWERCMDDIR='"'$POWERCMDDIR'"';bash ${POWERCMDDIR}/power_monitor.sh' &
#	break
    done
fi

if [[ "${SET_MAXQ_CLK:-}" == "1" ]] || [[ "${SET_MINEDP_CLK:-}" == "1" ]]; then
    if [[ "${SET_MAXQ_CLK:-}" == "1" ]]; then
        GPCCLK=${MAXQ_CLK}
    fi
    if [[ "${SET_MINEDP_CLK:-}" == "1" ]]; then
        GPCCLK=${MINEDP_CLK}
    fi
    for i in "${NODELIST[@]}"
    do
        ssh $i 'export GPCCLK='"'$GPCCLK'"';sudo nvidia-smi -lgc ${GPCCLK}'
    done
fi

# Run experiments
for _experiment_index in $(seq -w 1 "${NEXP}"); do
    (
        echo ":::DLPAL ${CONT} ${SLURM_JOB_ID} ${SLURM_JOB_NUM_NODES} ${SLURM_JOB_NODELIST} ${MLPERF_CLUSTER_NAME} ${DGXSYSTEM}"

        # Print system info
        echo ":::SYSJSON $(srun --ntasks=1 --container-name="${_cont_name}" mlperf-sysjson.sh)"

        if [[ $CLEAR_CACHES == 1 ]]; then
            srun --mpi="${SLURM_MPI_TYPE:-pmix}" --ntasks="${SLURM_JOB_NUM_NODES}" bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
            srun --mpi="${SLURM_MPI_TYPE:-pmix}" --ntasks="${SLURM_JOB_NUM_NODES}" --container-name="${_cont_name}" python3 -c "
import mlperf_logging.mllog as mllog
mllogger = mllog.get_mllogger()
mllogger.event(key=mllog.constants.CACHE_CLEAR, value=True)"
        fi
        echo "Beginning trial ${_experiment_index} of ${NEXP}"
        srun --mpi="${SLURM_MPI_TYPE:-pmix}" --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
             --container-name="${_cont_name}" --container-mounts="${_cont_mounts}" \
             ./run_and_time.sh
    ) |& tee "${_logfile_base}_raw_${_experiment_index}.log"

    # Sorting the MLPerf compliance logs by timestamps
    grep ":::.L..." "${_logfile_base}_raw_${_experiment_index}.log" | sort -k5 -n -s | tee "${_logfile_base}_${_experiment_index}.log"
    if [ "${CHECK_COMPLIANCE}" -eq 1 ]; then
      srun --ntasks=1 --nodes=1 --container-name="${_cont_name}" \
     --container-mounts="$(realpath ${LOGDIR}):/results"   \
     --container-workdir="/results"                        \
     python3 -m mlperf_logging.compliance_checker --usage training \
     --ruleset "${MLPERF_RULESET}"                                 \
     --log_output "/results/compliance_${DATESTAMP}.out"           \
     "/results/${DATESTAMP}_${_experiment_index}.log" \
     || true
    fi

    if [ "${JET:-0}" -eq 1 ]; then
      JET_CREATE=${JET_CREATE:-}" --data workload.spec.nodes=${DGXNNODES} --data workload.spec.name=${MODEL_NAME}_${MODEL_FRAMEWORK}_${DGXSYSTEM} --data workload.key=${MODEL_NAME}_${MODEL_FRAMEWORK}_${DGXSYSTEM} --mllogger "
      srun -N1 -n1 --container-name="${_cont_name}" --container-mounts="${_cont_mounts}" bash -c "${JET_CREATE} /results/${DATESTAMP}_${_experiment_index}.log --asset /results/slurm-${SLURM_JOB_ID}.out --data source_image.name=${CONT} --data slurm.job=${SLURM_JOB_ID} && ${JET_UPLOAD}"
    fi

done
