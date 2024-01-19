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

from argparse import Namespace

import mlperf_logging.mllog.constants as mllog_constants
from mlperf_common.logging import MLLoggerWrapper

# Parameters not supported in HugeCTR:
ADAGRAD_LR_DECAY = 0
WEIGHT_DECAY = 0
GRADIENT_ACC_STEPS = 1


def param_info(mllogger: MLLoggerWrapper, args: Namespace):
    mllogger.event(
        key=mllog_constants.GLOBAL_BATCH_SIZE,
        value=args.batchsize,
    )
    mllogger.event(
        key=mllog_constants.OPT_NAME,
        value=args.optimizer,
    )
    mllogger.event(
        key=mllog_constants.OPT_BASE_LR,
        value=args.lr,
    )
    mllogger.event(
        key=mllog_constants.OPT_ADAGRAD_LR_DECAY,
        value=ADAGRAD_LR_DECAY,
    )
    mllogger.event(
        key=mllog_constants.OPT_WEIGHT_DECAY,
        value=WEIGHT_DECAY,
    )
    mllogger.event(
        key=mllog_constants.OPT_ADAGRAD_INITIAL_ACCUMULATOR_VALUE,
        value=args.init_accu,
    )
    mllogger.event(
        key=mllog_constants.OPT_ADAGRAD_EPSILON,
        value=args.eps,
    )
    mllogger.event(
        key=mllog_constants.OPT_LR_WARMUP_STEPS,
        value=args.warmup_steps,
    )
    mllogger.event(
        key=mllog_constants.OPT_LR_DECAY_START_STEP,
        value=args.decay_start,
    )
    mllogger.event(
        key=mllog_constants.OPT_LR_DECAY_STEPS,
        value=args.decay_steps,
    )
    mllogger.event(
        key=mllog_constants.GRADIENT_ACCUMULATION_STEPS,
        value=GRADIENT_ACC_STEPS,
    )
