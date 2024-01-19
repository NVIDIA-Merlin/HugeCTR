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

from time import perf_counter
from typing import Dict

import mlperf_logging.mllog.constants as mlperf_constants
from mlperf_common.logging import MLLoggerWrapper

import hugectr


class LoggingCallback(hugectr.TrainingCallback):
    def __init__(
        self,
        mllogger: MLLoggerWrapper,
        auc_threshold: float,
        max_iter: int,
        batch_size: int,
    ):
        self.mllogger = mllogger
        self.auc_threshold = auc_threshold
        self.iter_per_epoch = max_iter
        self.batch_size = batch_size
        self._success = False
        self._start_time = -1.0
        self._total_time = -1.0
        self._throughput = -1.0
        self._hit_auc_iter = max_iter
        self.minimum_training_time = 0
        super().__init__()

    def _compute_stats(self, current_iter: int):
        self._total_time = perf_counter() - self._start_time
        self._throughput = (current_iter + 1) * self.batch_size / self._total_time

    def on_training_start(self):
        self._start_time = perf_counter()
        self.mllogger.log_init_stop_run_start()
        self.mllogger.start(
            key=mlperf_constants.EPOCH_START,
            metadata={mlperf_constants.EPOCH_NUM: 0},
        )

    def on_training_end(self, current_iter: int):
        epoch_num = current_iter / self.iter_per_epoch
        self.mllogger.end(
            key=mlperf_constants.EPOCH_STOP,
            metadata={mlperf_constants.EPOCH_NUM: epoch_num},
        )
        if not self._success:
            self.mllogger.log_run_stop(status=mlperf_constants.ABORTED, epoch_num=epoch_num)
        self._compute_stats(current_iter)
        if self.minimum_training_time > 0:
            output_max_iter = current_iter + 1
        else:
            output_max_iter = self.iter_per_epoch
        if self.mllogger.comm_handler.global_rank() == 0:
            if self._success:
                print(
                    f"Hit target accuracy AUC {self.auc_threshold:.5f} at "
                    f"{self._hit_auc_iter} / {output_max_iter} iterations with batchsize {self.batch_size} "
                    f"in {self._total_time:.2f}s. Average speed is {self._throughput:.2f} records/s."
                )
            else:
                print(
                    f"Finish {current_iter + 1} iterations with "
                    f"batchsize: {self.batch_size} in {self._total_time:.2f}s."
                )
        self.mllogger.event(
            key="tracked_stats",
            metadata={"step": current_iter / self.iter_per_epoch},
            value={"throughput": self._throughput},
        )

    def on_eval_start(self, current_iter: int) -> bool:
        self.mllogger.start(
            key=mlperf_constants.EVAL_START,
            metadata={mlperf_constants.EPOCH_NUM: current_iter / self.iter_per_epoch},
        )
        return False

    def on_eval_end(self, current_iter: int, eval_results: Dict[str, float]) -> bool:
        epoch_num = current_iter / self.iter_per_epoch
        auc = eval_results["AUC"]
        self.mllogger.event(
            key=mlperf_constants.EVAL_ACCURACY,
            value=auc,
            metadata={mlperf_constants.EPOCH_NUM: epoch_num},
        )
        self.mllogger.end(
            key=mlperf_constants.EVAL_STOP,
            metadata={mlperf_constants.EPOCH_NUM: epoch_num},
        )
        if not self._success:
            self._success = auc >= self.auc_threshold
            if self._success:
                self.mllogger.log_run_stop(status=mlperf_constants.SUCCESS, epoch_num=epoch_num)
                self._hit_auc_iter = current_iter
        self._total_time = perf_counter() - self._start_time
        if self.minimum_training_time > 0:
            if self._total_time < self.minimum_training_time * 60:
                return False
            else:
                return True
        else:
            return self._success
