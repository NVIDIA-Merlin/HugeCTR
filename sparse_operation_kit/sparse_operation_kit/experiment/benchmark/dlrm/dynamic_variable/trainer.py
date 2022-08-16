"""
 Copyright (c) 2022, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import time
import tensorflow as tf

from sparse_operation_kit import experiment as sok


def evaluate(model, dataset, thresholds):
    auc = tf.keras.metrics.AUC(
        num_thresholds=thresholds, curve="ROC", summation_method="interpolation", from_logits=True
    )

    @tf.function
    def _step(samples, labels):
        probs = model(samples, training=False)
        auc.update_state(labels, probs)

    for idx, (samples, labels) in enumerate(dataset):
        _step(samples, labels)

    return auc.result().numpy()


class LearningRateScheduler:
    """
    LR Scheduler combining Polynomial Decay with Warmup at the beginning.
    TF-based cond operations necessary for performance in graph mode.
    """

    def __init__(self, optimizers, base_lr, warmup_steps, decay_start_step, decay_steps):
        self.optimizers = optimizers
        self.warmup_steps = tf.constant(warmup_steps, dtype=tf.int32)
        self.decay_start_step = tf.constant(decay_start_step, dtype=tf.int32)
        self.decay_steps = tf.constant(decay_steps)
        self.decay_end_step = decay_start_step + decay_steps
        self.poly_power = 2
        self.base_lr = base_lr
        with tf.device("/CPU:0"):
            self.step = tf.Variable(0)

    @tf.function
    def __call__(self):
        with tf.device("/CPU:0"):
            # used for the warmup stage
            warmup_step = tf.cast(1 / self.warmup_steps, tf.float32)
            lr_factor_warmup = 1 - tf.cast(self.warmup_steps - self.step, tf.float32) * warmup_step
            lr_factor_warmup = tf.cast(lr_factor_warmup, tf.float32)

            # used for the constant stage
            lr_factor_constant = tf.cast(1.0, tf.float32)

            # used for the decay stage
            lr_factor_decay = (self.decay_end_step - self.step) / self.decay_steps
            lr_factor_decay = tf.math.pow(lr_factor_decay, self.poly_power)
            lr_factor_decay = tf.cast(lr_factor_decay, tf.float32)

            poly_schedule = tf.cond(
                self.step < self.decay_start_step,
                lambda: lr_factor_constant,
                lambda: lr_factor_decay,
            )

            lr_factor = tf.cond(
                self.step < self.warmup_steps, lambda: lr_factor_warmup, lambda: poly_schedule
            )

            lr = self.base_lr * lr_factor
            for optimizer in self.optimizers:
                optimizer.lr.assign(lr)

            self.step.assign(self.step + 1)


def scale_grad(grad, factor):
    if isinstance(grad, tf.IndexedSlices):
        # sparse gradient
        grad._values = grad._values * factor
        return grad
    else:
        # dense gradient
        return grad * factor


class Trainer:
    def __init__(
        self,
        model,
        dataset,
        test_dataset,
        auc_thresholds,
        base_lr,
        warmup_steps,
        decay_start_step,
        decay_steps,
        use_tf_optimizer,
    ):
        base_lr = float(base_lr)

        self._model = model
        self._dataset = dataset
        self._test_dataset = test_dataset
        self._auc_thresholds = auc_thresholds
        self._use_tf_optimizer = use_tf_optimizer

        self._loss_fn = tf.losses.BinaryCrossentropy(from_logits=True)

        optimizers = []
        self._dense_optimizer = tf.keras.optimizers.SGD(base_lr)
        optimizers.append(self._dense_optimizer)
        if not self._model._use_tf:
            if self._use_tf_optimizer:
                self._sok_optimizer = sok.OptimizerWrapper(tf.keras.optimizers.SGD(base_lr))
            else:
                self._sok_optimizer = sok.SGD(base_lr)
            optimizers.append(self._sok_optimizer)
        self._lr_scheduler = LearningRateScheduler(
            optimizers,
            base_lr,
            warmup_steps,
            decay_start_step,
            decay_steps,
        )

    @tf.function
    def _step(self, samples, labels):
        self._lr_scheduler()

        with tf.GradientTape() as tape:
            probs = self._model(samples, training=True)
            loss = self._loss_fn(labels, probs)

        if self._model._use_tf:
            dense_vars = self._model.trainable_variables
            dense_grads = tape.gradient(loss, dense_vars)
            for g in dense_grads:
                if isinstance(g, tf.IndexedSlices):
                    scale_grad(g, 8.0)
            self._dense_optimizer.apply_gradients(zip(dense_grads, dense_vars))
        else:
            all_vars = self._model.trainable_variables
            sok_vars, dense_vars = [], []
            for var in all_vars:
                if isinstance(var, sok.DynamicVariable):
                    sok_vars.append(var)
                else:
                    dense_vars.append(var)
            sok_grads, dense_grads = tape.gradient(loss, [sok_vars, dense_vars])
            for g in sok_grads:
                scale_grad(g, 8.0)
            self._dense_optimizer.apply_gradients(zip(dense_grads, dense_vars))
            self._sok_optimizer.apply_gradients(zip(sok_grads, sok_vars))

        return loss

    def train(self, interval=1000, eval_interval=3793, eval_in_last=False, early_stop=-1, epochs=1):
        eval_time = 0
        iter_time = time.time()
        total_time = time.time()
        throughputs = []
        for epoch in range(epochs):
            early_stop_flag = False
            for i, (samples, labels) in enumerate(self._dataset):
                idx = epoch * len(self._dataset) + i

                loss = self._step(samples, labels)

                if idx == 0:
                    print(
                        "Iteration 0 finished. The following log will be printed every %d iterations."
                        % interval
                    )

                if (idx % interval == 0) and (idx > 0):
                    t = time.time() - iter_time
                    throughput = interval * self._dataset._batch_size / t
                    print(
                        "Iteration:%d\tloss:%.6f\ttime:%.2fs\tthroughput:%.2fM"
                        % (idx, loss, t, throughput / 1000000)
                    )
                    throughputs.append(throughput)
                    iter_time = time.time()

                if (eval_interval is not None) and (idx % eval_interval == 0) and (idx > 0):
                    t = time.time()
                    auc = evaluate(self._model, self._test_dataset, self._auc_thresholds)
                    t = time.time() - t
                    eval_time += t
                    iter_time += t
                    print(
                        "Evaluate in %dth iteration, test time: %.2fs, AUC: %.6f." % (idx, t, auc)
                    )
                    if auc > 0.8025:
                        early_stop_flag = True
                        break

                if early_stop > 0 and (idx + 1) >= early_stop:
                    early_stop_flag = True
                    break

            if early_stop_flag:
                break

        if eval_in_last:
            t = time.time()
            auc = evaluate(self._model, self._test_dataset, self._auc_thresholds)
            t = time.time() - t
            eval_time += t
            print("Evaluate in the end, test time: %.2fs, AUC: %.6f." % (t, auc))

        total_time = time.time() - total_time
        training_time = total_time - eval_time
        avg_training_time = training_time / (idx + 1)
        print("total time: %.2fs, in %d iterations" % (total_time, (idx + 1)))
        if len(throughputs[1:]) == 0:
            average_throughput = 0
            average_time_per_iter = 0
        else:
            average_throughput = sum(throughputs[1:]) / len(throughputs[1:])
            average_time_per_iter = self._dataset._batch_size / average_throughput * 1000
        print(
            "only training time: %.2fs, average: %.2fms/iter, average throughput: %.2fM(%.2fms/iter)"
            % (
                training_time,
                avg_training_time * 1000,
                average_throughput / 1000000,
                average_time_per_iter,
            )
        )
        print("only evaluate time: %.2fs" % (eval_time))
