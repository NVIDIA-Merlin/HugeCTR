"""
 Copyright (c) 2021, NVIDIA CORPORATION.
 
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

import argparse
import tensorflow as tf
from model import DemoModel, sok
from gen_data import tf_dataset, restore_from_file
import os
import nvtx

def main(args):
    samples, labels = restore_from_file(args.filename)

    dataset = tf_dataset(samples, labels, batchsize=args.global_batch_size,
                         to_sparse_tensor=False, repeat=1)
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device(
        device='/GPU:0', buffer_size=tf.data.AUTOTUNE))

    model = DemoModel(max_vocabulary_size_per_gpu=args.vocabulary_size,
                      embedding_vec_size=args.embedding_vec_size,
                      slot_num=args.slot_num,
                      nnz_per_slot=args.nnz_per_slot,
                      num_dense_layers=args.num_dense_layers,
                      use_sok=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, 
                        reduction=tf.keras.losses.Reduction.NONE)
    def _replica_loss(labels, logits):
        loss = loss_fn(labels, logits)
        return tf.nn.compute_average_loss(loss, global_batch_size=args.global_batch_size)
                    

    @tf.function
    def _train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logit = model(inputs)
            loss = _replica_loss(labels, logit)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for step, (inputs, labels) in enumerate(dataset):
        if (-1 != args.early_stop_iter) and (step >= args.early_stop_iter):
            break

        rng = nvtx.start_range(message="Iteration_" + str(step), color='blue')

        loss = _train_step(inputs, labels)
        tf.print("[INFO]: Iter: %d, Loss: %.5f" %(step, loss))

        nvtx.end_range(rng)

    print("[INFO]: Profiling TF on single GPU done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--global_batch_size", type=int,
                        required=True,
                        help="the global batch-size used in each iteration.")
    parser.add_argument("--slot_num", type=int,
                        required=True,
                        help="the number of feature fields")
    parser.add_argument("--nnz_per_slot", type=int,
                        required=True,
                        help="the number of valid keys in each slot")
    parser.add_argument("--embedding_vec_size", type=int,
                        required=True,
                        help="the size of embedding vector")
    parser.add_argument("--num_dense_layers", type=int,
                        required=True,
                        help="how many dense layers except the last output one.")
    parser.add_argument("--vocabulary_size", type=int,
                        required=True)
    parser.add_argument("--early_stop_iter", type=int,
                        required=True,
                        help="early stop at which iter")
    parser.add_argument("--filename", type=str,
                        required=True)
    parser.add_argument("--data_splited", type=int, choices=[0, 1],
                        required=True,
                        help="whether the datas is splited.")
    parser.add_argument("--sparse_keys", type=int, choices=[0, 1],
                        required=False, default=0,
                        help="whether the dataset are sparse.")

    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main(args)