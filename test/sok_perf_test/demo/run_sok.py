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

def on_single_gpu(args):
    samples, labels = restore_from_file(args.filename)
    dataset = tf_dataset(samples, labels, batchsize=args.global_batch_size,
                         to_sparse_tensor=False, repeat=1)
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device(
                device="/GPU:0", buffer_size=32))
    
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        sok.Init(global_batch_size=args.global_batch_size)

        model = DemoModel(max_vocabulary_size_per_gpu=args.vocabulary_size,
                        embedding_vec_size=args.embedding_vec_size,
                        slot_num=args.slot_num,
                        nnz_per_slot=args.nnz_per_slot,
                        num_dense_layers=args.num_dense_layers,
                        use_sok=True)

        emb_opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        dense_opt = tf.keras.optimizers.Adam(learning_rate=0.1)

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
        emb_var, other_var = sok.split_embedding_variable_from_others(model.trainable_variables)
        emb_grads, other_grads = tape.gradient(loss, [emb_var, other_var])
        with sok.OptimizerScope(emb_var):
            emb_opt.apply_gradients(zip(emb_grads, emb_var),
                                    experimental_aggregate_gradients=False)
        dense_opt.apply_gradients(zip(other_grads, other_var))
        return loss

    for step, (inputs, labels) in enumerate(dataset):
        if (-1 != args.early_stop_iter) and (step >= args.early_stop_iter):
            break
        
        rng = nvtx.start_range(message="Iteration_" + str(step), color="blue")

        loss = strategy.run(_train_step, args=(inputs, labels))
        tf.print("[INFO]: Iter: %d, Loss: %.5f" %(step, loss))

        nvtx.end_range(rng)

    print("[INFO]: Profiling SOK on single GPU done.")


def on_multi_gpu(args):
    import json

    comm_options = tf.distribute.experimental.CommunicationOptions(
        bytes_per_pack=0,
        timeout_seconds=None,
        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
    )
    port = 12345
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": {"worker": ["localhost" + ":" + str(port + i) 
                                for i in range(args.gpu_num)]},
        "task": {"type": "worker", "index": args.task_id}
    })
    strategy = tf.distribute.MultiWorkerMirroredStrategy(
        communication_options=comm_options)

    if args.data_splited == 1:
        filename = args.filename + str(args.task_id) + ".file"
    else:
        filename = args.filename
    samples, labels = restore_from_file(filename)

    dataset = tf_dataset(samples, labels, batchsize=args.global_batch_size // args.gpu_num,
                         to_sparse_tensor=False, repeat=1)
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device(
                device="/GPU:0", buffer_size=tf.data.AUTOTUNE))

    with strategy.scope():
        sok.Init(global_batch_size=args.global_batch_size)

        model = DemoModel(max_vocabulary_size_per_gpu=args.vocabulary_size,
                          embedding_vec_size=args.embedding_vec_size,
                          slot_num=args.slot_num,
                          nnz_per_slot=args.nnz_per_slot,
                          num_dense_layers=args.num_dense_layers,
                          use_sok=True)

        emb_opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        dense_opt = tf.keras.optimizers.Adam(learning_rate=0.1)

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
        emb_var, other_var = sok.split_embedding_variable_from_others(model.trainable_variables)
        emb_grads, other_grads = tape.gradient(loss, [emb_var, other_var])
        with sok.OptimizerScope(emb_var):
            emb_opt.apply_gradients(zip(emb_grads, emb_var),
                                    experimental_aggregate_gradients=False)

        replica_ctx = tf.distribute.get_replica_context()
        reduced_grads = replica_ctx.all_reduce("sum", other_grads,
                                               options=comm_options)
        dense_opt.apply_gradients(zip(reduced_grads, other_var),
                                  experimental_aggregate_gradients=False)

        loss = replica_ctx.all_reduce("sum", loss, options=comm_options)
        return loss

    for step, (inputs, labels) in enumerate(dataset):
        if (-1 != args.early_stop_iter) and (step >= args.early_stop_iter):
            break
        
        rng = nvtx.start_range(message="Iteration_" + str(step), color="blue")

        loss = strategy.run(_train_step, args=(inputs, labels))
        tf.print("[INFO]: Iter: %d, Loss: %.5f" %(step, loss))

        nvtx.end_range(rng)

    print("[INFO]: Profiling SOK on %d GPU done." %(args.gpu_num))

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
    parser.add_argument("--whether_single_gpu", type=int, choices=[0, 1],
                        required=True, 
                        help="whether profiling on single GPU.")

    args = parser.parse_args()

    if 1 == args.whether_single_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        on_single_gpu(args)
    else:
        args.gpu_num = int(os.getenv("OMPI_COMM_WORLD_SIZE"))
        args.task_id = int(os.getenv("OMPI_COMM_WORLD_RANK"))

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.task_id)
        on_multi_gpu(args)