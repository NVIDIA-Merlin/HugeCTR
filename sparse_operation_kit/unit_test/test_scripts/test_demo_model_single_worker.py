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

import sys
sys.path.append("../../") # where to find plugin
import sparse_operation_kit as sok
import tensorflow as tf

import pickle
import utils

class SOKDemo(tf.keras.models.Model):
    def __init__(self,
                 combiner,
                 max_vocabulary_size_per_gpu,
                 slot_num,
                 max_nnz,
                 embedding_vec_size, 
                 **kwargs):
        super(SOKDemo, self).__init__(**kwargs)

        self.combiner = combiner
        self.max_vocabulary_size_per_gpu = max_vocabulary_size_per_gpu
        self.slot_num = slot_num
        self.max_nnz = max_nnz
        self.embedding_vec_size = embedding_vec_size

        self.embedding_layer = sok.DistributedEmbedding(combiner=self.combiner,
                                                           max_vocabulary_size_per_gpu=self.max_vocabulary_size_per_gpu,
                                                           embedding_vec_size=self.embedding_vec_size,
                                                           slot_num=self.slot_num,
                                                           max_nnz=self.max_nnz)

        self.dense_layer = tf.keras.layers.Dense(units=1, activation=None,
                                                 kernel_initializer="ones",
                                                 bias_initializer="zeros")

    def call(self, inputs, training=True):
        # [batchsize, slot_num, embedding_vec_size]
        embedding_vector = self.embedding_layer(inputs, training=training)
        # [batchsize, slot_num * embedding_vec_size]
        embedding_vector = tf.reshape(embedding_vector, shape=[-1, self.slot_num * self.embedding_vec_size])
        # [batchsize, 1]
        logit = self.dense_layer(embedding_vector)
        return logit, embedding_vector

class TfDemo(tf.keras.models.Model):
    def __init__(self, 
                 init_tensors, 
                 combiner, 
                 global_batch_size,
                 slot_num, 
                 embedding_vec_size,
                 **kwargs):
        super(TfDemo, self).__init__(**kwargs)
        self.combiner = combiner
        self.global_batch_size = global_batch_size
        self.slot_num = slot_num
        self.embedding_vec_size = embedding_vec_size

        self.init_tensors = init_tensors
        self.params = tf.Variable(initial_value=tf.concat(self.init_tensors, axis=0))

        self.dense_layer = tf.keras.layers.Dense(units=1, activation=None,
                                                 kernel_initializer="ones",
                                                 bias_initializer="zeros")

    def call(self, inputs, training=True):
        # [batchsize * slot_num, embedding_vec_size]
        embedding_vector = tf.nn.embedding_lookup_sparse(params=self.params, sp_ids=inputs,
                                                        sp_weights=None, combiner=self.combiner)

        # [batchsize, slot_num * embedding_vec_size]
        embedding_vector = tf.reshape(embedding_vector, shape=[self.global_batch_size, self.slot_num * self.embedding_vec_size])
        logit = self.dense_layer(embedding_vector)
        return logit, embedding_vector

def test_sok_demo(args, init_tensors, *random_samples):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        result = sok.Init(global_batch_size=args.global_batch_size)

        plugin_demo = SOKDemo(combiner=args.combiner, 
                                 max_vocabulary_size_per_gpu=args.max_vocabulary_size_per_gpu,
                                 slot_num=args.slot_num, max_nnz=args.max_nnz,
                                 embedding_vec_size=args.embedding_vec_size)

        emb_opt = utils.get_embedding_optimizer(args.optimizer)(learning_rate=0.1)
        dense_opt = utils.get_dense_optimizer(args.optimizer)(learning_rate=0.1)

    plugin_saver = sok.Saver()
    status = plugin_saver.load_tensors_to_variable(plugin_demo.embedding_layer.embedding_variable, init_tensors)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    def _replica_loss(labels, logits):
        loss = loss_fn(labels, logits)
        return tf.nn.compute_average_loss(loss, global_batch_size=args.global_batch_size)

    @tf.function
    def _train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logit, embedding_vector = plugin_demo(inputs, training=True)
            loss = _replica_loss(labels, logit)
        embedding_variables, other_variable = sok.split_embedding_variable_from_others(plugin_demo.trainable_variables)
        grads, emb_grads = tape.gradient(loss, [other_variable, embedding_variables])
        if 'plugin' not in args.optimizer:
            with sok.OptimizerScope(embedding_variables):
                emb_opt.apply_gradients(zip(emb_grads, embedding_variables),
                                        experimental_aggregate_gradients=False)
        else:
            emb_opt.apply_gradients(zip(emb_grads, embedding_variables),
                                    experimental_aggregate_gradients=False)
        dense_opt.apply_gradients(zip(grads, other_variable))
        return logit, embedding_vector

    sok_results = list()

    def _dataset_fn(input_context):
        replica_batch_size = input_context.get_per_replica_batch_size(args.global_batch_size)
        dataset = utils.tf_dataset(*random_samples, batchsize=replica_batch_size, to_sparse_tensor=True, repeat=1)
        dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
        return dataset

    dataset = strategy.distribute_datasets_from_function(_dataset_fn)
    
    for i, (sparse_tensors, replica_labels) in enumerate(dataset):
        print("-" * 30, "step ", str(i), "-" * 30)
        logit, embedding_vector = strategy.run(_train_step, args=(sparse_tensors, replica_labels))
        print("[INFO]: embedding_vector\n", embedding_vector)
        sok_results.append(embedding_vector)

        # FIXME: when the forward computation is too fast, there
        # may exist some conficts with datareader, which cause the program hang.
        import time
        time.sleep(0.2) # seconds
    
    return sok_results

def test_tf_demo(args, init_tensors, *random_samples):
    dataset = utils.tf_dataset(*random_samples, batchsize=args.global_batch_size, to_sparse_tensor=True, repeat=1)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    tf_demo = TfDemo(init_tensors, args.combiner, args.global_batch_size, args.slot_num, args.embedding_vec_size)

    optimizer = utils.get_dense_optimizer(args.optimizer)(learning_rate=0.1)

    @tf.function
    def _train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logit, embedding_vector = tf_demo(inputs, training=True)
            loss = loss_fn(labels, logit)
        grads = tape.gradient(loss, tf_demo.trainable_variables)
        optimizer.apply_gradients(zip(grads, tf_demo.trainable_variables))
        return logit, embedding_vector

    tf_results = list()

    for i, (sparse_tensors, labels) in enumerate(dataset):
        print("-"*30, str(i), "-"*30)
        logit, embedding_vector = _train_step(sparse_tensors, labels)
        print("[INFO]: embedding_vector:\n", embedding_vector)
        tf_results.append(embedding_vector)

        # FIXME: because plugin sleepd, here is only used for 
        # simulate the same DNN structure. 
        import time
        time.sleep(0.2) # seconds

    return tf_results


def compare_sok_with_tf(args):
    if (args.global_batch_size % args.gpu_num != 0):
        raise ValueError("global_batch_size: %d is not divisible by gpu_num: %d" 
            %(args.global_batch_size, args.gpu_num))

    if args.generate_new_datas:
        random_samples = utils.generate_random_samples(num_of_samples=args.global_batch_size * args.iter_num,
                                                    vocabulary_size=args.gpu_num * args.max_vocabulary_size_per_gpu * 1,
                                                    slot_num=args.slot_num,
                                                    max_nnz=args.max_nnz)
        utils.save_to_file(r"./random_samples.file", *random_samples)
    else:
        random_samples = utils.restore_from_file(r"./random_samples.file")

    init_tensors = utils.get_ones_tensor(max_vocab_size_per_gpu=args.max_vocabulary_size_per_gpu,
                                         embedding_vec_size=args.embedding_vec_size,
                                         num=args.gpu_num)

    sok_results = test_sok_demo(args, init_tensors, *random_samples)
    tf_results = test_tf_demo(args, init_tensors, *random_samples)

    if (len(sok_results) != len(tf_results)):
        raise ValueError("The length of plugin results is not equal to that of tensorflow.")
    if (len(tf_results) != args.iter_num):
        raise ValueError("The length of embedding vectors: %d is not equal to iteration number: %d."
                         %(len(tf_results), args.iter_num))

    for i, sok_vector in enumerate(sok_results):
        if args.gpu_num != 1:
            sok_vector = tf.stack(sok_vector.values, axis=0)
        tf.debugging.assert_near(tf.reshape(sok_vector,
                                            shape=[-1, tf.shape(sok_vector)[-1]]),
                                tf_results[i],
                                atol=1e-4,
                                rtol=1e-4)
    print("\n[INFO]: With MirroredStrategy, the embedding vector obtained from " +\
          "sparse operation kit and tensorflow are consistent for %d iterations." 
          %args.iter_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test demo model with single worker.')
    parser.add_argument('--gpu_num', type=int,
                        help='the number of GPUs used to do paralell training.',
                        required=False, default=8)
    parser.add_argument('--iter_num', type=int,
                        help='the number of testing iterations.',
                        required=False, default=100)
    parser.add_argument('--max_vocabulary_size_per_gpu', type=int,
                        required=False, default=128)
    parser.add_argument('--slot_num', type=int,
                        help='the number of feature fields',
                        required=False, default=1)
    parser.add_argument('--max_nnz', type=int,
                        help='the maximum number of keys in one slot',
                        required=False, default=1)
    parser.add_argument('--embedding_vec_size', type=int,
                        help='the dimention of embedding vector',
                        required=False, default=1)
    parser.add_argument('--combiner', type=str,
                        help='the combiner used to do reduction for sparse embedding layer. ' +\
                             'It is only respected in sparse embedding layer.',
                        required=False, default='mean', choices=['mean', 'sum'])
    parser.add_argument('--global_batch_size', type=int, required=False, default=16)
    parser.add_argument('--optimizer', type=str,
                        help="use what optimizer",
                        required=False, default='plugin_adam',
                        choices=['plugin_adam', 'adam', 'sgd'])
    parser.add_argument('--generate_new_datas', type=int, choices=[0, 1],
                        help='whether to generate new random samples',
                        required=False, default=1)

    args = parser.parse_args()

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(i) for i in range(args.gpu_num)])

    compare_sok_with_tf(args)