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
import sys, os
sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../")))
import sparse_operation_kit as sok
import tensorflow as tf
import utils
from dense_models import SOKDemo, TFDemo
import strategy_wrapper

def get_sok_results(args, init_tensors, *random_samples):
    if args.distributed_tool == "onedevice":
        import horovod.tensorflow as hvd
        hvd.init()
        strategy = strategy_wrapper.OneDeviceStrategy()
    elif args.distributed_tool == "horovod":
        import horovod.tensorflow as hvd
        hvd.init()
        strategy = strategy_wrapper.HorovodStrategy()
    elif args.distributed_tool == "mirrored":
        strategy = tf.distribute.MirroredStrategy()
    elif args.distributed_tool == "multiworker":
        raise ValueError(f"{args.distributed_tool} is not supported.")
    else:
        raise ValueError(f"{args.distributed_tool} is not supported.")
    
    with strategy.scope():
        sok_init_op = sok.Init(global_batch_size=args.global_batch_size)

        sok_dense_demo = SOKDemo(max_vocabulary_size_per_gpu=args.max_vocabulary_size_per_gpu,
                                 embedding_vec_size=args.embedding_vec_size,
                                 slot_num=args.slot_num,
                                 nnz_per_slot=args.nnz_per_slot,
                                 use_hashtable=args.use_hashtable,
                                 num_of_dense_layers=0)
        # freeze dense layers' variables
        # for layer in sok_dense_demo.dense_layers:
        #     layer.trainable = False
        # sok_dense_demo.out_layer.trainable = False
        
        emb_opt = utils.get_embedding_optimizer(args.optimizer)(learning_rate=0.1)
        dense_opt = utils.get_dense_optimizer(args.optimizer)(learning_rate=0.1)

    sok_saver = sok.Saver()
    if args.restore_params:
        filepath = r"./embedding_variables"
        restore_op = sok_saver.restore_from_file(sok_dense_demo.embedding_layer.embedding_variable, filepath)
    else:
        restore_op = sok_saver.load_embedding_values(sok_dense_demo.embedding_layer.embedding_variable, init_tensors)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
    def _replica_loss(labels, logits):
        loss = loss_fn(labels, logits)
        return tf.nn.compute_average_loss(loss, global_batch_size=args.global_batch_size)

    def _train_step(inputs, labels, training):
        def _step_fn(inputs, labels):
            logit, embedding_vector = sok_dense_demo(inputs, training=training)
            loss = _replica_loss(labels, logit)
            emb_var, other_var = sok.split_embedding_variable_from_others(sok_dense_demo.trainable_variables)
            grads = tf.gradients(loss, emb_var + other_var, colocate_gradients_with_ops=True,
                                    unconnected_gradients=tf.UnconnectedGradients.NONE)
            emb_grads, other_grads = grads[:len(emb_var)], grads[len(emb_var):]
            if "plugin" in args.optimizer:
                emb_train_op = emb_opt.apply_gradients(zip(emb_grads, emb_var))
            else:
                with sok.OptimizerScope(emb_var):
                    emb_train_op = emb_opt.apply_gradients(zip(emb_grads, emb_var))
            other_train_op = dense_opt.apply_gradients(zip(other_grads, other_var))

            with tf.control_dependencies([emb_train_op, other_train_op]):
                total_loss = strategy.reduce("sum", loss)
                total_loss = tf.identity(total_loss)
                return total_loss, embedding_vector, emb_grads
        return strategy.run(_step_fn, inputs, labels)

    replica_batch_size = args.global_batch_size // args.gpu_num
    dataset = utils.tf_dataset(*random_samples, batchsize=replica_batch_size,
                               to_sparse_tensor=False, repeat=1)
    train_iterator = dataset.make_initializable_iterator()
    iterator_init = train_iterator.initializer

    inputs, labels = train_iterator.get_next()
    graph_results = _train_step(inputs, labels, training=True)
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    if "plugin" in args.optimizer:
        init_op = tf.group(init_op, emb_opt.initializer)

    if args.save_params:
        filepath = r"./embedding_variables/"
        utils.try_make_dirs(filepath)
        save_op = sok_saver.dump_to_file(sok_dense_demo.embedding_layer.embedding_variable, filepath)
    else:
        save_op = tf.constant(1.0)

    sok_results = list()
    with tf.Session() as sess:
        sess.run(sok_init_op)
        sess.run([init_op, iterator_init])
        sess.run(restore_op)
        sess.graph.finalize()
        
        for step in range(args.iter_num):
            loss_v, emb_vector_v, emb_grads_v, inputs_v, labels_v = sess.run([*graph_results, inputs, labels])


            print("*" * 50)
            print(f"Inputs: {inputs_v}, labels: {labels_v}")
            print(f"emb_grads: {emb_grads_v}")
            print(f"Step: {step}, loss: {loss_v}, embedding_vector: {emb_vector_v}")
            sok_results.append(emb_vector_v)

        sess.run(save_op)

            
    if hasattr(sok_dense_demo.embedding_layer.embedding_variable, "values"):
        name = sok_dense_demo.embedding_layer.embedding_variable.values[0].m_var_name
    else:
        name = sok_dense_demo.embedding_layer.embedding_variable.m_var_name

    return sok_results, name



def compare_dense_emb_sok_with_tf(args):
    if args.global_batch_size % args.gpu_num != 0:
        raise ValueError(f"global_batch_size: {args.global_batch_size} is not divisible by"
                         f" gpu_num: {args.gpu_num}")

    if args.use_hashtable:
        vocabulary_size = args.max_vocabulary_size_per_gpu * args.gpu_num
    else:
        vocabulary_size = args.max_vocabulary_size_per_gpu

    if args.generate_new_datas:
        random_samples = utils.generate_random_samples(num_of_samples=args.global_batch_size * args.iter_num,
                                                       vocabulary_size=vocabulary_size,
                                                       slot_num=args.slot_num,
                                                       max_nnz=args.nnz_per_slot,
                                                       use_sparse_mask=False)
        utils.save_to_file(r"./random_samples.file", *random_samples)
    else:
        random_samples = utils.restore_from_file(r"./random_samples.file")

    if args.restore_params:
        filepath = r"./embedding_variables"
        # because we already checked the Variable consistency when saving
        # so that we can directly use TensorFlow Variable file to initialize
        # TF's Variable
        tf_values_filename = ops.path.join(filepath, r"tf_variable.file")
        init_tensors = utils.restore_from_file(tf_values_filename)
    else:
        init_tensors = utils.get_ones_tensor(max_vocab_size_per_gpu=args.max_vocabulary_size_per_gpu,
                                             embedding_vec_size=args.embedding_vec_size,
                                             num=args.gpu_num)

    sok_results, embedding_variable_name = get_sok_results(args, init_tensors, *random_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_num", type=int, help="the number of GPUs used in synchronized training.",
                        required=False, default=1)
    parser.add_argument("--distributed_tool", type=str, help="what is used to do the distributed synchronized training",
                        required=False, choices=["mirrored", "multiworker", "horovod", "onedevice"],
                        default="onedevice")
    parser.add_argument("--iter_num", type=int, help="the number of testing iterations.",
                        required=False, default=50)
    parser.add_argument("--max_vocabulary_size_per_gpu", type=int,
                        required=False, default=1024)
    parser.add_argument("--slot_num", type=int,
                        help="the number of feature fields",
                        required=False, default=1)
    parser.add_argument("--nnz_per_slot", type=int,
                        help="the number of keys in each slot.",
                        required=False, default=1)
    parser.add_argument("--embedding_vec_size", type=int, 
                        required=False, default=1)
    parser.add_argument("--global_batch_size", type=int, required=False, default=16)
    parser.add_argument("--optimizer", type=str, required=False, default="adam", 
                        choices=["plugin_adam", "adam", "sgd"])
    parser.add_argument("--generate_new_datas", type=int, choices=[0, 1],
                        required=False, default=1)
    parser.add_argument("--save_params", type=int, choices=[0, 1],
                        required=False, default=1)
    parser.add_argument("--restore_params", type=int, choices=[0, 1],
                        required=False, default=0)
    parser.add_argument("--use_hashtable", type=int, choices=[0, 1],
                        required=False, default=1)

    args = parser.parse_args()

    args.generate_new_datas = True if args.generate_new_datas == 1 else False
    args.save_params = True if args.save_params == 1 else False
    args.restore_params = True if args.restore_params == 1 else False
    args.use_hashtable = True if args.use_hashtable == 1 else False

    if not (args.distributed_tool == "onedevice" and args.gpu_num == 1):
        raise ValueError(f"When 'onedevice' is used as the distributed_tool, "
                         f"gpu_num must be 1, which is {args.gpu_num}")

    if args.distributed_tool == "mirrored" or args.distributed_tool == "onedevice":
        available_gpus = ",".join(map(str, range(args.gpu_num)))
        rank_size = args.gpu_num
        rank_idx = 0
    else:
        # gpu_num will be ignored.
        rank_size = os.getenv("OMPI_COMM_WORLD_SIZE")
        if rank_size is None:
            raise ValueError(f"When distributed_tool is set to {args.distributed_tool}, "
                             "mpiexec / mpirun must be used to launch this program.")
        rank_size = int(rank_size)
        rank_idx = int(os.getenv("OMPI_COMM_WORLD_RANK"))

        available_gpus = str(rank_idx)

    os.environ["CUDA_VISIBLE_DEVICES"] = available_gpus

    args.rank_size = rank_size
    args.rank_idx = rank_idx
    args.gpu_num = rank_size

    compare_dense_emb_sok_with_tf(args)
        
    

    