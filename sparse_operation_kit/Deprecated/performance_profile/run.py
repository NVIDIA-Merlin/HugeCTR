"""
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import nvtx.plugins.tf as nvtx_tf

import txt2tfrecord as utils
from read_data import create_dataset
import tensorflow as tf
import sys
sys.path.append("../python")
import hugectr_tf_ops
from model import DeepFM_PluginEmbedding, DeepFM_OriginalEmbedding
import argparse
import logging
import time

tf.debugging.set_log_device_placement(False)
devices = tf.config.list_physical_devices("GPU")
for dev in devices:
    tf.config.experimental.set_memory_growth(dev, True)
    
def main(args):
    cols = [utils.idx2key(idx, False) for idx in range(0, utils.NUM_TOTAL_COLUMNS)]
    feature_desc = dict()
    for col in cols:
        if col == 'label' or col.startswith("I"):
            feature_desc[col] = tf.io.FixedLenFeature([], tf.int64) # scaler
        else: 
            feature_desc[col] = tf.io.FixedLenFeature([1], tf.int64) # [slot_num, nnz]

    # dataset_names = ["train_" + str(i) + ".tfrecord" for i in range(10)]
    dataset_names = ["train.tfrecord"]
    dataset = create_dataset(dataset_names=dataset_names, 
                             feature_desc=feature_desc, 
                             batch_size=args.batch_size,
                             n_epochs=args.n_epochs, 
                             distribute_keys=tf.constant(args.distribute_keys != 0, dtype=tf.bool),
                             gpu_count=len(args.gpus), 
                             embedding_type=tf.constant(args.embedding_type, dtype=tf.string))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    if args.which_embedding == "OriginalEmbedding":
        model = DeepFM_OriginalEmbedding(vocabulary_size=args.vocabulary_size, embedding_vec_size=args.embedding_vec_size, 
                   which_embedding=args.which_embedding, embedding_type=args.embedding_type,
                   dropout_rate=[0.5] * 10, deep_layers=[1024] * 10,
                   initializer='uniform', gpus=args.gpus, batch_size=args.batch_size, batch_size_eval=args.batch_size_eval,
                   slot_num=args.slot_num)
    elif args.which_embedding == "PluginEmbedding":
        model = DeepFM_PluginEmbedding(vocabulary_size=args.vocabulary_size, embedding_vec_size=args.embedding_vec_size, 
                   which_embedding=args.which_embedding, embedding_type=args.embedding_type,
                   dropout_rate=[0.5] * 10, deep_layers=[1024] * 10,
                   initializer='uniform', gpus=args.gpus, batch_size=args.batch_size, batch_size_eval=args.batch_size_eval,
                   slot_num=args.slot_num)

    @tf.function
    def _train_step(dense_batch, sparse_batch, y_batch, model, loss_fn, optimizer):
        with tf.GradientTape(persistent=False) as tape:
            y_batch = tf.cast(y_batch, dtype=tf.float32)
            logits = model(dense_batch, sparse_batch, training=True)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss

    logging.info("begin to train.")
    begin_time = time.time()
    train_loss_list = []
    display_begin = begin_time

    # with tf.profiler.experimental.Profile("./origin_1030"):
    for step, items in enumerate(dataset):
        label, dense, others = items[0], items[1], items[2:]
        if (tf.convert_to_tensor(args.distribute_keys != 0, dtype=tf.bool)):
            sparse = others[0:3]
        else:
            sparse = others[-1]

        train_loss = _train_step(dense, sparse, label, model, loss_fn, optimizer)
        loss_value = train_loss.numpy()

        train_loss_list.append(loss_value)
        if (step % args.display == 0 and step != 0):
            display_end = time.time()
            logging.info("step: %d, loss: %.5f, elapsed time: %.5f seconds." %(step, loss_value, (display_end - display_begin)))
            display_begin = display_end
        if step >= 50:
            break

    end_time = time.time()
    logging.info("Train End. Elapsed Time: %.3f seconds." %(end_time - begin_time))



if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.root.setLevel('INFO')

    parser = argparse.ArgumentParser(description='Run DeepFM')

    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--batch_size_eval', type=int, required=True)
    parser.add_argument('--n_epochs', type=int, required=True)
    parser.add_argument('--distribute_keys', type=int, required=True, choices=[0,1])
    parser.add_argument('--gpus', nargs='+', type=int, required=True)
    parser.add_argument('--embedding_type', type=str, required=False, default='localized')
    parser.add_argument('--vocabulary_size', type=int, required=True)
    parser.add_argument('--embedding_vec_size', type=int, required=True)
    parser.add_argument('--which_embedding', type=str, required=False, choices=['OriginalEmbedding', 'PluginEmbedding'],
                        default='PluginEmbedding')
    parser.add_argument('--slot_num', type=int, required=False, default=1)
    parser.add_argument('--display', type=int, required=False, default=100)

    args = parser.parse_args()

    main(args)


"""ww
python3 run.py --batch_size=16384 --n_epochs=1 --distribute_keys=1 --gpus 0 1 3 4 --embedding_type='distributed' \
 --vocabulary_size=1737710 --embedding_vec_size=10 --slot_num=26 --batch_size_eval=4

python3 run.py --batch_size=16384 --n_epochs=1 --distribute_keys=0 --gpus 0 1 3 4 \
--vocabulary_size=1737710 --embedding_vec_size=10 --slot_num=26 --batch_size_eval=4 --which_embedding=OriginalEmbedding
"""