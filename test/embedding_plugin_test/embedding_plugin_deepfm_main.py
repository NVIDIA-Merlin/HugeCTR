"""
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

import sys
sys.path.append("../../tools/embedding_plugin/python/")
sys.path.append("../../tools/embedding_plugin/performance_profile/")

import hugectr_tf_ops
from read_data import create_dataset
import txt2tfrecord as utils
from model import DeepFM_PluginEmbedding

import tensorflow as tf
import argparse
import logging
import time

def main(args):
    #---------------- feature description for criteo dataset in tfrecord. ---------- #
    cols = [utils.idx2key(idx, False) for idx in range(0, utils.NUM_TOTAL_COLUMNS)]
    feature_desc = dict()
    for col in cols:
        if col == 'label' or col.startswith("I"):
            feature_desc[col] = tf.io.FixedLenFeature([], tf.int64) # scaler
        else: 
            feature_desc[col] = tf.io.FixedLenFeature([1], tf.int64) # [slot_num, nnz]

    # -------------- create dataset pipeline --------------------------------------- #
    dataset_names = [args.data_path + "/train_0.tfrecord"]
    dataset = create_dataset(dataset_names=dataset_names,
                             feature_desc=feature_desc,
                             batch_size=args.batch_size,
                             n_epochs=args.n_epochs,
                             distribute_keys=tf.constant(True, dtype=tf.bool),
                             gpu_count=len(args.gpus),
                             embedding_type='distributed')

    # ----------- build model and optimizers ---------------------------------------- #
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    model = DeepFM_PluginEmbedding(vocabulary_size=args.vocabulary_size, embedding_vec_size=args.embedding_vec_size, 
                which_embedding="Plugin", embedding_type="distributed",
                dropout_rate=[0.5] * 10, deep_layers=[1024] * 10,
                initializer='uniform', gpus=args.gpus, batch_size=args.batch_size, batch_size_eval=args.batch_size_eval,
                slot_num=args.slot_num)
    
    # ----------- define train step ------------------------------------------------- #
    @tf.function
    def _train_step(dense_batch, sparse_batch, label_batch, model, loss_fn, optimizer):
        with tf.GradientTape() as tape:
            label_batch = tf.cast(label_batch, dtype=tf.float32)
            logits = model(dense_batch, sparse_batch, training=True)
            loss = loss_fn(label_batch, logits)
            loss /= dense_batch.shape[0]
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss

    # ------------------------ training loop ---------------------------------------- #
    logging.info("Begin to train..")
    begin_time = time.time()
    display_begin = begin_time

    for step, datas in enumerate(dataset):
        label, dense, sparse = datas[0], datas[1], datas[2:-1]

        train_loss = _train_step(dense, sparse, label, model, loss_fn, optimizer)
        loss_v = train_loss.numpy()

        if (step % args.display == 0 and step != 0):
            display_end = time.time()
            logging.info("step: %d, loss: %.7f, elapsed time: %.5f seconds." %(step, loss_v, (display_end - display_begin)))
            display_begin = display_end

    end_time = time.time()
    logging.info("Train end. Elapsed time: %.3f seconds." %(end_time - begin_time))

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.root.setLevel('INFO')

    parser = argparse.ArgumentParser(description="Embedding Plugin Testing")
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--batch_size_eval', type=int, required=True)
    parser.add_argument('--n_epochs', type=int, required=True)
    parser.add_argument('--gpus', nargs="+", type=int, required=True)
    parser.add_argument('--vocabulary_size', type=int, required=False, default=1737710)
    parser.add_argument('--embedding_vec_size', type=int, required=True)
    parser.add_argument('--slot_num', type=int, required=False, default=26)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--display', type=int, required=False, default=50)

    args = parser.parse_args()

    main(args)


"""
python3 embedding_plugin_deepfm_main.py --batch_size=16384 --batch_size_eval=4 --n_epochs=1 --gpus 0 1 3 4 \
--embedding_vec_size=10 --data_path=../../tools/embedding_plugin/performance_profile/

python3 embedding_plugin_deepfm_main.py --batch_size=16384 --batch_size_eval=4 --n_epochs=1 --gpus 0 \
--embedding_vec_size=10 --data_path='../../tools/embedding_plugin/performance_profile/'
"""
