"""
 Copyright (c) 2023, NVIDIA CORPORATION.
 
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

import hierarchical_parameter_server as hps
import os
import numpy as np
import tensorflow as tf
import struct
import json

args = dict()

args["gpu_num"] = 1  # the number of available GPUs
args["iter_num"] = 10  # the number of training iteration
args["global_batch_size"] = 1024  # the globally batchsize for all GPUs

args["slot_num_per_table"] = [3, 2]  # the number of feature fields for two embedding tables
args["embed_vec_size_per_table"] = [
    16,
    32,
]  # the dimension of embedding vectors for two embedding tables
args["max_vocabulary_size_per_table"] = [
    30000,
    2000,
]  # the vocabulary size for two embedding tables
args["vocabulary_range_per_slot_per_table"] = [
    [[0, 10000], [10000, 20000], [20000, 30000]],
    [[0, 1000], [1000, 2000]],
]
args["max_nnz_per_slot_per_table"] = [
    [4, 2, 3],
    [1, 1],
]  # the max number of non-zeros for each slot for two embedding tables

args["dense_model_path"] = "multi_table_sparse_input_dense.model"
args["ps_config_file"] = "multi_table_sparse_input.json"
args["embedding_table_path"] = [
    "multi_table_sparse_input_sparse_0.model",
    "multi_table_sparse_input_sparse_1.model",
]
args["saved_path"] = "multi_table_sparse_input_tf_saved_model"
args["np_key_type"] = np.int64
args["np_vector_type"] = np.float32
args["tf_key_type"] = tf.int64
args["tf_vector_type"] = tf.float32

hps_config = {
    "supportlonglong": True,
    "models": [
        {
            "model": "multi_table_sparse_input",
            "sparse_files": [
                "multi_table_sparse_input_sparse_0.model",
                "multi_table_sparse_input_sparse_1.model",
            ],
            "num_of_worker_buffer_in_pool": 3,
            "embedding_table_names": ["sparse_embedding0", "sparse_embedding1"],
            "embedding_vecsize_per_table": [16, 32],
            "maxnum_catfeature_query_per_table_per_sample": [9, 2],
            "default_value_for_each_table": [1.0, 1.0],
            "deployed_device_list": [0],
            "max_batch_size": 1024,
            "cache_refresh_percentage_per_iteration": 0.2,
            "hit_rate_threshold": 1.0,
            "gpucacheper": 1.0,
            "gpucache": True,
        }
    ],
}
hps_config_json_object = json.dumps(hps_config, indent=4)
with open(args["ps_config_file"], "w") as outfile:
    outfile.write(hps_config_json_object)

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args["gpu_num"])))


def generate_random_samples(
    num_samples, vocabulary_range_per_slot_per_table, max_nnz_per_slot_per_table
):
    def generate_sparse_keys(
        num_samples, vocabulary_range_per_slot, max_nnz_per_slot, key_dtype=args["np_key_type"]
    ):
        slot_num = len(max_nnz_per_slot)
        max_nnz_of_all_slots = max(max_nnz_per_slot)
        indices = []
        values = []
        for i in range(num_samples):
            for j in range(slot_num):
                vocab_range = vocabulary_range_per_slot[j]
                max_nnz = max_nnz_per_slot[j]
                nnz = np.random.randint(low=1, high=max_nnz + 1)
                entries = sorted(np.random.choice(max_nnz, nnz, replace=False))
                for entry in entries:
                    indices.append([i, j, entry])
                values.extend(
                    np.random.randint(low=vocab_range[0], high=vocab_range[1], size=(nnz,))
                )
        values = np.array(values, dtype=key_dtype)
        return tf.sparse.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=(num_samples, slot_num, max_nnz_of_all_slots),
        )

    def generate_dense_keys(num_samples, vocabulary_range_per_slot, key_dtype=args["np_key_type"]):
        dense_keys = list()
        for vocab_range in vocabulary_range_per_slot:
            keys_per_slot = np.random.randint(
                low=vocab_range[0], high=vocab_range[1], size=(num_samples, 1), dtype=key_dtype
            )
            dense_keys.append(keys_per_slot)
        dense_keys = np.concatenate(np.array(dense_keys), axis=1)
        return dense_keys

    assert len(vocabulary_range_per_slot_per_table) == 2, "there should be two embedding tables"
    assert (
        max(max_nnz_per_slot_per_table[0]) > 1
    ), "the first embedding table has sparse key input (multi-hot)"
    assert (
        min(max_nnz_per_slot_per_table[1]) == 1
    ), "the second embedding table has dense key input (one-hot)"

    sparse_keys = generate_sparse_keys(
        num_samples, vocabulary_range_per_slot_per_table[0], max_nnz_per_slot_per_table[0]
    )
    dense_keys = generate_dense_keys(num_samples, vocabulary_range_per_slot_per_table[1])
    labels = np.random.randint(low=0, high=2, size=(num_samples, 1))
    return sparse_keys, dense_keys, labels


def tf_dataset(sparse_keys, dense_keys, labels, batchsize):
    dataset = tf.data.Dataset.from_tensor_slices((sparse_keys, dense_keys, labels))
    dataset = dataset.batch(batchsize, drop_remainder=True)
    return dataset


class TrainModel(tf.keras.models.Model):
    def __init__(
        self,
        init_tensors_per_table,
        slot_num_per_table,
        embed_vec_size_per_table,
        max_nnz_per_slot_per_table,
        **kwargs
    ):
        super(TrainModel, self).__init__(**kwargs)

        self.slot_num_per_table = slot_num_per_table
        self.embed_vec_size_per_table = embed_vec_size_per_table
        self.max_nnz_per_slot_per_table = max_nnz_per_slot_per_table
        self.max_nnz_of_all_slots_per_table = [max(ele) for ele in self.max_nnz_per_slot_per_table]

        self.init_tensors_per_table = init_tensors_per_table
        self.params0 = tf.Variable(initial_value=tf.concat(self.init_tensors_per_table[0], axis=0))
        self.params1 = tf.Variable(initial_value=tf.concat(self.init_tensors_per_table[1], axis=0))

        self.reshape = tf.keras.layers.Reshape(
            (self.max_nnz_of_all_slots_per_table[0],),
            input_shape=(self.slot_num_per_table[0], self.max_nnz_of_all_slots_per_table[0]),
        )

        self.fc_1 = tf.keras.layers.Dense(
            units=256,
            activation=None,
            kernel_initializer="ones",
            bias_initializer="zeros",
            name="fc_1",
        )
        self.fc_2 = tf.keras.layers.Dense(
            units=256,
            activation=None,
            kernel_initializer="ones",
            bias_initializer="zeros",
            name="fc_2",
        )
        self.fc_3 = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer="ones",
            bias_initializer="zeros",
            name="fc_3",
        )

    def call(self, inputs):
        # SparseTensor of keys, shape: (batch_size*slot_num, max_nnz)
        embeddings0 = tf.reshape(
            tf.nn.embedding_lookup_sparse(
                params=self.params0, sp_ids=inputs[0], sp_weights=None, combiner="mean"
            ),
            shape=[-1, self.slot_num_per_table[0] * self.embed_vec_size_per_table[0]],
        )
        # Tensor of keys, shape: (batch_size, slot_num)
        embeddings1 = tf.reshape(
            tf.nn.embedding_lookup(params=self.params1, ids=inputs[1]),
            shape=[-1, self.slot_num_per_table[1] * self.embed_vec_size_per_table[1]],
        )

        logit = self.fc_3(tf.math.add(self.fc_1(embeddings0), self.fc_2(embeddings1)))
        return logit, embeddings0, embeddings1

    def summary(self):
        inputs = [
            tf.keras.Input(
                shape=(self.max_nnz_of_all_slots_per_table[0],),
                sparse=True,
                dtype=args["tf_key_type"],
            ),
            tf.keras.Input(shape=(self.slot_num_per_table[1],), dtype=args["tf_key_type"]),
        ]
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()


class InferenceModel(tf.keras.models.Model):
    def __init__(
        self,
        slot_num_per_table,
        embed_vec_size_per_table,
        max_nnz_per_slot_per_table,
        dense_model_path,
        **kwargs
    ):
        super(InferenceModel, self).__init__(**kwargs)

        self.slot_num_per_table = slot_num_per_table
        self.embed_vec_size_per_table = embed_vec_size_per_table
        self.max_nnz_per_slot_per_table = max_nnz_per_slot_per_table
        self.max_nnz_of_all_slots_per_table = [max(ele) for ele in self.max_nnz_per_slot_per_table]

        self.sparse_lookup_layer = hps.SparseLookupLayer(
            model_name="multi_table_sparse_input",
            table_id=0,
            emb_vec_size=self.embed_vec_size_per_table[0],
            emb_vec_dtype=args["tf_vector_type"],
        )
        self.lookup_layer = hps.LookupLayer(
            model_name="multi_table_sparse_input",
            table_id=1,
            emb_vec_size=self.embed_vec_size_per_table[1],
            emb_vec_dtype=args["tf_vector_type"],
        )
        self.dense_model = tf.keras.models.load_model(dense_model_path)

    def call(self, inputs):
        # SparseTensor of keys, shape: (batch_size*slot_num, max_nnz)
        embeddings0 = tf.reshape(
            self.sparse_lookup_layer(sp_ids=inputs[0], sp_weights=None, combiner="mean"),
            shape=[-1, self.slot_num_per_table[0] * self.embed_vec_size_per_table[0]],
        )
        # Tensor of keys, shape: (batch_size, slot_num)
        embeddings1 = tf.reshape(
            self.lookup_layer(inputs[1]),
            shape=[-1, self.slot_num_per_table[1] * self.embed_vec_size_per_table[1]],
        )

        logit = self.dense_model([embeddings0, embeddings1])
        return logit, embeddings0, embeddings1

    def summary(self):
        inputs = [
            tf.keras.Input(
                shape=(self.max_nnz_of_all_slots_per_table[0],),
                sparse=True,
                dtype=args["tf_key_type"],
            ),
            tf.keras.Input(shape=(self.slot_num_per_table[1],), dtype=args["tf_key_type"]),
        ]
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()


def train(args):
    def _train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logit, _, _ = model(inputs)
            loss = loss_fn(labels, logit)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return logit, loss

    init_tensors_per_table = [
        np.ones(
            shape=[args["max_vocabulary_size_per_table"][0], args["embed_vec_size_per_table"][0]],
            dtype=args["np_vector_type"],
        ),
        np.ones(
            shape=[args["max_vocabulary_size_per_table"][1], args["embed_vec_size_per_table"][1]],
            dtype=args["np_vector_type"],
        ),
    ]

    model = TrainModel(
        init_tensors_per_table,
        args["slot_num_per_table"],
        args["embed_vec_size_per_table"],
        args["max_nnz_per_slot_per_table"],
    )
    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    sparse_keys, dense_keys, labels = generate_random_samples(
        args["global_batch_size"] * args["iter_num"],
        args["vocabulary_range_per_slot_per_table"],
        args["max_nnz_per_slot_per_table"],
    )
    dataset = tf_dataset(sparse_keys, dense_keys, labels, args["global_batch_size"])
    for i, (sparse_keys, dense_keys, labels) in enumerate(dataset):
        sparse_keys = tf.sparse.reshape(sparse_keys, [-1, sparse_keys.shape[-1]])
        inputs = [sparse_keys, dense_keys]
        _, loss = _train_step(inputs, labels)
        print("-" * 20, "Step {}, loss: {}".format(i, loss), "-" * 20)
    return model


def create_and_save_inference_graph(args):
    model = InferenceModel(
        args["slot_num_per_table"],
        args["embed_vec_size_per_table"],
        args["max_nnz_per_slot_per_table"],
        args["dense_model_path"],
    )
    model.summary()
    inputs = [
        tf.keras.Input(
            shape=(max(args["max_nnz_per_slot_per_table"][0]),),
            sparse=True,
            dtype=args["tf_key_type"],
        ),
        tf.keras.Input(shape=(args["slot_num_per_table"][1],), dtype=args["tf_key_type"]),
    ]
    _, _, _ = model(inputs)
    model.save(args["saved_path"])


def convert_to_sparse_model(embeddings_weights, embedding_table_path, embedding_vec_size):
    os.system("mkdir -p {}".format(embedding_table_path))
    with open("{}/key".format(embedding_table_path), "wb") as key_file, open(
        "{}/emb_vector".format(embedding_table_path), "wb"
    ) as vec_file:
        for key in range(embeddings_weights.shape[0]):
            vec = embeddings_weights[key]
            key_struct = struct.pack("q", key)
            vec_struct = struct.pack(str(embedding_vec_size) + "f", *vec)
            key_file.write(key_struct)
            vec_file.write(vec_struct)


def inference_with_saved_model(args):
    hps.Init(global_batch_size=args["global_batch_size"], ps_config_file=args["ps_config_file"])
    model = tf.keras.models.load_model(args["saved_path"])
    model.summary()

    def _infer_step(inputs, labels):
        logit, embeddings0, embeddings1 = model(inputs)
        return logit, embeddings0, embeddings1

    embeddings0_peek = list()
    embeddings1_peek = list()
    inputs_peek = list()
    logit_peek = list()
    sparse_keys, dense_keys, labels = generate_random_samples(
        args["global_batch_size"] * args["iter_num"],
        args["vocabulary_range_per_slot_per_table"],
        args["max_nnz_per_slot_per_table"],
    )
    dataset = tf_dataset(sparse_keys, dense_keys, labels, args["global_batch_size"])
    for i, (sparse_keys, dense_keys, labels) in enumerate(dataset):
        sparse_keys = tf.sparse.reshape(sparse_keys, [-1, sparse_keys.shape[-1]])
        inputs = [sparse_keys, dense_keys]
        logit, embeddings0, embeddings1 = _infer_step(inputs, labels)
        embeddings0_peek.append(embeddings0)
        embeddings1_peek.append(embeddings1)
        inputs_peek.append(inputs)
        logit_peek.append(logit)
        print("-" * 20, "Step {}".format(i), "-" * 20)
    return embeddings0_peek, embeddings1_peek, inputs_peek, logit_peek


def test_multi_table_sparse_input_hps():
    trained_model = train(args)
    weights_list = trained_model.get_weights()
    embedding_weights_per_table = weights_list[-2:]
    dense_model = tf.keras.Model(
        [trained_model.get_layer("fc_1").input, trained_model.get_layer("fc_2").input],
        trained_model.get_layer("fc_3").output,
    )
    dense_model.summary()
    dense_model.save(args["dense_model_path"])
    convert_to_sparse_model(
        embedding_weights_per_table[0],
        args["embedding_table_path"][0],
        args["embed_vec_size_per_table"][0],
    )
    convert_to_sparse_model(
        embedding_weights_per_table[1],
        args["embedding_table_path"][1],
        args["embed_vec_size_per_table"][1],
    )

    create_and_save_inference_graph(args)
    embeddings0, embeddings1, inputs, logit = inference_with_saved_model(args)

    embeddings0_gt = []
    embeddings1_gt = []
    logit_gt = []
    for i in range(len(inputs)):
        embeddings0_gt.append(
            tf.nn.embedding_lookup_sparse(
                params=embedding_weights_per_table[0],
                sp_ids=inputs[i][0],
                sp_weights=None,
                combiner="mean",
            )
        )
        embeddings1_gt.append(
            tf.nn.embedding_lookup(params=embedding_weights_per_table[1], ids=inputs[i][1])
        )
        logit_gt_batch, _, _ = trained_model(inputs[i])
        logit_gt.append(logit_gt_batch)

    embeddings0 = np.array(
        tf.reshape(
            tf.concat(embeddings0, axis=0),
            [
                -1,
            ],
        )
    )
    embeddings1 = np.array(
        tf.reshape(
            tf.concat(embeddings1, axis=0),
            [
                -1,
            ],
        )
    )
    logit = np.array(
        tf.reshape(
            tf.concat(logit, axis=0),
            [
                -1,
            ],
        )
    )
    embeddings0_gt = np.array(
        tf.reshape(
            tf.concat(embeddings0_gt, axis=0),
            [
                -1,
            ],
        )
    )
    embeddings1_gt = np.array(
        tf.reshape(
            tf.concat(embeddings1_gt, axis=0),
            [
                -1,
            ],
        )
    )
    logit_gt = np.array(
        tf.reshape(
            tf.concat(logit_gt, axis=0),
            [
                -1,
            ],
        )
    )

    diff1 = embeddings0 - embeddings0_gt
    diff2 = embeddings1 - embeddings1_gt
    diff3 = logit - logit_gt
    mse1 = np.mean(diff1 * diff1)
    mse2 = np.mean(diff2 * diff2)
    mse3 = np.mean(diff3 * diff3)
    assert mse1 <= 1e-6
    assert mse2 <= 1e-6
    assert mse3 <= 1e-6
    print(
        "HPS TF Plugin does well for embedding vector lookup, mse1: {}, mse2: {}, mse3: {}".format(
            mse1, mse2, mse3
        )
    )
