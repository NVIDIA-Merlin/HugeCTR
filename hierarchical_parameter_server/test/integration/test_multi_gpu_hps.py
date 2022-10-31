import hierarchical_parameter_server as hps
import os
import numpy as np
import tensorflow as tf
import struct
import json
import atexit

args = dict()

args["gpu_num"] = 4  # the number of available GPUs
args["iter_num"] = 10  # the number of training iteration
args["slot_num"] = 10  # the number of feature fields in this embedding layer
args["embed_vec_size"] = 16  # the dimension of embedding vectors
args["dense_dim"] = 10  # the dimension of dense features
args["global_batch_size"] = 1024  # the globally batchsize for all GPUs
args["max_vocabulary_size"] = 100000
args["vocabulary_range_per_slot"] = [[i * 10000, (i + 1) * 10000] for i in range(10)]
args["max_nnz"] = 5  # the max number of non-zeros for all slots
args["combiner"] = "mean"

args["ps_config_file"] = "multi_gpu.json"
args["dense_model_path"] = "multi_gpu_dense.model"
args["embedding_table_path"] = "multi_gpu_sparse.model"
args["saved_path"] = "multi_gpu_tf_saved_model"
args["np_key_type"] = np.int64
args["np_vector_type"] = np.float32
args["tf_key_type"] = tf.int64
args["tf_vector_type"] = tf.float32

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args["gpu_num"])))

hps_config = {
    "supportlonglong": True,
    "models": [
        {
            "model": "multi_gpu",
            "sparse_files": ["multi_gpu_sparse.model"],
            "num_of_worker_buffer_in_pool": 3,
            "embedding_table_names": ["sparse_embedding0"],
            "embedding_vecsize_per_table": [16],
            "maxnum_catfeature_query_per_table_per_sample": [50],
            "default_value_for_each_table": [1.0],
            "deployed_device_list": [0, 1, 2, 3],
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


def generate_random_samples(num_samples, vocabulary_range_per_slot, max_nnz, dense_dim):
    def generate_sparse_keys(
        num_samples, vocabulary_range_per_slot, max_nnz, key_dtype=args["np_key_type"]
    ):
        slot_num = len(vocabulary_range_per_slot)
        indices = []
        values = []
        for i in range(num_samples):
            for j in range(slot_num):
                vocab_range = vocabulary_range_per_slot[j]
                nnz = np.random.randint(low=1, high=max_nnz + 1)
                entries = sorted(np.random.choice(max_nnz, nnz, replace=False))
                for entry in entries:
                    indices.append([i, j, entry])
                values.extend(
                    np.random.randint(low=vocab_range[0], high=vocab_range[1], size=(nnz,))
                )
        values = np.array(values, dtype=key_dtype)
        return tf.sparse.SparseTensor(
            indices=indices, values=values, dense_shape=(num_samples, slot_num, max_nnz)
        )

    sparse_keys = generate_sparse_keys(num_samples, vocabulary_range_per_slot, max_nnz)
    dense_features = np.random.random((num_samples, dense_dim)).astype(np.float32)
    labels = np.random.randint(low=0, high=2, size=(num_samples, 1))
    return sparse_keys, dense_features, labels


def tf_dataset(sparse_keys, dense_features, labels, batchsize):
    dataset = tf.data.Dataset.from_tensor_slices((sparse_keys, dense_features, labels))
    dataset = dataset.batch(batchsize, drop_remainder=True)
    return dataset


class DNN(tf.keras.models.Model):
    def __init__(
        self, init_tensors, combiner, embed_vec_size, slot_num, max_nnz, dense_dim, **kwargs
    ):
        super(DNN, self).__init__(**kwargs)

        self.combiner = combiner
        self.embed_vec_size = embed_vec_size
        self.slot_num = slot_num
        self.max_nnz = max_nnz
        self.dense_dim = dense_dim
        self.params = tf.Variable(initial_value=tf.concat(init_tensors, axis=0))
        self.fc1 = tf.keras.layers.Dense(units=1024, activation="relu", name="fc1")
        self.fc2 = tf.keras.layers.Dense(units=256, activation="relu", name="fc2")
        self.fc3 = tf.keras.layers.Dense(units=1, activation=None, name="fc3")

    def call(self, inputs, training=True):
        input_cat = inputs[0]
        input_dense = inputs[1]

        # SparseTensor of keys, shape: (batch_size*slot_num, max_nnz)
        embeddings = tf.reshape(
            tf.nn.embedding_lookup_sparse(
                params=self.params, sp_ids=input_cat, sp_weights=None, combiner=self.combiner
            ),
            shape=[-1, self.slot_num * self.embed_vec_size],
        )
        concat_feas = tf.concat([embeddings, input_dense], axis=1)
        logit = self.fc3(self.fc2(self.fc1(concat_feas)))
        return logit, embeddings

    def summary(self):
        inputs = [
            tf.keras.Input(shape=(self.max_nnz,), sparse=True, dtype=args["tf_key_type"]),
            tf.keras.Input(shape=(self.dense_dim,), dtype=tf.float32),
        ]
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()


def train(args):
    init_tensors = np.ones(
        shape=[args["max_vocabulary_size"], args["embed_vec_size"]], dtype=args["np_vector_type"]
    )
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = DNN(
            init_tensors,
            args["combiner"],
            args["embed_vec_size"],
            args["slot_num"],
            args["max_nnz"],
            args["dense_dim"],
        )
        model.summary()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )

    def _replica_loss(labels, logits):
        loss = loss_fn(labels, logits)
        return tf.nn.compute_average_loss(loss, global_batch_size=args["global_batch_size"])

    def _reshape_input(sparse_keys):
        sparse_keys = tf.sparse.reshape(sparse_keys, [-1, sparse_keys.shape[-1]])
        return sparse_keys

    def _train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logit, _ = model(inputs)
            loss = _replica_loss(labels, logit)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return logit, loss

    def _dataset_fn(input_context):
        replica_batch_size = input_context.get_per_replica_batch_size(args["global_batch_size"])
        sparse_keys, dense_features, labels = generate_random_samples(
            args["global_batch_size"] * args["iter_num"],
            args["vocabulary_range_per_slot"],
            args["max_nnz"],
            args["dense_dim"],
        )
        dataset = tf_dataset(sparse_keys, dense_features, labels, replica_batch_size)
        dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
        return dataset

    dataset = strategy.distribute_datasets_from_function(_dataset_fn)
    for i, (sparse_keys, dense_features, labels) in enumerate(dataset):
        sparse_keys = strategy.run(_reshape_input, args=(sparse_keys,))
        inputs = [sparse_keys, dense_features]
        _, loss = strategy.run(_train_step, args=(inputs, labels))
        print("-" * 20, "Step {}, loss: {}".format(i, loss), "-" * 20)
    atexit.register(strategy._extended._collective_ops._pool.close)
    return model


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


class PreTrainedEmbedding(tf.keras.models.Model):
    def __init__(self, combiner, embed_vec_size, slot_num, max_nnz, dense_dim, **kwargs):
        super(PreTrainedEmbedding, self).__init__(**kwargs)

        self.combiner = combiner
        self.embed_vec_size = embed_vec_size
        self.slot_num = slot_num
        self.max_nnz = max_nnz
        self.dense_dim = dense_dim

        self.sparse_lookup_layer = hps.SparseLookupLayer(
            model_name="multi_gpu",
            table_id=0,
            emb_vec_size=self.embed_vec_size,
            emb_vec_dtype=args["tf_vector_type"],
        )
        # Only use one FC layer when leveraging pre-trained embeddings
        self.new_fc = tf.keras.layers.Dense(units=1, activation=None, name="new_fc")

    def call(self, inputs, training=True):
        input_cat = inputs[0]
        input_dense = inputs[1]

        # SparseTensor of keys, shape: (batch_size*slot_num, max_nnz)
        embeddings = tf.reshape(
            self.sparse_lookup_layer(sp_ids=input_cat, sp_weights=None, combiner=self.combiner),
            shape=[-1, self.slot_num * self.embed_vec_size],
        )
        concat_feas = tf.concat([embeddings, input_dense], axis=1)
        logit = self.new_fc(concat_feas)
        return logit, embeddings

    def summary(self):
        inputs = [
            tf.keras.Input(shape=(self.max_nnz,), sparse=True, dtype=args["tf_key_type"]),
            tf.keras.Input(shape=(self.dense_dim,), dtype=tf.float32),
        ]
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()


def train_with_pretrained_embeddings(args):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        hps.Init(global_batch_size=args["global_batch_size"], ps_config_file=args["ps_config_file"])
        model = PreTrainedEmbedding(
            args["combiner"],
            args["embed_vec_size"],
            args["slot_num"],
            args["max_nnz"],
            args["dense_dim"],
        )
        model.summary()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )

    def _replica_loss(labels, logits):
        loss = loss_fn(labels, logits)
        return tf.nn.compute_average_loss(loss, global_batch_size=args["global_batch_size"])

    def _reshape_input(sparse_keys):
        sparse_keys = tf.sparse.reshape(sparse_keys, [-1, sparse_keys.shape[-1]])
        return sparse_keys

    def _train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logit, _ = model(inputs)
            loss = _replica_loss(labels, logit)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return logit, loss

    def _dataset_fn(input_context):
        replica_batch_size = input_context.get_per_replica_batch_size(args["global_batch_size"])
        sparse_keys, dense_features, labels = generate_random_samples(
            args["global_batch_size"] * args["iter_num"],
            args["vocabulary_range_per_slot"],
            args["max_nnz"],
            args["dense_dim"],
        )
        dataset = tf_dataset(sparse_keys, dense_features, labels, replica_batch_size)
        dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
        return dataset

    dataset = strategy.distribute_datasets_from_function(_dataset_fn)
    for i, (sparse_keys, dense_features, labels) in enumerate(dataset):
        sparse_keys = strategy.run(_reshape_input, args=(sparse_keys,))
        inputs = [sparse_keys, dense_features]
        _, loss = strategy.run(_train_step, args=(inputs, labels))
        print("-" * 20, "Step {}, loss: {}".format(i, loss), "-" * 20)
    atexit.register(strategy._extended._collective_ops._pool.close)
    return model


def test_multi_gpu_hps():
    trained_model = train(args)
    weights_list = trained_model.get_weights()
    embedding_weights = weights_list[-1]

    convert_to_sparse_model(embedding_weights, args["embedding_table_path"], args["embed_vec_size"])
    model = train_with_pretrained_embeddings(args)
