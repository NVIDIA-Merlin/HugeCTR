import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
import struct
import json

args = dict()

args["slot_num"] = 26  # the number of feature fields in this embedding layer
args["iter_num"] = 10  # the number of feature fields in this embedding layer
args["embed_vec_sizes"] = [16] * args["slot_num"]  # the dimension of embedding vectors
args["dense_dim"] = 13  # the dimension of dense features
args["global_batch_size"] = 1024  # the globally batchsize for all GPUs
args["table_names"] = ["table" + str(i) for i in range(args["slot_num"])]  # embedding table names
args["max_vocabulary_sizes"] = np.random.randint(1000, 1200, size=args["slot_num"]).tolist()
args["max_nnz"] = [1] * args["slot_num"]

args["ps_config_file"] = "dlrm.json"
args["dense_model_path"] = "dlrm_dense.model"
args["sparse_model_path"] = "dlrm_sparse.model"
args["saved_path"] = "dlrm_tf_saved_model"
args["np_key_type"] = np.int64
args["np_vector_type"] = np.float32
args["tf_key_type"] = tf.int64
args["tf_vector_type"] = tf.float32

# prepare
data = {
    "supportlonglong": True,
    "models": [
        {
            "model": "dlrm",
            "sparse_files": ["dlrm_sparse.model/table" + str(i) for i in range(args["slot_num"])],
            "num_of_worker_buffer_in_pool": 30,
            "embedding_table_names": ["table" + str(i) for i in range(args["slot_num"])],
            "embedding_vecsize_per_table": [16] * args["slot_num"],
            "maxnum_catfeature_query_per_table_per_sample": [10] * args["slot_num"],
            "default_value_for_each_table": [1.0] * args["slot_num"],
            "deployed_device_list": [0],
            "max_batch_size": 1024,
            "cache_refresh_percentage_per_iteration": 0.2,
            "hit_rate_threshold": 1.0,
            "gpucacheper": 1.0,
            "gpucache": True,
        }
    ],
}

with open("dlrm.json", "w") as json_file:
    json.dump(data, json_file, indent=4)

import hierarchical_parameter_server as hps


def generate_random_samples(num_samples, vocabulary_range_per_slot, max_nnz, dense_dim):
    def generate_sparse_keys(
        num_samples, vocabulary_range_per_slot, max_nnz, key_dtype=args["np_key_type"]
    ):
        slot_num = len(vocabulary_range_per_slot)
        total_indices = []
        for i in range(slot_num):
            indices = []
            values = []
            for j in range(num_samples):
                vocab_range = vocabulary_range_per_slot[i]
                nnz = np.random.randint(low=1, high=max_nnz + 1)
                entries = sorted(np.random.choice(max_nnz, nnz, replace=False))
                for entry in entries:
                    indices.append([j, 0, entry])
                values.extend(np.random.randint(low=0, high=vocab_range, size=(nnz,)))
            values = np.array(values, dtype=key_dtype)
            total_indices.append(
                tf.sparse.SparseTensor(
                    indices=indices, values=values, dense_shape=(num_samples, 1, max_nnz)
                )
            )
        return total_indices

    sparse_keys = generate_sparse_keys(num_samples, vocabulary_range_per_slot, max_nnz)
    dense_features = np.random.random((num_samples, dense_dim)).astype(np.float32)
    labels = np.random.randint(low=0, high=2, size=(num_samples, 1))
    return sparse_keys, dense_features, labels


def tf_dataset(sparse_keys, dense_features, labels, batchsize):
    total_data = []
    # total_data.extend(sparse_keys)
    total_data.extend(sparse_keys)
    total_data.append(dense_features)
    total_data.append(labels)
    dataset = tf.data.Dataset.from_tensor_slices(tuple(total_data))
    dataset = dataset.batch(batchsize, drop_remainder=True)
    return dataset


class InferenceModel(tf.keras.models.Model):
    def __init__(self, slot_num, embed_vec_size, dense_model_path, **kwargs):
        super(InferenceModel, self).__init__(**kwargs)

        self.slot_num = slot_num
        self.embed_vec_size = embed_vec_size

        self.sparse_lookup_layer_list = []
        for i in range(self.slot_num):
            self.sparse_lookup_layer_list.append(
                hps.SparseLookupLayer(
                    model_name="dlrm",
                    table_id=i,
                    emb_vec_size=self.embed_vec_size[i],
                    emb_vec_dtype=args["tf_vector_type"],
                )
            )
        self.reshape_layer_list = []
        for i in range(self.slot_num):
            self.reshape_layer_list.append(
                tf.keras.layers.Reshape((1, args["embed_vec_sizes"][i]), name="reshape" + str(i))
            )
        self.concat1 = tf.keras.layers.Concatenate(axis=1, name="concat1")

        self.dense_model = tf.keras.models.load_model(dense_model_path)
        self.dense_model.summary()

    def call(self, inputs):
        # input_sparse = inputs[:]
        input_sparse = inputs[:-1]
        input_dense = inputs[-1]
        embeddings = []
        for i in range(self.slot_num):
            tmp_embedding = self.sparse_lookup_layer_list[i](
                sp_ids=input_sparse[i], sp_weights=None, combiner="mean"
            )
            embeddings.append(tmp_embedding)

        concat_embeddings = []
        for i in range(args["slot_num"]):
            concat_embeddings.append(self.reshape_layer_list[i](embeddings[i]))
        concat_embeddings = self.concat1(concat_embeddings)

        logit = self.dense_model([concat_embeddings, input_dense])
        return logit, embeddings

    def summary(self):
        inputs = []
        for i in range(args["slot_num"]):
            inputs.append(
                tf.keras.Input(shape=(args["max_nnz"][i],), sparse=True, dtype=args["tf_key_type"])
            )
        dense_input = tf.keras.Input(shape=(args["dense_dim"],), dtype=tf.float32)
        inputs.append(dense_input)
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()

        return model.summary()


def create_and_save_inference_graph(args):
    model = InferenceModel(args["slot_num"], args["embed_vec_sizes"], args["dense_model_path"])
    model.summary()
    inputs = []
    for i in range(args["slot_num"]):
        inputs.append(
            tf.keras.Input(shape=(args["max_nnz"][i],), sparse=True, dtype=args["tf_key_type"])
        )
    dense_input = tf.keras.Input(shape=(args["dense_dim"],), dtype=tf.float32)
    inputs.append(dense_input)
    _ = model(inputs)
    model.save(args["saved_path"])


create_and_save_inference_graph(args)


def inference_with_saved_model(args):
    hps.Init(global_batch_size=args["global_batch_size"], ps_config_file=args["ps_config_file"])
    model = tf.keras.models.load_model(args["saved_path"])
    model.summary()

    def _infer_step(tmp_inputs, labels):
        logit, embeddings = model(tmp_inputs)
        return logit, embeddings

    embeddings_peek = list()
    inputs_peek = list()

    sparse_keys, dense_features, labels = generate_random_samples(
        args["global_batch_size"] * args["iter_num"],
        args["max_vocabulary_sizes"],
        args["max_nnz"][0],
        args["dense_dim"],
    )
    dataset = tf_dataset(sparse_keys, dense_features, labels, args["global_batch_size"])
    for i, input_pack in enumerate(dataset):
        inputs = []
        for table_id in range(args["slot_num"]):
            inputs.append(
                tf.sparse.reshape(input_pack[table_id], [-1, input_pack[table_id].shape[-1]])
            )
        inputs.append(input_pack[-2])
        labels = input_pack[-1]
        logit, embeddings = _infer_step(inputs, labels)
        embeddings_peek.append(embeddings)
        inputs_peek.append(inputs)
        print("-" * 20, "Step {}".format(i), "-" * 20)

    return embeddings_peek, inputs_peek


embeddings_peek, inputs_peek = inference_with_saved_model(args)

# embedding table, input keys are SparseTensor
print(inputs_peek[-1][0].values)
print(embeddings_peek[0][0])
