import os
import numpy as np
import tensorflow as tf
import struct
import json
import hierarchical_parameter_server as hps

args = dict()
args["gpu_num"] = 1
args["slot_num"] = 26
args["embed_vec_size"] = 128
args["dense_dim"] = 13
args["global_batch_size"] = 131072
args["max_vocabulary_size"] = 32709138

args["dlrm_saved_path"] = "dlrm_tf_saved_model"
args["hps_plugin_dlrm_saved_path"] = "hps_plugin_dlrm_tf_saved_model"
args["dlrm_dense_saved_path"] = "dlrm_dense_tf_saved_model"
args["dlrm_embedding_table_saved_path"] = "dlrm_embedding_table"
args["ps_config_file"] = "dlrm.json"
args["np_vector_type"] = np.float32
args["tf_key_type"] = tf.int32
args["tf_vector_type"] = tf.float32

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args["gpu_num"])))
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

hps_config = {
    "supportlonglong": False,
    "models": [
        {
            "model": "dlrm",
            "sparse_files": [args["dlrm_embedding_table_saved_path"]],
            "num_of_worker_buffer_in_pool": 3,
            "embedding_table_names": ["sparse_embedding0"],
            "embedding_vecsize_per_table": [128],
            "maxnum_catfeature_query_per_table_per_sample": [26],
            "default_value_for_each_table": [1.0],
            "deployed_device_list": [0],
            "max_batch_size": args["global_batch_size"],
            "cache_refresh_percentage_per_iteration": 0.0,
            "hit_rate_threshold": 1.0,
            "gpucacheper": 0.2,
            "gpucache": True,
        }
    ],
}
hps_config_json_object = json.dumps(hps_config, indent=4)
with open(args["ps_config_file"], "w") as outfile:
    outfile.write(hps_config_json_object)


class MLP(tf.keras.layers.Layer):
    def __init__(self, arch, activation="relu", out_activation=None, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.layers = []
        index = 0
        for units in arch[:-1]:
            self.layers.append(
                tf.keras.layers.Dense(
                    units, activation=activation, name="{}_{}".format(kwargs["name"], index)
                )
            )
            index += 1
        self.layers.append(
            tf.keras.layers.Dense(
                arch[-1], activation=out_activation, name="{}_{}".format(kwargs["name"], index)
            )
        )

    def call(self, inputs, training=True):
        x = self.layers[0](inputs)
        for layer in self.layers[1:]:
            x = layer(x)
        return x


class SecondOrderFeatureInteraction(tf.keras.layers.Layer):
    def __init__(self):
        super(SecondOrderFeatureInteraction, self).__init__()

    def call(self, inputs, num_feas):
        dot_products = tf.reshape(
            tf.matmul(inputs, inputs, transpose_b=True), (-1, num_feas * num_feas)
        )
        indices = tf.constant([i * num_feas + j for j in range(1, num_feas) for i in range(j)])
        flat_interactions = tf.gather(dot_products, indices, axis=1)
        return flat_interactions


class DLRM(tf.keras.models.Model):
    def __init__(
        self, init_tensors, embed_vec_size, slot_num, dense_dim, arch_bot, arch_top, **kwargs
    ):
        super(DLRM, self).__init__(**kwargs)

        self.init_tensors = init_tensors

        with tf.device("/CPU:0"):
            self.params = tf.Variable(
                initial_value=tf.concat(self.init_tensors, axis=0), name="cpu_embedding"
            )

        self.embed_vec_size = embed_vec_size
        self.slot_num = slot_num
        self.dense_dim = dense_dim

        self.bot_nn = MLP(arch_bot, name="bottom", out_activation="relu")
        self.top_nn = MLP(arch_top, name="top", out_activation="sigmoid")
        self.interaction_op = SecondOrderFeatureInteraction()

        self.interaction_out_dim = self.slot_num * (self.slot_num + 1) // 2
        self.reshape_layer1 = tf.keras.layers.Reshape((1, arch_bot[-1]), name="reshape1")
        self.concat1 = tf.keras.layers.Concatenate(axis=1, name="concat1")
        self.concat2 = tf.keras.layers.Concatenate(axis=1, name="concat2")

    def call(self, inputs, training=True):
        categorical_features = inputs["categorical_features"]
        numerical_features = inputs["numerical_features"]

        embedding_vector = tf.nn.embedding_lookup(params=self.params, ids=categorical_features)
        dense_x = self.bot_nn(numerical_features)
        concat_features = self.concat1([embedding_vector, self.reshape_layer1(dense_x)])

        Z = self.interaction_op(concat_features, self.slot_num + 1)
        z = self.concat2([dense_x, Z])
        logit = self.top_nn(z)
        return logit

    def summary(self):
        inputs = {
            "categorical_features": tf.keras.Input(
                shape=(self.slot_num,), dtype=args["tf_key_type"], name="categorical_features"
            ),
            "numerical_features": tf.keras.Input(
                shape=(self.dense_dim,), dtype=tf.float32, name="numrical_features"
            ),
        }
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()


class HPS_Plugin_DLRM(tf.keras.models.Model):
    def __init__(self, slot_num, dense_dim, embed_vec_size, dense_model_path, **kwargs):
        super(HPS_Plugin_DLRM, self).__init__(**kwargs)

        self.slot_num = slot_num
        self.dense_dim = dense_dim
        self.embed_vec_size = embed_vec_size
        self.lookup_layer = hps.LookupLayer(
            model_name="dlrm",
            table_id=0,
            emb_vec_size=self.embed_vec_size,
            emb_vec_dtype=args["tf_vector_type"],
            name="lookup",
            ps_config_file="dlrm.json",
            global_batch_size=args["global_batch_size"],
        )
        self.dense_model = tf.keras.models.load_model(dense_model_path)

    def call(self, inputs):
        categorical_features = inputs["categorical_features"]
        numerical_features = inputs["numerical_features"]

        embedding_vector = self.lookup_layer(categorical_features)
        logit = self.dense_model([embedding_vector, numerical_features])
        return logit

    def summary(self):
        inputs = {
            "categorical_features": tf.keras.Input(
                shape=(self.slot_num,), dtype=args["tf_key_type"], name="categorical_features"
            ),
            "numerical_features": tf.keras.Input(
                shape=(self.dense_dim,), dtype=tf.float32, name="numerical_features"
            ),
        }
        model = tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()


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


def native_tf(args):
    init_tensors = np.random.random((args["max_vocabulary_size"], args["embed_vec_size"])).astype(
        args["np_vector_type"]
    )
    model = DLRM(
        init_tensors,
        args["embed_vec_size"],
        args["slot_num"],
        args["dense_dim"],
        arch_bot=[512, 256, args["embed_vec_size"]],
        arch_top=[1024, 1024, 512, 256, 1],
        name="dlrm",
    )
    model.summary()
    inputs = {
        "categorical_features": tf.keras.Input(
            shape=(args["slot_num"],), dtype=args["tf_key_type"], name="categorical_features"
        ),
        "numerical_features": tf.keras.Input(
            shape=(args["dense_dim"],), dtype=tf.float32, name="numerical_features"
        ),
    }
    pred = model(inputs)
    model.save(
        args["dlrm_saved_path"],
        options=tf.saved_model.SaveOptions(
            experimental_variable_policy=tf.saved_model.experimental.VariablePolicy.SAVE_VARIABLE_DEVICES
        ),
    )

    dense_model = tf.keras.models.Model(
        [model.get_layer("concat1").input[0], model.get_layer("bottom").input],
        model.get_layer("top").output,
    )
    dense_model.summary()
    dense_model.save(args["dlrm_dense_saved_path"])

    embedding_weights = model.get_weights()[-1]
    convert_to_sparse_model(
        embedding_weights, args["dlrm_embedding_table_saved_path"], args["embed_vec_size"]
    )


def hps_plugin_tf(args):
    model = HPS_Plugin_DLRM(
        args["slot_num"], args["dense_dim"], args["embed_vec_size"], args["dlrm_dense_saved_path"]
    )
    model.summary()
    inputs = {
        "categorical_features": tf.keras.Input(
            shape=(args["slot_num"],), dtype=args["tf_key_type"], name="categorical_features"
        ),
        "numerical_features": tf.keras.Input(
            shape=(args["dense_dim"],), dtype=tf.float32, name="numerical_features"
        ),
    }
    pred = model(inputs)
    model.save(args["hps_plugin_dlrm_saved_path"])


if __name__ == "__main__":
    native_tf(args)
    hps_plugin_tf(args)
