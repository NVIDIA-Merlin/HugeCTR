import tensorflow as tf
devices = tf.config.list_physical_devices("GPU")
for dev in devices:
    tf.config.experimental.set_memory_growth(dev, True)

tf.debugging.set_log_device_placement(False)

import sys
sys.path.append("../python")
import hugectr_tf_ops_v2
from read_data import CreateDataset
import pickle
from tensorflow.python.distribute.values import PerReplica

import nvtx.plugins.tf as nvtx_tf
import txt2tfrecord as utils

from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
# tf.config.run_functions_eagerly(False)

import argparse


class PluginSparseModel(tf.keras.models.Model):
    def __init__(self,
                batch_size,
                gpus,
                init_value, 
                name_,
                embedding_type,
                optimizer_type,
                max_vocabulary_size_per_gpu,
                opt_hparams,
                update_type,
                atomic_update,
                scaler,
                slot_num,
                max_nnz,
                max_feature_num,
                embedding_vec_size,
                combiner,
                num_dense_layers,
                input_buffer_reset=False):
        super(PluginSparseModel, self).__init__()

        self.num_dense_layers = num_dense_layers
        self.input_buffer_reset = input_buffer_reset

        self.batch_size = batch_size
        self.slot_num = slot_num
        self.embedding_vec_size = embedding_vec_size
        self.gpus = gpus

        hugectr_tf_ops_v2.init(visible_gpus=gpus, seed=0, key_type='int64', value_type='float',
                                batch_size=batch_size, batch_size_eval=len(gpus))

        self.embedding_name = hugectr_tf_ops_v2.create_embedding(init_value=init_value, 
                                name_=name_, embedding_type=embedding_type, optimizer_type=optimizer_type, 
                                max_vocabulary_size_per_gpu=max_vocabulary_size_per_gpu, opt_hparams=opt_hparams,
                                update_type=update_type, atomic_update=atomic_update, scaler=scaler, slot_num=slot_num,
                                max_nnz=max_nnz, max_feature_num=max_feature_num, embedding_vec_size=embedding_vec_size, 
                                combiner=combiner)

        self.dense_layers = []
        for _ in range(self.num_dense_layers - 1):
            self.dense_layers.append(tf.keras.layers.Dense(units=1024, activation='relu'))

        self.out_layer = tf.keras.layers.Dense(units=1, activation='sigmoid', use_bias=True,
                                                kernel_initializer='glorot_normal', 
                                                bias_initializer='glorot_normal')

    def build(self, _):
        self.bp_trigger = self.add_weight(name='bp_trigger', shape=(1,), dtype=tf.float32, trainable=True)

    @tf.function
    def call(self, row_offset, values, nnz, training=True):
        replica_ctx = tf.distribute.get_replica_context()

        embedding_forward = hugectr_tf_ops_v2.fprop_experimental(self.embedding_name, replica_ctx.replica_id_in_sync_group,
                                             row_offset, values, nnz, self.bp_trigger, is_training=training,
                                             input_buffer_reset=self.input_buffer_reset)
            
        embedding_forward, dense_ctx = nvtx_tf.ops.start(embedding_forward, message='dense_fprop',
                                                         grad_message='dense_bprop')

        hidden = tf.reshape(embedding_forward, [self.batch_size // len(self.gpus), self.slot_num * self.embedding_vec_size])

        for i in range(self.num_dense_layers - 1):
            hidden = self.dense_layers[i](hidden)
        
        logit = self.out_layer(hidden)

        logit = nvtx_tf.ops.end(logit, dense_ctx)

        return logit

    @property
    def get_embedding_name(self):
        return self.embedding_name


class PluginSparseModelV2(tf.keras.models.Model):
    def __init__(self,
                batch_size,
                gpus,
                init_value, 
                name_,
                embedding_type,
                optimizer_type,
                max_vocabulary_size_per_gpu,
                opt_hparams,
                update_type,
                atomic_update,
                scaler,
                slot_num,
                max_nnz,
                max_feature_num,
                embedding_vec_size,
                combiner,
                num_dense_layers,
                input_buffer_reset=False):
        super(PluginSparseModelV2, self).__init__()

        self.num_dense_layers = num_dense_layers
        self.input_buffer_reset = input_buffer_reset

        self.batch_size = batch_size
        self.slot_num = slot_num
        self.embedding_vec_size = embedding_vec_size
        self.gpus = gpus

        hugectr_tf_ops_v2.init(visible_gpus=gpus, seed=0, key_type='int64', value_type='float',
                                batch_size=batch_size, batch_size_eval=len(gpus))

        self.embedding_name = hugectr_tf_ops_v2.create_embedding(init_value=init_value, 
                                name_=name_, embedding_type=embedding_type, optimizer_type=optimizer_type, 
                                max_vocabulary_size_per_gpu=max_vocabulary_size_per_gpu, opt_hparams=opt_hparams,
                                update_type=update_type, atomic_update=atomic_update, scaler=scaler, slot_num=slot_num,
                                max_nnz=max_nnz, max_feature_num=max_feature_num, embedding_vec_size=embedding_vec_size, 
                                combiner=combiner)

        self.dense_layers = []
        for _ in range(self.num_dense_layers - 1):
            self.dense_layers.append(tf.keras.layers.Dense(units=1024, activation='relu'))

        self.out_layer = tf.keras.layers.Dense(units=1, activation='sigmoid', use_bias=True,
                                                kernel_initializer='glorot_normal', 
                                                bias_initializer='glorot_normal')

    def build(self, _):
        self.bp_trigger = self.add_weight(name='bp_trigger', shape=(1,), dtype=tf.float32, trainable=True)

    @tf.function
    def call(self, each_replica, training=True):
        replica_ctx = tf.distribute.get_replica_context()

        embedding_forward = hugectr_tf_ops_v2.fprop(self.embedding_name, replica_ctx.replica_id_in_sync_group,
                                            each_replica, self.bp_trigger, is_training=training)
            
        embedding_forward, dense_ctx = nvtx_tf.ops.start(embedding_forward, message='dense_fprop',
                                                         grad_message='dense_bprop')

        hidden = tf.reshape(embedding_forward, [self.batch_size // len(self.gpus), self.slot_num * self.embedding_vec_size])

        for i in range(self.num_dense_layers - 1):
            hidden = self.dense_layers[i](hidden)
        
        logit = self.out_layer(hidden)

        logit = nvtx_tf.ops.end(logit, dense_ctx)

        return logit

    @property
    def get_embedding_name(self):
        return self.embedding_name


def save_tfrecord_to_python_file(embedding_type, gpu_count, num_batch=50, fprop_version='v1'):
    cols = [utils.idx2key(idx, False) for idx in range(0, utils.NUM_TOTAL_COLUMNS)]
    feature_desc = dict()
    for col in cols:
        if col == 'label' or col.startswith("I"):
            feature_desc[col] = tf.io.FixedLenFeature([], tf.int64) # scaler
        else: 
            feature_desc[col] = tf.io.FixedLenFeature([1], tf.int64) # [slot_num, nnz]


    if fprop_version == "v1":
        dataset = CreateDataset(dataset_names=["./train.tfrecord"],
                                feature_desc=feature_desc,
                                batch_size=65536,
                                n_epochs=1,
                                slot_num=26,
                                max_nnz=1,
                                convert_to_csr=True,
                                gpu_count=gpu_count,
                                embedding_type=embedding_type,
                                get_row_indices=False)

        save_dict = dict()
        for step, datas in enumerate(dataset()):
            if (step >= num_batch):
                break  

            label, dense, others = datas[0], datas[1], datas[2:]

            py_batch_datas = dict()
            py_batch_datas["label"] = label.numpy()
            py_batch_datas['dense'] = dense.numpy()
            sparse = others[0:3]
            py_batch_datas['row_offsets'] = sparse[0].numpy()
            py_batch_datas['value_tensors'] = sparse[1].numpy()
            py_batch_datas['nnz_array'] = sparse[2].numpy()

            save_dict["step_" + str(step)] = py_batch_datas

        save_name = "plugin_v2_" + embedding_type + "_" + str(gpu_count) + "_" + fprop_version
        with open(save_name, "wb") as file:
            pickle.dump(save_dict, file)
    elif fprop_version == "v2":
        dataset = CreateDataset(dataset_names=["./train.tfrecord"],
                                feature_desc=feature_desc,
                                batch_size=65536,
                                n_epochs=1,
                                slot_num=26,
                                max_nnz=1,
                                convert_to_csr=False,
                                gpu_count=gpu_count,
                                embedding_type=embedding_type,
                                get_row_indices=True)

        save_dict = dict()
        for step, datas in enumerate(dataset()):
            if (step >= num_batch):
                break  

            label, dense, others = datas[0], datas[1], datas[2:]

            py_batch_datas = dict()
            py_batch_datas["label"] = label.numpy()
            py_batch_datas['dense'] = dense.numpy()
            sparse = others[0:2]
            py_batch_datas['row_indices'] = sparse[0].numpy()
            py_batch_datas['values'] = sparse[1].numpy()

            save_dict["step_" + str(step)] = py_batch_datas

        save_name = "plugin_v2_" + embedding_type + "_" + str(gpu_count) + "_" + fprop_version
        with open(save_name, "wb") as file:
            pickle.dump(save_dict, file)
    else:
        raise ValueError("fprop_version can only be one of ['v1', 'v2'], but got %s." %fprop_version)

    print("[INFO]: Save %s done." %save_name)

def read_from_py_obj(file_name):
    with open(file_name, "rb") as file:
        obj = pickle.load(file)
    return obj


def plugin_reader(file_name, gpu_count, fprop_version='v1'):
    datas = read_from_py_obj(file_name)

    if fprop_version == 'v1':
        def trans_func(row_offsets, value_tensors, nnz_array, labels):
            row_offsets = tf.cast(row_offsets, dtype=tf.int64)
            value_tensors = tf.cast(value_tensors, dtype=tf.int64)
            nnz_array = tf.cast(nnz_array, dtype=tf.int64)
            labels = tf.cast(labels, dtype=tf.float32)

            return row_offsets, value_tensors, nnz_array, labels

        row_offsets_numpy = [datas[key]['row_offsets'] for key in datas.keys()]
        value_tensors_numpy = [datas[key]['value_tensors'] for key in datas.keys()]
        nnz_array_numpy = [datas[key]['nnz_array'] for key in datas.keys()]
        labels_numpy = [datas[key]['label'] for key in datas.keys()]

        dataset = tf.data.Dataset.from_tensor_slices((row_offsets_numpy, value_tensors_numpy, nnz_array_numpy, labels_numpy))
        dataset = dataset.repeat(1)
        dataset = dataset.map(lambda a, b, c, d: trans_func(a, b, c, d),
                            num_parallel_calls=16,
                            deterministic=False)
        dataset = dataset.prefetch(buffer_size=16)
        # dataset = dataset.apply(tf.data.experimental.prefetch_to_device(device='/gpu:0', buffer_size=16))
    elif fprop_version == "v2":
        def trans_func(row_indices, values, labels):
            row_indices = tf.cast(row_indices, dtype=tf.int64)
            values = tf.cast(values, dtype=tf.int64)
            labels = tf.cast(labels, dtype=tf.float32)

            return row_indices, values, labels

        row_indices_numpy = [datas[key]['row_indices'] for key in datas.keys()]
        values_numpy = [datas[key]['values'] for key in datas.keys()]
        labels_numpy = [datas[key]['label'] for key in datas.keys()]

        dataset = tf.data.Dataset.from_tensor_slices((row_indices_numpy, values_numpy, labels_numpy))
        dataset = dataset.repeat(1)
        dataset = dataset.map(lambda a, b, c: trans_func(a, b, c),
                                num_parallel_calls=16,
                                deterministic=False)
        dataset = dataset.prefetch(buffer_size=16)

    else:
        raise ValueError("fprop_version must be one of ['v1', 'v2'], but got %s" %fprop_version)

    return dataset

def profile_plugin(embedding_type, gpu_count, vocabulary_size=1737710, 
                    slot_num=26, max_nnz=1, batch_size=65536,
                    fprop_version='v1'):
    file_name = "plugin_v2_" + embedding_type + "_" + str(gpu_count) + "_" + fprop_version
    dataset = plugin_reader(file_name, gpu_count, fprop_version)

    # # build model
    strategy = tf.distribute.MirroredStrategy(devices=["/GPU:" + str(i) for i in range(gpu_count)])
    with strategy.scope():
        if fprop_version == 'v1':
            model = PluginSparseModel(batch_size=batch_size, 
                                    gpus=[i for i in range(gpu_count)],
                                    init_value=False, name_='hugectr_embedding', 
                                    embedding_type=embedding_type, optimizer_type='Adam',
                                    max_vocabulary_size_per_gpu=(vocabulary_size // gpu_count) + 1,
                                    opt_hparams=[0.1, 0.9, 0.999, 1e-3],
                                    update_type='Local',
                                    atomic_update=True,
                                    scaler=1.0,
                                    slot_num=slot_num,
                                    max_nnz=max_nnz,
                                    max_feature_num=100,
                                    embedding_vec_size=32,
                                    combiner='sum',
                                    num_dense_layers=7,
                                    input_buffer_reset=False)
        elif fprop_version == 'v2':
            model = PluginSparseModelV2(batch_size=batch_size, 
                                    gpus=[i for i in range(gpu_count)],
                                    init_value=False, name_='hugectr_embedding', 
                                    embedding_type=embedding_type, optimizer_type='Adam',
                                    max_vocabulary_size_per_gpu=(vocabulary_size // gpu_count) + 1,
                                    opt_hparams=[0.1, 0.9, 0.999, 1e-3],
                                    update_type='Local',
                                    atomic_update=True,
                                    scaler=1.0,
                                    slot_num=slot_num,
                                    max_nnz=max_nnz,
                                    max_feature_num=100,
                                    embedding_vec_size=32,
                                    combiner='sum',
                                    num_dense_layers=7,
                                    input_buffer_reset=False)
        dense_opt = tf.keras.optimizers.SGD()

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    def _replica_loss(labels, logits):
        loss_v = loss_fn(labels, logits)
        return tf.nn.compute_average_loss(loss_v, global_batch_size=batch_size)

    if fprop_version == 'v1':
        @tf.function
        def _train_step(row_offset, values, nnz, label):
            with tf.GradientTape() as tape:
                label = tf.expand_dims(label, axis=1)
                logit = model(row_offset, values, nnz)
                replica_loss = _replica_loss(label, logit)

            replica_grads = tape.gradient(replica_loss, model.trainable_weights)
            dense_opt.apply_gradients(zip(replica_grads, model.trainable_weights))
            return replica_loss
    elif fprop_version == 'v2':
        @tf.function
        def _train_step(each_replica, label):
            with tf.GradientTape() as tape:
                label = tf.expand_dims(label, axis=1)
                logit = model(each_replica)
                replica_loss = _replica_loss(label, logit)

            replica_grads = tape.gradient(replica_loss, model.trainable_weights)
            dense_opt.apply_gradients(zip(replica_grads, model.trainable_weights))
            return replica_loss

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    dataset = dataset.with_options(options)

    input_options = tf.distribute.InputOptions(
        experimental_prefetch_to_device=True, # TODO: not working..
        experimental_replication_mode=tf.distribute.InputReplicationMode.PER_WORKER,
        experimental_place_dataset_on_device=False
    )

    if fprop_version == "v1":
        dataset = strategy.experimental_distribute_dataset(dataset, input_options)

        for step, (row_offsets, value_tensors, nnz_array, labels) in enumerate(dataset):
            step, step_ctx = nvtx_tf.ops.start(tf.convert_to_tensor(step, dtype=tf.int32),
                                                message='Iteration_' + str(step))

            replica_loss = strategy.run(_train_step, args=(row_offsets, value_tensors, nnz_array, labels))
            total_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, replica_loss, axis=None)

            tf.print("step:%d, loss:%.5f" %(step, total_loss))

            step = nvtx_tf.ops.end(step, step_ctx)

    elif fprop_version == 'v2':
        for step, (row_indices, values, labels) in enumerate(dataset):
            step, step_ctx = nvtx_tf.ops.start(tf.convert_to_tensor(step, dtype=tf.int32),
                                                message='Iteration_' + str(step))

            to_each_replicas = hugectr_tf_ops_v2.broadcast_then_convert_to_csr(
                        model.get_embedding_name, row_indices, values, T = [tf.int32] * gpu_count)
            to_each_replicas = PerReplica(to_each_replicas)
            labels = tf.split(labels, num_or_size_splits=gpu_count)
            labels = PerReplica(labels)

            replica_loss = strategy.run(_train_step, args=(to_each_replicas, labels))
            total_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, replica_loss, axis=None)

            tf.print("step:%d, loss:%.5f" %(step, total_loss))
            
            step = nvtx_tf.ops.end(step, step_ctx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='profiling_v2')

    parser.add_argument('--prepare_datas', type=int, help='whether to prepare datasets.', required=False, 
                        default=0, choices=[0, 1])
    parser.add_argument('--embedding_type', type=str, required=True, choices=['distributed', 'localized'])
    parser.add_argument('--gpu_count', type=int, required=True, choices=[1, 2, 4, 8, 16])
    parser.add_argument('--fprop_version', type=str, required=True, choices=['v1', 'v2'])

    args = parser.parse_args()

    if (1 == args.prepare_datas):
        for count in [1, 2, 4, 8]:
            save_tfrecord_to_python_file(embedding_type=args.embedding_type, gpu_count=count, 
                                        fprop_version=args.fprop_version)
    else:
        profile_plugin(embedding_type=args.embedding_type, gpu_count=args.gpu_count,
                        fprop_version=args.fprop_version)
        print("-" * 30, "Done.", "*"*30)