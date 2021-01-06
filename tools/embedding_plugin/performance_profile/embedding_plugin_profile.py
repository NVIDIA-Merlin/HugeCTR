import tensorflow as tf
devices = tf.config.list_physical_devices("GPU")
for dev in devices:
    tf.config.experimental.set_memory_growth(dev, True)

import sys
sys.path.append("../python")
import hugectr_tf_ops
import pickle
from model import OriginalEmbedding, PluginEmbedding
from tensorflow.python.distribute.values import PerReplica
import nvtx.plugins.tf as nvtx_tf

import argparse

# --------------------------- prepare datas ------------------------------------------ #

def save_dataset_to_python_obj(batch_size, num_batch, save_name, gpu_count, convert_to_csr=True, embedding_type='distributed', 
                                get_row_indices=False):
    """
    this function will save num_batch * batch_size samples to python obj.
    so that it can be load into CPU memory rather than read from tfrecord.
    """
    import txt2tfrecord as utils
    from read_data import  CreateDataset

    cols = [utils.idx2key(idx, False) for idx in range(0, utils.NUM_TOTAL_COLUMNS)]
    feature_desc = dict()
    for col in cols:
        if col == 'label' or col.startswith("I"):
            feature_desc[col] = tf.io.FixedLenFeature([], tf.int64) # scaler
        else: 
            feature_desc[col] = tf.io.FixedLenFeature([1], tf.int64) # [slot_num, nnz]

    dataset_names = ["train.tfrecord"]
    dataset = CreateDataset(dataset_names=dataset_names,
                            feature_desc=feature_desc,
                            batch_size=batch_size, 
                            n_epochs=1,
                            slot_num=26,
                            max_nnz=1,
                            convert_to_csr=tf.constant(convert_to_csr, dtype=tf.bool),
                            gpu_count=gpu_count,
                            embedding_type=embedding_type,
                            get_row_indices=get_row_indices)()

    # read datas into python dict
    save_dict = dict()
    for step, datas in enumerate(dataset):
        if (step >= num_batch):
            break

        py_batch_datas = dict()
        label, dense, others = datas[0], datas[1], datas[2:]
        py_batch_datas["label"] = label.numpy()
        py_batch_datas["dense"] = dense.numpy()
        if (convert_to_csr):
            sparse = others[0:3]
            py_batch_datas["row_offsets"] = sparse[0].numpy()
            py_batch_datas["values"] = sparse[1].numpy()
            py_batch_datas["nnz_array"] = sparse[2].numpy()
        else:
            if get_row_indices:
                sparse = others[0:2]
                py_batch_datas['row_indices'] = sparse[0].numpy()
                py_batch_datas['values'] = sparse[1].numpy()
            else:
                sparse = others[-1]
                py_batch_datas["indices"] = sparse.indices.numpy()
                py_batch_datas["values"] = sparse.values.numpy()
                py_batch_datas["dense_shape"] = sparse.dense_shape.numpy()

        save_dict["step_" + str(step)] = py_batch_datas


    if (convert_to_csr or get_row_indices):
        file_name = save_name + "_" + embedding_type + "_" + str(gpu_count)
    else:
        file_name = save_name + "_" + str(gpu_count)

    # save dict into file
    with open(file_name, 'wb') as file:
        pickle.dump(save_dict, file)
    print("Save done %s." %file_name)


def read_from_py_obj(file_name):
    with open(file_name, 'rb') as file:
        obj = pickle.load(file)
    return obj


# --------------------------- define models ---------------------------------------------- #
class DenseModel(tf.keras.models.Model):
    def __init__(self, num_layers):
        super(DenseModel, self).__init__()
        self.num_layers = num_layers

        self.dense_layers = []
        for _ in range(num_layers - 1):
            self.dense_layers.append(tf.keras.layers.Dense(units=1024, activation='relu'))

        self.out_layer = tf.keras.layers.Dense(units=1, activation='sigmoid', use_bias=True,
                                                kernel_initializer='glorot_normal', 
                                                bias_initializer='glorot_normal')

    @tf.function
    def call(self, inputs, training=True):
        hidden = tf.reshape(inputs, [tf.shape(inputs)[0], 26 * 32]) # [batchsize, slot_num * embedding_vec_size]

        for i in range(self.num_layers - 1):
            hidden = self.dense_layers[i](hidden)

        result = self.out_layer(hidden)
        return result

class PluginSparseModel(tf.keras.models.Model):
    def __init__(self, gpus, batch_size, embedding_type='distributed', fprop_version='v3'):
        super(PluginSparseModel, self).__init__()

        hugectr_tf_ops.init(visiable_gpus=gpus, seed=123, key_type='int64', value_type='float',
                     batch_size=batch_size, batch_size_eval=len(gpus))

        self.embedding_layer = PluginEmbedding(vocabulary_size=1737710,
                                               slot_num=26,
                                               embedding_vec_size=32,
                                               gpu_count=len(gpus),
                                               initializer=False,
                                               name='plugin_embedding',
                                               embedding_type=embedding_type,
                                               optimizer='Adam',
                                               opt_hparam=[0.1, 0.9, 0.99, 1e-3],
                                               update_type='LazyGlobal',
                                               combiner='sum',
                                               fprop_version=fprop_version)

        if (fprop_version == 'v3'):
            self.call_func = self.call_v3
        elif (fprop_version == 'v4'):
            self.call_func = self.call_v4
                
    @tf.function
    def call_v3(self, input_placeholder, row_offsets, value_tensors, nnz_array, output_shape, training=True):
        return self.embedding_layer(input_placeholder=input_placeholder, row_offsets=row_offsets, value_tensors=value_tensors, 
                                    nnz_array=nnz_array, output_shape=output_shape, training=training)

    @tf.function
    def call_v4(self, input_placeholder, row_indices, values, output_shape, training=True):
        return self.embedding_layer(input_placeholder=input_placeholder, row_indices=row_indices, values=values, 
                                    output_shape=output_shape, training=training)

    @tf.function
    def call(self, input_placeholder, **kwargs):
        return self.call_func(input_placeholder, **kwargs)
        


class OriginSparseModel(tf.keras.models.Model):
    def __init__(self, gpus):
        super(OriginSparseModel, self).__init__()

        self.embedding_layer = OriginalEmbedding(vocabulary_size=1737710,
                                                 embedding_vec_size=32,
                                                 initializer='uniform',
                                                 combiner='sum',
                                                 gpus=gpus)

    @tf.function
    def call(self, indices, values, dense_shape, output_shape):
        sparse_tensor = tf.sparse.SparseTensor(indices, values, dense_shape)
        return self.embedding_layer(sparse_tensor, output_shape)

# --------------- data reader pipeline ------------------------------------------------ #
def plugin_reader(file_name, fprop_version='v3'):
    datas = read_from_py_obj(file_name)

    if fprop_version == 'v3':
        row_offsets_numpy = [datas[key]['row_offsets'] for key in datas.keys()]
        value_tensors_numpy = [datas[key]['values'] for key in datas.keys()]
        nnz_array_numpy = [datas[key]['nnz_array'] for key in datas.keys()]
        labels_numpy = [datas[key]['label'] for key in datas.keys()]
        dataset = tf.data.Dataset.from_tensor_slices((row_offsets_numpy, value_tensors_numpy, nnz_array_numpy, labels_numpy))
        dataset = dataset.repeat(1)
        dataset = dataset.map(lambda a, b, c, d: (tf.cast(a, dtype=tf.int64), 
                                                tf.cast(b, dtype=tf.int64),
                                                tf.cast(c, dtype=tf.int64),
                                                tf.cast(d, dtype=tf.float32)),
                            num_parallel_calls=16,
                            deterministic=False)
        dataset = dataset.prefetch(buffer_size=16)
        # dataset = dataset.apply(tf.data.experimental.prefetch_to_device(device='/gpu:0', buffer_size=16))
        return dataset
    elif fprop_version == 'v4':
        row_indices_numpy = [datas[key]['row_indices'] for key in datas.keys()]
        values_numpy = [datas[key]['values'] for key in datas.keys()]
        labels_numpy = [datas[key]['label'] for key in datas.keys()]

        dataset = tf.data.Dataset.from_tensor_slices((row_indices_numpy, values_numpy, labels_numpy))
        dataset = dataset.repeat(1)
        dataset = dataset.map(lambda a, b, c: (tf.cast(a, dtype=tf.int64), 
                                                tf.cast(b, dtype=tf.int64),
                                                tf.cast(c, dtype=tf.float32)),
                            num_parallel_calls=16,
                            deterministic=False)
        dataset = dataset.prefetch(buffer_size=16)
        # dataset = dataset.apply(tf.data.experimental.prefetch_to_device(device='/gpu:0', buffer_size=16))
        return dataset
        

def origin_reader(file_name):
    datas = read_from_py_obj(file_name)
    indices_numpy = [datas[key]['indices'] for key in datas.keys()]
    values_numpy = [datas[key]['values'] for key in datas.keys()]
    dense_shape_numpy = [datas[key]['dense_shape'] for key in datas.keys()]
    labels_numpy = [datas[key]['label'] for key in datas.keys()]

    dataset = tf.data.Dataset.from_tensor_slices((indices_numpy, values_numpy, dense_shape_numpy, labels_numpy))
    dataset = dataset.repeat(1)
    dataset = dataset.map(lambda a, b, c, d: (tf.cast(a, dtype=tf.int64),
                                              tf.cast(b, dtype=tf.int64),
                                              tf.cast(c, dtype=tf.int64),
                                              tf.cast(d, dtype=tf.float32)),
                        num_parallel_calls=16,
                        deterministic=False)
    dataset = dataset.prefetch(buffer_size=16)
    # dataset = dataset.apply(tf.data.experimental.prefetch_to_device(device='/gpu:0', buffer_size=16))
    return dataset

# ------------------------ train loop ----------------------------------------------------- #
def profile_plugin(gpus, num_layers, batch_size, embedding_type='distributed', fprop_version='v3'):
    # load data first

    dataset = plugin_reader("./plugin" + "_" + embedding_type + "_" + str(len(gpus)), fprop_version=fprop_version)  

    # build model
    sparse_model = PluginSparseModel(gpus, batch_size, embedding_type=embedding_type, fprop_version=fprop_version)
    sparse_opt = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-3)

    if len(gpus) > 1:
        distribute_strategy = tf.distribute.MirroredStrategy(devices=['/gpu:' + str(i) for i in gpus])
        with distribute_strategy.scope():
            dense_model = DenseModel(num_layers)
            dense_opt = tf.keras.optimizers.SGD()
    else:
        dense_model = DenseModel(num_layers)
        dense_opt = tf.keras.optimizers.SGD()

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    def _dense_loss(labels, logits):
        loss_v = loss_fn(labels, logits)
        return tf.nn.compute_average_loss(loss_v, global_batch_size=batch_size)

    def dense_train_step(dense_inputs, dense_labels):
        with tf.GradientTape() as tape:
            tape.watch(dense_inputs)
            
            dense_result = dense_model(dense_inputs)
            dense_loss = _dense_loss(dense_labels, dense_result)
        dense_grads, input_grads = tape.gradient(dense_loss, [dense_model.trainable_weights, dense_inputs])
        dense_opt.apply_gradients(zip(dense_grads, dense_model.trainable_weights))
        return dense_loss, input_grads

    if fprop_version == 'v3':
        if len(gpus) > 1:
            @tf.function
            def total_train_step(row_offsets, value_tensors, nnz_array, labels):
                with tf.GradientTape() as tape:
                    row_offsets, emb_ctx = nvtx_tf.ops.start(row_offsets, 
                                                            message='emb_fprop',
                                                            domain_name='forward')

                    embedding_result = sparse_model(0, row_offsets=row_offsets, 
                                                    value_tensors=value_tensors, 
                                                    nnz_array=nnz_array, 
                                                    output_shape = [-1, 26, 32], # [batch_size, slot_num, embedding_vec_size] 
                                                    training=True)

                    embedding_result = nvtx_tf.ops.end(embedding_result, emb_ctx)

                    embedding_result, dense_ctx = nvtx_tf.ops.start(embedding_result,
                                                                    message='dense_fprop',
                                                                    domain_name='forward')

                    labels = tf.expand_dims(labels, axis=1)
                    dense_inputs = tf.split(embedding_result, num_or_size_splits=len(gpus))
                    dense_labels = tf.split(labels, num_or_size_splits=len(gpus))
                    dense_inputs_replicas = PerReplica(dense_inputs)
                    dense_labels_replicas = PerReplica(dense_labels)

                    dense_losses, input_grads = distribute_strategy.run(dense_train_step,
                                                                        args=(dense_inputs_replicas, dense_labels_replicas))

                    all_grads = tf.concat(input_grads.values, axis=0)

                    all_grads = nvtx_tf.ops.end(all_grads, dense_ctx)

                embedding_grads = tape.gradient(embedding_result, sparse_model.trainable_weights, output_gradients=all_grads)
                sparse_opt.apply_gradients(zip(embedding_grads, sparse_model.trainable_weights))
                return distribute_strategy.reduce(tf.distribute.ReduceOp.SUM, dense_losses, axis=None) 
        else:
            @tf.function
            def total_train_step(row_offsets, value_tensors, nnz_array, labels):
                with tf.GradientTape() as tape:
                    row_offsets, emb_ctx = nvtx_tf.ops.start(row_offsets, 
                                                            message='emb_fprop',
                                                            domain_name='forward')

                    embedding_result = sparse_model(0, row_offsets=row_offsets, 
                                                    value_tensors=value_tensors, 
                                                    nnz_array=nnz_array,
                                                    output_shape=[-1, 26, 32],
                                                    training=True)

                    embedding_result = nvtx_tf.ops.end(embedding_result, emb_ctx)

                    embedding_result, dense_ctx = nvtx_tf.ops.start(embedding_result,
                                                                    message='dense_fprop',
                                                                    domain_name='forward')

                    labels = tf.expand_dims(labels, axis=1)

                    dense_loss, embedding_top_grad = dense_train_step(embedding_result, labels)

                    embedding_top_grad = nvtx_tf.ops.end(embedding_top_grad, dense_ctx)

                grads = tape.gradient(embedding_result, sparse_model.trainable_weights, output_gradients=embedding_top_grad)
                sparse_opt.apply_gradients(zip(grads, sparse_model.trainable_weights))
                return dense_loss

        # train loop
        for step, (row_offsets, value_tensors, nnz_array, labels) in enumerate(dataset):
            step, nvtx_context = nvtx_tf.ops.start(tf.convert_to_tensor(step, dtype=tf.int32), 
                                                message='Iteration_' + str(step), 
                                                domain_name='forward',
                                                grad_message='Back_Iteration_' + str(step),
                                                grad_domain_name='backward')
            total_loss = total_train_step(row_offsets,
                                        value_tensors,
                                        nnz_array,
                                        labels)
            tf.print("step:%d, loss:%.5f" %(step, total_loss))
            step = nvtx_tf.ops.end(step, nvtx_context)

    elif fprop_version == 'v4':
        if len(gpus) > 1:
            @tf.function
            def total_train_step(row_indices, values, labels):
                with tf.GradientTape() as tape:
                    row_indices, emb_ctx = nvtx_tf.ops.start(row_indices, 
                                                            message='emb_fprop',
                                                            domain_name='forward')

                    embedding_result = sparse_model(0, row_indices=row_indices, 
                                                    values=values, 
                                                    output_shape = [-1, 26, 32], # [batch_size, slot_num, embedding_vec_size] 
                                                    training=True)

                    embedding_result = nvtx_tf.ops.end(embedding_result, emb_ctx)

                    embedding_result, dense_ctx = nvtx_tf.ops.start(embedding_result,
                                                                    message='dense_fprop',
                                                                    domain_name='forward')

                    labels = tf.expand_dims(labels, axis=1)
                    dense_inputs = tf.split(embedding_result, num_or_size_splits=len(gpus))
                    dense_labels = tf.split(labels, num_or_size_splits=len(gpus))
                    dense_inputs_replicas = PerReplica(dense_inputs)
                    dense_labels_replicas = PerReplica(dense_labels)

                    dense_losses, input_grads = distribute_strategy.run(dense_train_step,
                                                                        args=(dense_inputs_replicas, dense_labels_replicas))

                    all_grads = tf.concat(input_grads.values, axis=0)

                    all_grads = nvtx_tf.ops.end(all_grads, dense_ctx)

                embedding_grads = tape.gradient(embedding_result, sparse_model.trainable_weights, output_gradients=all_grads)
                sparse_opt.apply_gradients(zip(embedding_grads, sparse_model.trainable_weights))
                return distribute_strategy.reduce(tf.distribute.ReduceOp.SUM, dense_losses, axis=None) 
        else:
            @tf.function
            def total_train_step(row_indices, values, labels):
                with tf.GradientTape() as tape:
                    row_indices, emb_ctx = nvtx_tf.ops.start(row_indices, 
                                                            message='emb_fprop',
                                                            domain_name='forward')

                    embedding_result = sparse_model(0, row_indices=row_indices, 
                                                    values=values,
                                                    output_shape=[-1, 26, 32],
                                                    training=True)

                    embedding_result = nvtx_tf.ops.end(embedding_result, emb_ctx)

                    embedding_result, dense_ctx = nvtx_tf.ops.start(embedding_result,
                                                                    message='dense_fprop',
                                                                    domain_name='forward')

                    labels = tf.expand_dims(labels, axis=1)

                    dense_loss, embedding_top_grad = dense_train_step(embedding_result, labels)

                    embedding_top_grad = nvtx_tf.ops.end(embedding_top_grad, dense_ctx)

                grads = tape.gradient(embedding_result, sparse_model.trainable_weights, output_gradients=embedding_top_grad)
                sparse_opt.apply_gradients(zip(grads, sparse_model.trainable_weights))
                return dense_loss

        # train loop
        for step, (row_indices, values, labels) in enumerate(dataset):
            step, nvtx_context = nvtx_tf.ops.start(tf.convert_to_tensor(step, dtype=tf.int32), 
                                                message='Iteration_' + str(step), 
                                                domain_name='forward',
                                                grad_message='Back_Iteration_' + str(step),
                                                grad_domain_name='backward')
            total_loss = total_train_step(row_indices,
                                        values,
                                        labels)
            tf.print("step:%d, loss:%.5f" %(step, total_loss))
            step = nvtx_tf.ops.end(step, nvtx_context)

def profile_origin(gpus, num_layers, batch_size):
    # load data first
    dataset = origin_reader("./origin" + "_" + str(len(gpus))) 

    # build model
    sparse_model = OriginSparseModel(gpus)
    sparse_opt = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-3)

    if len(gpus) > 1:
        distribute_strategy = tf.distribute.MirroredStrategy(devices=['/gpu:' + str(i) for i in gpus])
        with distribute_strategy.scope():
            dense_model = DenseModel(num_layers)
            dense_opt = tf.keras.optimizers.SGD()
    else:
        dense_model = DenseModel(num_layers)
        dense_opt = tf.keras.optimizers.SGD()

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    def _dense_loss(labels, logits):
        loss_v = loss_fn(labels, logits)
        return tf.nn.compute_average_loss(loss_v, global_batch_size=batch_size)

    def dense_train_step(dense_inputs, dense_labels):
        with tf.GradientTape() as tape:
            tape.watch(dense_inputs)
            
            dense_result = dense_model(dense_inputs)
            dense_loss = _dense_loss(dense_labels, dense_result)
        dense_grads, input_grads = tape.gradient(dense_loss, [dense_model.trainable_weights, dense_inputs])
        dense_opt.apply_gradients(zip(dense_grads, dense_model.trainable_weights))
        return dense_loss, input_grads

    if len(gpus) > 1:
        @tf.function
        def total_train_step(indices, values, dense_shape, labels):
            with tf.GradientTape() as tape:
                indices, emb_ctx = nvtx_tf.ops.start(indices,
                                                     message='emb_fprop',
                                                     domain_name='forward')

                embedding_result = sparse_model(indices, values, dense_shape, 
                                                output_shape = [-1, 26, 32] # [batch_size, slot_num, embedding_vec_size] 
                                                )
                
                embedding_result = nvtx_tf.ops.end(embedding_result, emb_ctx)

                embedding_result, dense_ctx = nvtx_tf.ops.start(embedding_result, 
                                                     message='dense_fprop',
                                                     domain_name='forward')

                labels = tf.expand_dims(labels, axis=1)
                dense_inputs = tf.split(embedding_result, num_or_size_splits=len(gpus))
                dense_labels = tf.split(labels, num_or_size_splits=len(gpus))
                dense_inputs_replicas = PerReplica(dense_inputs)
                dense_labels_replicas = PerReplica(dense_labels)

                dense_losses, input_grads = distribute_strategy.run(dense_train_step,
                                                                    args=(dense_inputs_replicas, dense_labels_replicas))

                all_grads = tf.concat(input_grads.values, axis=0)

                all_grads = nvtx_tf.ops.end(all_grads, dense_ctx)

            embedding_grads = tape.gradient(embedding_result, sparse_model.trainable_weights, output_gradients=all_grads)
            sparse_opt.apply_gradients(zip(embedding_grads, sparse_model.trainable_weights))
            return distribute_strategy.reduce(tf.distribute.ReduceOp.SUM, dense_losses, axis=None) 
    else:
        @tf.function
        def total_train_step(indices, values, dense_shape, labels):
            with tf.GradientTape() as tape:
                indices, emb_ctx = nvtx_tf.ops.start(indices,
                                                     message='emb_fprop',
                                                     domain_name='forward')

                embedding_result = sparse_model(indices, values, dense_shape,
                                                output_shape = [-1, 26, 32])

                embedding_result = nvtx_tf.ops.end(embedding_result, emb_ctx)

                embedding_result, dense_ctx = nvtx_tf.ops.start(embedding_result, 
                                                                message='dense_fprop',
                                                                domain_name='forward')

                labels = tf.expand_dims(labels, axis=1)

                dense_losses, embedding_top_grad = dense_train_step(embedding_result, labels)

                embedding_top_grad = nvtx_tf.ops.end(embedding_top_grad, dense_ctx)

            grads = tape.gradient(embedding_result, sparse_model.trainable_weights, output_gradients=embedding_top_grad)
            sparse_opt.apply_gradients(zip(grads, sparse_model.trainable_weights))
            return dense_losses

    # train loop
    for step, (indices, values, dense_shape, labels) in enumerate(dataset):
        step, nvtx_context = nvtx_tf.ops.start(tf.convert_to_tensor(step, dtype=tf.int32), 
                                               message='Iteration_' + str(step), 
                                               domain_name='forward',
                                               grad_message='Back_Iteration_' + str(step),
                                               grad_domain_name='backward')

        total_loss = total_train_step(indices, values, dense_shape, labels)

        tf.print("step:%d, loss:%.5f" %(step, total_loss))
        step = nvtx_tf.ops.end(step, nvtx_context)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="profiling")

    parser.add_argument('--gpus', nargs='+', type=int, required=True)
    parser.add_argument('--which', type=str, required=True, choices=['origin', 'plugin'])
    parser.add_argument('--num_layers', type=int, help='number of dense layers', required=False, default=1)
    parser.add_argument('--batch_size', type=int, required=False, default=16384)
    parser.add_argument('--embedding_type', type=str, required=False, default='distributed', 
                        choices=['distributed', 'localized'])
    parser.add_argument('--fprop_version', type=str, required=False, default='v3', choices=['v3', 'v4'])

    parser.add_argument('--prepare_datas', type=int, required=False, default=0, choices=[0, 1])

    args = parser.parse_args()

    if (1 == args.prepare_datas):
        gpus_list = [[i for i in range(j)] for j in [1,2,4,8]]
        for gpus in gpus_list:
            if (args.fprop_version == 'v3'):
                save_dataset_to_python_obj(batch_size=args.batch_size, num_batch=50, save_name="./plugin", 
                                        gpu_count=len(gpus), convert_to_csr=True, embedding_type=args.embedding_type,
                                        get_row_indices=False)
            elif (args.fprop_version == 'v4'):
                save_dataset_to_python_obj(batch_size=args.batch_size, num_batch=50, save_name="./plugin", 
                                        gpu_count=len(gpus), convert_to_csr=False, embedding_type=args.embedding_type,
                                        get_row_indices=True)              
            save_dataset_to_python_obj(batch_size=args.batch_size, num_batch=50, save_name="./origin", 
                                    gpu_count=len(gpus), convert_to_csr=False)
    else:
        if 'origin' == args.which:
            profile_origin(gpus=args.gpus, num_layers=args.num_layers, batch_size=args.batch_size)
        elif 'plugin' == args.which:
            profile_plugin(gpus=args.gpus, num_layers=args.num_layers, batch_size=args.batch_size, 
                            embedding_type=args.embedding_type, fprop_version=args.fprop_version)

        print("*" * 30, "Done.", "*" * 30)
