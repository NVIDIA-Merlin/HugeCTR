import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from dump import DumpToTF, struct
from hugectr_layers import *

def read_a_sample(slot_num=26):
    """
    read a sample from criteo dataset.
    """
    with open("./sparse_embedding0.data", 'rb') as file:
        # skip data_header
        file.seek(4 + 64 + 1, 0)

        # one sample
        length_buffer = file.read(4) # int
        length = struct.unpack('i', length_buffer)

        label_buffer = file.read(4) # int
        label = struct.unpack('i', label_buffer)[0]

        dense_buffer = file.read(4 * 13) # dense_dim * float
        dense = struct.unpack("13f", dense_buffer)

        keys = []
        for _ in range(slot_num):
            nnz_buffer = file.read(4) # int
            nnz = struct.unpack("i", nnz_buffer)[0]
            key_buffer = file.read(8 * nnz) # nnz * long long 
            key = struct.unpack(str(nnz) + "q", key_buffer)
            keys += list(key)

        check_bit_buffer = file.read(1) # char
        check_bit = struct.unpack("c", check_bit_buffer)[0]

    label = np.int64(label)
    dense = np.reshape(np.array(dense, dtype=np.float32), newshape=(1, 13))
    keys = np.reshape(np.array(keys, dtype=np.int64), newshape=(1, 26, 1))

    return label, dense, keys

def dcn_model():
    """
    this function build "dcn" computing-graph with tf.
    and initialize the weights with values dumped from
    hugectr.
    """
    batchsize = 1
    slot_num = 26
    max_nnz_per_slot = 1

    dense_dim = 13

    samples_dir = r'/workspace/hugectr/samples/'
    model_json = os.path.join(samples_dir, r'dcn/dcn.json')

    model_path = r'./hugectr_model_file/'
    sparse_model_names = [os.path.join(model_path, r'dcn_model0_sparse_200000.model')]
    dense_model_name = os.path.join(model_path, r'dcn_model_dense_200000.model')

    dump = DumpToTF(sparse_model_names = sparse_model_names,
                         dense_model_name = dense_model_name,
                         model_json = model_json,
                         non_training_params_json = None)

    checkpoint = r'./tf_checkpoint/dcn_model'

    #----------------- build computing graph--------------------#
    tf.reset_default_graph()

    graph = tf.Graph()
    with graph.as_default():
        # dense-input [batch, dense-dim]
        dense_input = tf.placeholder(shape=(batchsize, dense_dim), 
                        dtype=tf.float32, name='dense-input') 

        # sparse-input = [batch, slot_num, max_nnz_per_slot]
        sparse_input = tf.placeholder(shape=(batchsize, slot_num, max_nnz_per_slot), 
                        dtype=tf.int64, name='sparse-input')

        # dump embedding to tf
        layer_name, init_values = dump.parse_embedding().__next__()
        vocabulary_size = init_values.shape[0]
        embedding_feature = embedding_layer(sparse_input, init_values, combiner=0)

        # reshape
        leading_dim = 416
        reshape1 = tf.reshape(embedding_feature, [-1, leading_dim])

        # concat
        concat1 = tf.concat([reshape1, dense_input], axis=-1)
        
        #slice
        slice1, slice2 = slice_layer(concat1, [0, 0], [concat1.shape[1], concat1.shape[1]])

        # dump multicross to tf
        layer_type = "MultiCross"
        num_layers = 6
        layer_bytes = slice1.shape[1] * 2 * num_layers * 4
        weights, bias = dump.parse_dense(layer_bytes, layer_type, 
                                        vec_length=slice1.shape[1], 
                                        num_layers=num_layers)
        multicross1 = multicross_layer(slice1, weights, bias, layers=num_layers)

        # dum fc1 to tf
        layer_type = "InnerProduct"
        num_output = 1024
        layer_bytes = (slice2.shape[1] * num_output + 1 * num_output) * 4
        weight_fc1, bias_fc1 = dump.parse_dense(layer_bytes, layer_type, 
                                        in_feature=slice2.shape[1],
                                        out_feature=num_output)
        fc1 = innerproduct_layer(slice2, weight_fc1, bias_fc1)

        # relu
        relu1 = tf.nn.relu(fc1)

        # dropout
        rate = 0
        dropout1 = tf.nn.dropout(relu1, rate=rate)

        # dump fc2 to tf
        layer_type = "InnerProduct"
        num_output = 1024
        layer_bytes = (dropout1.shape[1] * num_output + 1 * num_output) * 4
        weight_fc2, bias_fc2 = dump.parse_dense(layer_bytes, layer_type,
                                        in_feature=dropout1.shape[1],
                                        out_feature=num_output)
        fc2 = innerproduct_layer(dropout1, weight_fc2, bias_fc2)

        # relu
        relu2 = tf.nn.relu(fc2)

        # dropout
        rate = 0
        dropout2 = tf.nn.dropout(relu2, rate=rate)

        # concat
        concat2 = tf.concat([dropout2, multicross1[-1]], axis=-1)

        # dump fc4 to tf
        layer_type = "InnerProduct"
        num_output = 1
        layer_bytes = (concat2.shape[1] * num_output + 1 * num_output) * 4
        weight_fc4, bias_fc4 = dump.parse_dense(layer_bytes, layer_type,
                                        in_feature=concat2.shape[1],
                                        out_feature=num_output)
        fc4 = innerproduct_layer(concat2, weight_fc4, bias_fc4)

        # check whether all dense weights are parsed.
        dump.read_dense_complete()

        init_op = tf.group(tf.local_variables_initializer(),
                            tf.global_variables_initializer())

        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.graph.finalize()

        sess.run(init_op)

        # check inference output
        label, dense, keys = read_a_sample()
        keys[keys == -1] = vocabulary_size # map -1 to invalid zeros embedding feature
        output = sess.run(fc4, feed_dict={dense_input: dense,
                                          sparse_input: keys})
        print("[INFO] output = %f" %output)

        # save checkpoint
        saver.save(sess, checkpoint, global_step=0)
        print("[INFO] save done.")


if __name__ == "__main__":
    dcn_model()