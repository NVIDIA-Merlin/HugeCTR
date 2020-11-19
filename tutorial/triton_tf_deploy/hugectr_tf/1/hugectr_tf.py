import sys
sys.path.append("/hugectr/tutorial/dump_to_tf")

from dump import DumpToTF, struct
from hugectr_layers import *

import os
import json
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' 
os.environ['CUDA_VISIBLE_DEVICES']='0'

try:
    # tensorflow 2.x
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    tf.disable_eager_execution()
except:
    # tensorflow 1.x
    import tensorflow as tf
from tensorflow.python.framework import graph_util

def hugectr_tf_export(model, json_file, dense_model, sparse_models):
    if model == "criteo":
        batchsize = 1
        slot_num = 1
        max_nnz_per_slot = 39
        dense_dim = 0
        model_json = json_file
        sparse_model_names = sparse_models
        dense_model_name = dense_model
        dump = DumpToTF(sparse_model_names = sparse_model_names,
                            dense_model_name = dense_model_name,
                            model_json = model_json,
                            non_training_params_json = None)
        #----------------- build computing graph--------------------#
        tf.reset_default_graph()
        # no dense input
        # sparse input = [batchsize, slot_num, max_nnz_per_slot]
        input0 = tf.placeholder(shape=(max_nnz_per_slot), dtype=tf.int64, name="INPUT0")
        sparse_input = tf.reshape(input0, [batchsize, slot_num, max_nnz_per_slot],
                        name="sparse_input")
        # dump embedding to tf
        layer_name, init_values = dump.parse_embedding().__next__()
        vocabulary_size = init_values.shape[0]
        embedding_feature = embedding_layer(sparse_input, init_values, combiner=0)
        # reshape1
        leading_dim = 64
        reshape1 = tf.reshape(embedding_feature, [-1, leading_dim])
        # dump fc1 to tf
        layer_type = "InnerProduct"
        num_output = 200
        layer_bytes = (reshape1.shape[1] * num_output + 1 * num_output) * 4
        weight_fc1, bias_fc1 = dump.parse_dense(layer_bytes, layer_type,
                                            in_feature=reshape1.shape[1],
                                            out_feature=num_output)
        fc1 = innerproduct_layer(reshape1, weight_fc1, bias_fc1)
        # relu
        relu1 = tf.nn.relu(fc1)
        # dump fc2 to tf
        layer_type = "InnerProduct"
        num_output = 200
        layer_bytes = (relu1.shape[1] * num_output + 1 * num_output) * 4
        weight_fc2, bias_fc2 = dump.parse_dense(layer_bytes, layer_type,
                                            in_feature=relu1.shape[1],
                                            out_feature=num_output)
        fc2 = innerproduct_layer(relu1, weight_fc2, bias_fc2)
        # relu2
        relu2 = tf.nn.relu(fc2)
        # dump fc3 to tf
        layer_type = "InnerProduct"
        num_output = 200
        layer_bytes = (relu2.shape[1] * num_output + 1 * num_output) * 4
        weight_fc3, bias_fc3 = dump.parse_dense(layer_bytes, layer_type,
                                            in_feature=relu2.shape[1],
                                            out_feature=num_output)
        fc3 = innerproduct_layer(relu2, weight_fc3, bias_fc3)
        # relu3
        relu3 = tf.nn.relu(fc3)
        # dump fc4 to tf
        layer_type = "InnerProduct"
        num_output = 1
        layer_bytes = (relu3.shape[1] * num_output + 1 * num_output) * 4
        weight_fc4, bias_fc4 = dump.parse_dense(layer_bytes, layer_type,
                                            in_feature=relu3.shape[1],
                                            out_feature=num_output)
        fc4 = innerproduct_layer(relu3, weight_fc4, bias_fc4)
        output0 = tf.reshape(fc4, [1], name="OUTPUT0")
        # check whether all dense weights are parsed
        dump.read_dense_complete()
        init_op = tf.group(tf.local_variables_initializer(),
                        tf.global_variables_initializer())
        print("[INFO] hugectr tf graph has been constructed.")
    elif model == "dcn":
        batchsize = 1
        slot_num = 26
        max_nnz_per_slot = 1
        dense_dim = 13
        model_json = json_file
        dense_model_name = dense_model
        sparse_model_names = sparse_models
        dump = DumpToTF(sparse_model_names = sparse_model_names,
                            dense_model_name = dense_model_name,
                            model_json = model_json,
                            non_training_params_json = None)
        #----------------- build computing graph--------------------#
        tf.reset_default_graph()
        # dense-input [batch, dense-dim]
        input0 = tf.placeholder(shape=(slot_num), dtype=tf.int64, name="INPUT0")
        input1 = tf.placeholder(shape=(dense_dim), dtype=tf.float32, name="INPUT1")
        sparse_input = tf.reshape(input0, shape=(batchsize, slot_num, max_nnz_per_slot), 
                                   name='sparse-input')
        dense_input = tf.reshape(input1, shape=(batchsize, dense_dim), 
                                   name='dense-input') 
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
        output0 = tf.reshape(fc4, [1], name="OUTPUT0")
        # check whether all dense weights are parsed.
        dump.read_dense_complete()
        init_op = tf.group(tf.local_variables_initializer(),
                        tf.global_variables_initializer())
        print("[INFO] hugectr tf graph has been constructed.")
    else:
        print("[ERROR] can only deal with Criteo or DCN model!")
        assert(False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        graph_def = tf.get_default_graph().as_graph_def()
        tensor_name_list = [tensor.name for tensor in graph_def.node]
        for tensor_name in tensor_name_list:
            print("[INFO] ",tensor_name)
        output_graphDef = graph_util.convert_variables_to_constants(sess, graph_def, output_node_names=["OUTPUT0"])
        try:
            with tf.gfile.GFile("./hugectr_tf.graphdef", 'wb') as f:
                f.write(output_graphDef.SerializeToString())
            print("[INFO] tf graphdef saved successfully!")
        except:
            print("[ERROR] something wrong happend")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="dcn", choices=['dcn', 'criteo'],
                        help="decide to run which model, must be 'dcn' or 'criteo'.")
    parser.add_argument("--json_file", type=str, 
                        help="the configuration json file.")
    parser.add_argument("--dense_model", type=str, 
                        help="where to find dense model file.")
    parser.add_argument("--sparse_models", nargs="+", type=str,
                        help="where to find sparse model files.")
    args = parser.parse_args()
    hugectr_tf_export(args.model, args.json_file, args.dense_model, args.sparse_models)
