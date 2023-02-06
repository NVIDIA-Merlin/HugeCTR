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

import tensorrt as trt
import ctypes

import os
import numpy as np
import struct

import json
import pytest

import onnx_graphsurgeon as gs
from onnx import shape_inference
import onnx

import pycuda.driver as cuda
import pycuda.autoinit
import onnxruntime as ort

import hugectr2onnx

plugin_lib_name = "/usr/local/hps_trt/lib/libhps_plugin.so"
handle = ctypes.CDLL(plugin_lib_name, mode=ctypes.RTLD_GLOBAL)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_ENGINE_ONNX_SAVED_PATH = "dlrm_hugectr_with_hps.trt"
BZ = 1024
KEY_DTYPE = np.int32
TARGET_DTYPE = np.float32

GPU_NUM = 1
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(GPU_NUM)))

hps_config = {
    "supportlonglong": False,
    "models": [
        {
            "model": "dlrm",
            "sparse_files": ["dlrm_hugectr0_sparse_1000.model"],
            "num_of_worker_buffer_in_pool": 3,
            "embedding_table_names": ["sparse_embedding0"],
            "embedding_vecsize_per_table": [128],
            "maxnum_catfeature_query_per_table_per_sample": [26],
            "default_value_for_each_table": [1.0],
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
with open("dlrm_hugectr.json", "w") as outfile:
    outfile.write(hps_config_json_object)


def onnx_graph_surgery():
    graph = gs.import_onnx(onnx.load("dlrm_hugectr_dense.onnx"))
    saved = []

    for i in graph.inputs:
        if i.name == "sparse_embedding1":
            categorical_features = gs.Variable(
                name="categorical_features", dtype=np.int32, shape=("unknown_1", 26)
            )
            node = gs.Node(
                op="HPS_TRT",
                attrs={
                    "ps_config_file": "dlrm_hugectr.json\0",
                    "model_name": "dlrm\0",
                    "table_id": 0,
                    "emb_vec_size": 128,
                },
                inputs=[categorical_features],
                outputs=[i],
            )
            graph.nodes.append(node)
            saved.append(categorical_features)
        elif i.name == "numerical_features":
            i.shape = ("unknown_2", 13)
            saved.append(i)

    graph.inputs = saved

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), "dlrm_hugectr_with_hps.onnx")


def build_engine_from_onnx(onnx_model_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        EXPLICIT_BATCH
    ) as network, trt.OnnxParser(
        network, TRT_LOGGER
    ) as parser, builder.create_builder_config() as builder_config:
        model = open(onnx_model_path, "rb")
        parser.parse(model.read())

        profile = builder.create_optimization_profile()
        profile.set_shape("categorical_features", (1, 26), (1024, 26), (1024, 26))
        profile.set_shape("numerical_features", (1, 13), (1024, 13), (1024, 13))
        builder_config.add_optimization_profile(profile)
        engine = builder.build_serialized_network(network, builder_config)
        return engine


def create_hps_plugin_creator():
    trt_version = [int(n) for n in trt.__version__.split(".")]
    handle = ctypes.CDLL(plugin_lib_name, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    plg_registry = trt.get_plugin_registry()
    hps_plugin_creator = plg_registry.get_plugin_creator("HPS_TRT", "1", "")
    return hps_plugin_creator


def create_hps_plugin(hps_plugin_creator):
    ps_config_file = trt.PluginField(
        "ps_config_file",
        np.array(["dlrm_hugectr.json\0"], dtype=np.string_),
        trt.PluginFieldType.CHAR,
    )
    model_name = trt.PluginField(
        "model_name", np.array(["dlrm\0"], dtype=np.string_), trt.PluginFieldType.CHAR
    )
    table_id = trt.PluginField("table_id", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32)
    emb_vec_size = trt.PluginField(
        "emb_vec_size", np.array([128], dtype=np.int32), trt.PluginFieldType.INT32
    )
    params = trt.PluginFieldCollection([ps_config_file, model_name, table_id, emb_vec_size])
    hps_plugin = hps_plugin_creator.create_plugin(name="hps", field_collection=params)
    return hps_plugin


def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    trt_engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return trt_engine


def predict(
    categorical_h_input,
    numerical_h_input,
    h_output,
    categorical_d_input,
    numerical_d_input,
    d_output,
    stream,
    bindings,
    context,
):
    cuda.memcpy_htod_async(categorical_d_input, categorical_h_input, stream)
    cuda.memcpy_htod_async(numerical_d_input, numerical_h_input, stream)
    context.execute_async_v2(bindings, stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return h_output


def read_sparse_model(sparse_model_path, embedding_table):
    vocab_size = embedding_table.shape[0]
    embedding_vec_size = embedding_table.shape[1]
    with open(sparse_model_path + "/key", "rb") as key_file, open(
        sparse_model_path + "/emb_vector", "rb"
    ) as vec_file:
        while True:
            key_buffer = key_file.read(8)
            vec_buffer = vec_file.read(4 * embedding_vec_size)
            if len(key_buffer) == 0 or len(vec_buffer) == 0:
                break
            key = struct.unpack("q", key_buffer)[0]
            values = struct.unpack(str(embedding_vec_size) + "f", vec_buffer)
            assert key >= 0 and key < vocab_size
            embedding_table[key] = values
    return embedding_table


def create_onnx_inference():
    embedding_table = np.ones((260000, 128)).astype(TARGET_DTYPE)
    read_sparse_model("dlrm_hugectr0_sparse_1000.model", embedding_table)
    sess = ort.InferenceSession("dlrm_hugectr_dense.onnx")
    return embedding_table, sess


def run_onnx_inference(embedding_table, sess, categorical_h, numerical_h):
    embedding = embedding_table[categorical_h]
    res = sess.run(
        output_names=[sess.get_outputs()[0].name],
        input_feed={sess.get_inputs()[0].name: numerical_h, sess.get_inputs()[1].name: embedding},
    )
    return res


def test_build_engine_for_hugectr():
    hugectr2onnx.converter.convert(
        onnx_model_path="dlrm_hugectr_dense.onnx",
        graph_config="dlrm_hugectr_graph.json",
        dense_model="dlrm_hugectr_dense_1000.model",
        convert_embedding=False,
    )
    onnx_graph_surgery()

    hps_plugin_creator = create_hps_plugin_creator()
    hps_plugin = create_hps_plugin(hps_plugin_creator)

    serialized_engine = build_engine_from_onnx("dlrm_hugectr_with_hps.onnx")
    with open("dlrm_hugectr_with_hps.trt", "wb") as fout:
        fout.write(serialized_engine)
    print("Succesfully build the TensorRT engine")
    assert True


def test_run_engine():

    hps_plugin_creator = create_hps_plugin_creator()
    hps_plugin = create_hps_plugin(hps_plugin_creator)

    embedding_table, sess = create_onnx_inference()

    engine = load_engine(TRT_ENGINE_ONNX_SAVED_PATH)
    context = engine.create_execution_context()
    context.set_input_shape("numerical_features", (BZ, 13))
    context.set_input_shape("categorical_features", (BZ, 26))

    categorical_h_input = np.random.randint(0, 260000, (BZ, 26)).astype(KEY_DTYPE)
    numerical_h_input = np.random.random((BZ, 13)).astype(TARGET_DTYPE)
    h_output = np.empty([BZ, 1], dtype=TARGET_DTYPE)

    categorical_d_input = cuda.mem_alloc(1 * categorical_h_input.nbytes)
    numerical_d_input = cuda.mem_alloc(1 * numerical_h_input.nbytes)
    d_output = cuda.mem_alloc(1 * h_output.nbytes)
    bindings = [int(numerical_d_input), int(categorical_d_input), int(d_output)]
    stream = cuda.Stream()

    for i in range(5):
        categorical_h_input = np.random.randint(0, 260000, (BZ, 26)).astype(KEY_DTYPE)
        numerical_h_input = np.random.random((BZ, 13)).astype(TARGET_DTYPE)

        preds = predict(
            categorical_h_input,
            numerical_h_input,
            h_output,
            categorical_d_input,
            numerical_d_input,
            d_output,
            stream,
            bindings,
            context,
        )

        onnx_preds = run_onnx_inference(
            embedding_table, sess, categorical_h_input, numerical_h_input
        )

        diff = preds.flatten() - onnx_preds[0].flatten()
        mse = np.mean(diff * diff)
        print(mse)
        print(preds)
        print(onnx_preds[0])
        assert mse <= 1e-6
