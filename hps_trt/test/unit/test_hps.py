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

import argparse
import ctypes
import json
import numpy as np
import os
import os.path
import re
import sys
import time
import struct
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import pytest

PLUGIN_LIB_PATH = "/usr/local/hps_trt/lib/libhps_plugin.so"
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

args = dict()
args["ps_config_file"] = "foo_bar.json"
args["global_batch_size"] = 1024
args["max_vocabulary_size_per_table_per_model"] = {"foo": [30000], "bar": [50000, 2000]}

hps_config = {
    "supportlonglong": False,
    "models": [
        {
            "model": "foo",
            "sparse_files": ["foo_sparse.model"],
            "num_of_worker_buffer_in_pool": 3,
            "embedding_table_names": ["sparse_embedding0"],
            "embedding_vecsize_per_table": [16],
            "maxnum_catfeature_query_per_table_per_sample": [10],
            "default_value_for_each_table": [1.0],
            "deployed_device_list": [0],
            "max_batch_size": 16384,
            "cache_refresh_percentage_per_iteration": 0.2,
            "hit_rate_threshold": 1.0,
            "gpucacheper": 1.0,
            "gpucache": True,
            "use_static_table": False,
            "use_context_stream": False,
        },
        {
            "model": "bar",
            "sparse_files": ["bar_sparse_0.model", "bar_sparse_1.model"],
            "num_of_worker_buffer_in_pool": 3,
            "embedding_table_names": ["sparse_embedding0", "sparse_embedding1"],
            "embedding_vecsize_per_table": [64, 32],
            "maxnum_catfeature_query_per_table_per_sample": [3, 5],
            "default_value_for_each_table": [1.0, 1.0],
            "deployed_device_list": [0],
            "max_batch_size": 1024,
            "cache_refresh_percentage_per_iteration": 0.2,
            "hit_rate_threshold": 1.0,
            "gpucacheper": 1.0,
            "gpucache": True,
            "use_static_table": True,
            "use_context_stream": True,
        },
    ],
}

hps_config_json_object = json.dumps(hps_config, indent=4)
with open(args["ps_config_file"], "w") as outfile:
    outfile.write(hps_config_json_object)


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


def _generate_embedding_tables():
    embedding_tables = {}
    for model in hps_config["models"]:
        model_name = model["model"]
        max_vocabulary_size_per_table = args["max_vocabulary_size_per_table_per_model"][model_name]
        embedding_vecsize_per_table = model["embedding_vecsize_per_table"]
        embedding_table_paths = model["sparse_files"]
        for table_id in range(len(max_vocabulary_size_per_table)):
            embedding_weights = np.random.random(
                (max_vocabulary_size_per_table[table_id], embedding_vecsize_per_table[table_id])
            ).astype(np.float32)
            convert_to_sparse_model(
                embedding_weights,
                embedding_table_paths[table_id],
                embedding_vecsize_per_table[table_id],
            )
            if model_name not in embedding_tables:
                embedding_tables[model_name] = [embedding_weights]
            else:
                embedding_tables[model_name].append(embedding_weights)
    return embedding_tables


def create_hps_plugin_creator():
    trt_version = [int(n) for n in trt.__version__.split(".")]
    plugin_lib_name = PLUGIN_LIB_PATH
    handle = ctypes.CDLL(plugin_lib_name, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    plg_registry = trt.get_plugin_registry()
    hps_plugin_creator = plg_registry.get_plugin_creator("HPS_TRT", "1", "")
    return hps_plugin_creator


def create_hps_plugin(hps_plugin_creator, model_name, table_id, embedding_vec_size):
    ps_config_file = trt.PluginField(
        "ps_config_file",
        np.array([args["ps_config_file"] + "\0"], dtype=np.string_),
        trt.PluginFieldType.CHAR,
    )
    model_name = trt.PluginField(
        "model_name", np.array([model_name + "\0"], dtype=np.string_), trt.PluginFieldType.CHAR
    )
    table_id = trt.PluginField(
        "table_id", np.array([table_id], dtype=np.int32), trt.PluginFieldType.INT32
    )
    emb_vec_size = trt.PluginField(
        "emb_vec_size", np.array([embedding_vec_size], dtype=np.int32), trt.PluginFieldType.INT32
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


class TestHPS:
    embedding_tables = _generate_embedding_tables()

    hps_plugin_creator = create_hps_plugin_creator()
    assert hps_plugin_creator

    def test_build_engine1(self):
        plugin1 = create_hps_plugin(self.hps_plugin_creator, "foo", 0, 16)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as builder_config:
            input_tensor = network.add_input(name="input", dtype=trt.int32, shape=(-1, 10))
            foobar_hps_layer = network.add_plugin_v2(inputs=[input_tensor], plugin=plugin1)
            foobar_hps_layer.name = "foobar_hps_layer"
            foobar_hps_layer.set_output_type(0, trt.float32)
            foobar_hps_layer.get_output(0).name = "output_0"
            network.mark_output(foobar_hps_layer.get_output(0))

            profile = builder.create_optimization_profile()
            profile.set_shape("input", (1, 10), (1024, 10), (1024, 10))
            builder_config.add_optimization_profile(profile)

            build_start_time = time.time()
            engine = builder.build_serialized_network(network, builder_config)
            build_time_elapsed = time.time() - build_start_time
            TRT_LOGGER.log(
                TRT_LOGGER.INFO, "build foo engine in {:.3f} Sec".format(build_time_elapsed)
            )
            assert engine
            with open("foo.trt", "wb") as fout:
                fout.write(engine)

    def test_build_engine2(self):
        plugin2 = create_hps_plugin(self.hps_plugin_creator, "bar", 0, 64)
        plugin3 = create_hps_plugin(self.hps_plugin_creator, "bar", 1, 32)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as builder_config:
            input_tensor1 = network.add_input(name="input1", dtype=trt.int32, shape=(-1, 3))
            bazquz_hps_layer1 = network.add_plugin_v2(inputs=[input_tensor1], plugin=plugin2)
            bazquz_hps_layer1.name = "bazquz_hps_layer1"
            bazquz_hps_layer1.set_output_type(0, trt.float32)
            bazquz_hps_layer1.get_output(0).name = "output_1"
            network.mark_output(bazquz_hps_layer1.get_output(0))

            input_tensor2 = network.add_input(name="input2", dtype=trt.int32, shape=(-1, 5))
            bazquz_hps_layer2 = network.add_plugin_v2(inputs=[input_tensor2], plugin=plugin3)
            bazquz_hps_layer2.name = "bazquz_hps_layer2"
            bazquz_hps_layer2.set_output_type(0, trt.float32)
            bazquz_hps_layer2.get_output(0).name = "output_2"
            network.mark_output(bazquz_hps_layer2.get_output(0))

            profile = builder.create_optimization_profile()
            profile.set_shape("input1", (1, 3), (1024, 3), (1024, 3))
            profile.set_shape("input2", (1, 5), (1024, 5), (1024, 5))
            builder_config.add_optimization_profile(profile)

            build_start_time = time.time()
            engine = builder.build_serialized_network(network, builder_config)
            build_time_elapsed = time.time() - build_start_time
            TRT_LOGGER.log(
                TRT_LOGGER.INFO, "build bar engine in {:.3f} Sec".format(build_time_elapsed)
            )
            assert engine
            with open("bar.trt", "wb") as fout:
                fout.write(engine)

    def test_execute_engine1(self):
        engine = load_engine("foo.trt")

        BZ = 1
        KEY_DTYPE = np.int32
        TARGET_DTYPE = np.float32
        context = engine.create_execution_context()
        context.set_input_shape("input", (BZ, 10))

        h_input = np.random.randint(30000, size=(BZ, 10)).astype(KEY_DTYPE)
        h_output = np.empty([BZ, 10, 16], dtype=TARGET_DTYPE)
        d_input = cuda.mem_alloc(1 * h_input.nbytes)
        d_output = cuda.mem_alloc(1 * h_output.nbytes)
        bindings = [int(d_input), int(d_output)]
        stream = cuda.Stream()

        for _ in range(5):
            h_input = np.random.randint(30000, size=(BZ, 10)).astype(KEY_DTYPE)
            cuda.memcpy_htod_async(d_input, h_input, stream)
            context.execute_async_v2(bindings, stream.handle)
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()
            ground_truth = self.embedding_tables["foo"][0][h_input]
            diff = h_output.flatten() - ground_truth.flatten()
            mse = np.mean(diff * diff)
            print(mse)
            assert mse <= 1e-6

    def test_execute_engine2(self):
        engine = load_engine("bar.trt")

        BZ = 1
        KEY_DTYPE = np.int32
        TARGET_DTYPE = np.float32
        context = engine.create_execution_context()

        context.set_input_shape("input1", (BZ, 3))
        context.set_input_shape("input2", (BZ, 5))

        h_input1 = np.random.randint(50000, size=(BZ, 3)).astype(KEY_DTYPE)
        h_output1 = np.empty([BZ, 3, 64], dtype=TARGET_DTYPE)
        d_input1 = cuda.mem_alloc(1 * h_input1.nbytes)
        d_output1 = cuda.mem_alloc(1 * h_output1.nbytes)

        h_input2 = np.random.randint(2000, size=(BZ, 5)).astype(KEY_DTYPE)
        h_output2 = np.empty([BZ, 5, 32], dtype=TARGET_DTYPE)
        d_input2 = cuda.mem_alloc(1 * h_input2.nbytes)
        d_output2 = cuda.mem_alloc(1 * h_output2.nbytes)

        bindings = [int(d_input1), int(d_output1), int(d_input2), int(d_output2)]
        stream = cuda.Stream()

        for _ in range(5):
            h_input1 = np.random.randint(50000, size=(BZ, 3)).astype(KEY_DTYPE)
            h_input2 = np.random.randint(2000, size=(BZ, 5)).astype(KEY_DTYPE)
            cuda.memcpy_htod_async(d_input1, h_input1, stream)
            cuda.memcpy_htod_async(d_input2, h_input2, stream)

            context.execute_async_v2(bindings, stream.handle)
            cuda.memcpy_dtoh_async(h_output1, d_output1, stream)
            cuda.memcpy_dtoh_async(h_output2, d_output2, stream)
            stream.synchronize()
            ground_truth1 = self.embedding_tables["bar"][0][h_input1]
            ground_truth2 = self.embedding_tables["bar"][1][h_input2]
            diff1 = h_output1.flatten() - ground_truth1.flatten()
            diff2 = h_output2.flatten() - ground_truth2.flatten()
            mse1 = np.mean(diff1 * diff1)
            mse2 = np.mean(diff2 * diff2)
            print(mse1)
            print(mse2)
            assert mse1 <= 1e-6 and mse2 <= 1e-6
