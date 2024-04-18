import onnx_graphsurgeon as gs
from onnx import shape_inference
import numpy as np
import onnx
import tensorrt as trt
import ctypes
import os

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

args = dict()
args["hps_trt_plugin_lib_path"] = "/usr/local/hps_trt/lib/libhps_plugin.so"
args["dlrm_dense_saved_path"] = "dlrm_dense_tf_saved_model"
args["dlrm_dense_onnx_path"] = "dlrm_dense.onnx"
args["hps_plugin_dlrm_onnx_path"] = "hps_plugin_dlrm.onnx"
args["hps_plugin_dlrm_trt_path"] = "hps_plugin_dlrm.trt"


def onnx_surgery(args):
    graph = gs.import_onnx(onnx.load(args["dlrm_dense_onnx_path"]))
    saved = []
    for i in graph.inputs:
        if i.name == "args_0":
            categorical_features = gs.Variable(
                name="categorical_features", dtype=np.int32, shape=("unknown", 26)
            )
            node = gs.Node(
                op="HPS_TRT",
                attrs={
                    "ps_config_file": "dlrm.json\0",
                    "model_name": "dlrm\0",
                    "table_id": 0,
                    "emb_vec_size": 128,
                },
                inputs=[categorical_features],
                outputs=[i],
            )
            graph.nodes.append(node)
            saved.append(categorical_features)
        if i.name == "args_0_1":
            i.name = "numerical_features"
            saved.append(i)
    graph.inputs = saved
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), args["hps_plugin_dlrm_onnx_path"])


def create_hps_plugin_creator(args):
    plugin_lib_name = args["hps_trt_plugin_lib_path"]
    handle = ctypes.CDLL(plugin_lib_name, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    plg_registry = trt.get_plugin_registry()
    for plugin_creator in plg_registry.plugin_creator_list:
        if plugin_creator.name[0] == "H":
            print(plugin_creator.name)
    hps_plugin_creator = plg_registry.get_plugin_creator("HPS_TRT", "1", "")
    return hps_plugin_creator


def build_engine_from_onnx(args, fp16):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        EXPLICIT_BATCH
    ) as network, trt.OnnxParser(
        network, TRT_LOGGER
    ) as parser, builder.create_builder_config() as builder_config:
        model = open(args["hps_plugin_dlrm_onnx_path"], "rb")
        parser.parse(model.read())
        print(network.num_layers)

        trt_engine_saved_path = "fp32_" + args["hps_plugin_dlrm_trt_path"]
        if fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
            trt_engine_saved_path = "fp16_" + args["hps_plugin_dlrm_trt_path"]

        profile = builder.create_optimization_profile()
        profile.set_shape("categorical_features", (1, 26), (1024, 26), (131072, 26))
        profile.set_shape("numerical_features", (1, 13), (1024, 13), (131072, 13))
        builder_config.add_optimization_profile(profile)

        engine = builder.build_serialized_network(network, builder_config)
        with open(trt_engine_saved_path, "wb") as fout:
            fout.write(engine)


if __name__ == "__main__":
    os.system(
        "python -m tf2onnx.convert --saved-model "
        + args["dlrm_dense_saved_path"]
        + " --output "
        + args["dlrm_dense_onnx_path"]
    )
    onnx_surgery(args)
    create_hps_plugin_creator(args)
    build_engine_from_onnx(args, fp16=False)
    build_engine_from_onnx(args, fp16=True)
