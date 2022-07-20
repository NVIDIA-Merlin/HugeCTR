#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from hugectr2onnx.hugectr_loader import HugeCTRLoader, LayerParams
from hugectr2onnx.graph_builder import GraphBuilder
import argparse


def convert(
    onnx_model_path,
    graph_config,
    dense_model,
    convert_embedding=False,
    sparse_models=None,
    ntp_file=None,
    graph_name="hugectr",
):
    """Convert a HugeCTR model to an ONNX model
    Args:
        onnx_model_path: the path to store the ONNX model
        graph_config: the graph configuration JSON file of the HugeCTR model
        dense_model: the file of the dense weights for the HugeCTR model
        convert_embedding: whether to convert the sparse embeddings for the HugeCTR model (optional)
        sparse_models: the files of the sparse embeddings for the HugeCTR model (optional)
        ntp_file: the file of the non-trainable parameters for the HugeCTR model (optional)
        graph_name: the graph name for the ONNX model (optional)
    """
    loader = HugeCTRLoader(graph_config, dense_model, convert_embedding, sparse_models, ntp_file)
    builder = GraphBuilder(convert_embedding)
    for _ in range(loader.layers):
        layer_params, weights_dict, dimensions = loader.load_layer()
        print(f"[HUGECTR2ONNX][INFO]: Converting {layer_params.layer_type} layer to ONNX")
        builder.add_layer(layer_params, weights_dict, dimensions)
    builder.create_graph(graph_name)
    builder.save_model(onnx_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HugeCTR model to ONNX.")
    parser.add_argument("--onnx_model_path", type=str, required=True, help="Output onnx model path")
    parser.add_argument(
        "--graph_config", type=str, required=True, help="HugeCTR graph configuration file"
    )
    parser.add_argument("--dense_model", type=str, required=True, help="HugeCTR dense model")
    parser.add_argument(
        "--convert_embedding",
        action="store_true",
        help="Convert sparse embedding or not (optional)",
    )
    parser.add_argument("--sparse_models", nargs="*", help="HugeCTR sparse models (optional)")
    parser.add_argument(
        "--ntp_file", type=str, default=None, help="HugeCTR non-trainable parameters (optional)"
    )
    parser.add_argument(
        "--graph_name", type=str, default="hugectr", help="Graph name for the ONNX model (optional)"
    )
    args = parser.parse_args()
    print(args)
    convert(
        args.onnx_model_path,
        args.graph_config,
        args.dense_model,
        args.convert_embedding,
        args.sparse_models,
        args.ntp_file,
        args.graph_name,
    )
