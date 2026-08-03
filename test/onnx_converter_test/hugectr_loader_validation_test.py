"""
 Copyright (c) 2026, NVIDIA CORPORATION.

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

import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest


def load_hugectr_loader():
    repo_root = Path(__file__).resolve().parents[2]
    loader_path = repo_root / "onnx_converter" / "hugectr2onnx" / "hugectr_loader.py"
    spec = importlib.util.spec_from_file_location("hugectr_loader_under_test", loader_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HugeCTRLoader


HugeCTRLoader = load_hugectr_loader()


class HugeCTRLoaderValidationTest(unittest.TestCase):
    def write_json(self, path, content):
        with open(path, "w") as file:
            json.dump(content, file)

    def make_data_layer(self, dense_dim=1, sparse_layers=None):
        return {
            "type": "Data",
            "label": {"top": "label", "label_dim": 1},
            "dense": {"top": "dense", "dense_dim": dense_dim},
            "sparse": sparse_layers or [],
        }

    def expect_last_layer_value_error(self, layers, expected_message, ntp_file=None):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = os.path.join(tmpdir, "graph.json")
            model_path = os.path.join(tmpdir, "dense.model")
            self.write_json(graph_path, {"layers": layers})
            open(model_path, "wb").close()

            loader = HugeCTRLoader(graph_path, model_path, ntp_file=ntp_file)
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(len(layers) - 1):
                    loader.load_layer()

            with self.assertRaisesRegex(ValueError, expected_message):
                loader.load_layer()

    def test_weighted_dense_layers_reject_truncated_model_before_unpack(self):
        sparse_input = {"top": "sparse", "slot_num": 2, "nnz_per_slot": [1]}
        embedding_layer = {
            "type": "DistributedSlotSparseEmbeddingHash",
            "bottom": "sparse",
            "top": "embedding",
            "sparse_embedding_hparam": {
                "embedding_vec_size": 3,
                "max_vocabulary_size_global": 4,
                "combiner": "sum",
            },
        }
        cases = [
            (
                "BatchNorm",
                [
                    self.make_data_layer(dense_dim=3),
                    {
                        "type": "BatchNorm",
                        "bottom": "dense",
                        "top": "bn",
                        "bn_param": {"factor": 1.0, "eps": 0.001},
                    },
                ],
                "BatchNorm layer requires 24 bytes",
            ),
            (
                "LayerNorm",
                [
                    self.make_data_layer(dense_dim=0, sparse_layers=[sparse_input]),
                    embedding_layer,
                    {
                        "type": "LayerNorm",
                        "bottom": "embedding",
                        "top": "ln",
                        "ln_param": {"eps": 0.001},
                    },
                ],
                "LayerNorm layer requires 24 bytes",
            ),
            (
                "InnerProduct",
                [
                    self.make_data_layer(dense_dim=2),
                    {
                        "type": "InnerProduct",
                        "bottom": "dense",
                        "top": "fc",
                        "fc_param": {"num_output": 3},
                    },
                ],
                "InnerProduct layer requires 36 bytes",
            ),
            (
                "MLP",
                [
                    self.make_data_layer(dense_dim=2),
                    {
                        "type": "MLP",
                        "bottom": "dense",
                        "top": "mlp",
                        "mlp_param": {"num_outputs": [3]},
                    },
                ],
                "MLP layer requires 36 bytes",
            ),
            (
                "MultiCross",
                [
                    self.make_data_layer(dense_dim=2),
                    {
                        "type": "MultiCross",
                        "bottom": "dense",
                        "top": "mc",
                        "mc_param": {"num_layers": 2},
                    },
                ],
                "MultiCross layer requires 32 bytes",
            ),
            (
                "WeightMultiply",
                [
                    self.make_data_layer(dense_dim=1),
                    {
                        "type": "WeightMultiply",
                        "bottom": "dense",
                        "top": "wm",
                        "weight_dims": [2, 3],
                    },
                ],
                "WeightMultiply layer requires 24 bytes",
            ),
        ]

        for case_name, layers, expected_message in cases:
            with self.subTest(case_name=case_name):
                self.expect_last_layer_value_error(layers, expected_message)

    def test_huge_dense_dimension_is_rejected_without_memory_error(self):
        layers = [
            self.make_data_layer(dense_dim=1),
            {
                "type": "InnerProduct",
                "bottom": "dense",
                "top": "fc",
                "fc_param": {"num_output": 250000000},
            },
        ]
        self.expect_last_layer_value_error(
            layers, "InnerProduct layer requires 2000000000 bytes"
        )

    def test_sparse_embedding_rejects_partial_vector_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = os.path.join(tmpdir, "graph.json")
            model_path = os.path.join(tmpdir, "dense.model")
            sparse_model = os.path.join(tmpdir, "sparse.model")
            os.mkdir(sparse_model)
            open(model_path, "wb").close()
            with open(os.path.join(sparse_model, "key"), "wb") as key_file:
                key_file.write(struct.pack("q", 0))
            with open(os.path.join(sparse_model, "emb_vector"), "wb") as vec_file:
                vec_file.write(struct.pack("f", 1.0))
            self.write_json(
                graph_path,
                {
                    "layers": [
                        self.make_data_layer(
                            dense_dim=0,
                            sparse_layers=[{"top": "sparse", "slot_num": 1, "nnz_per_slot": [1]}],
                        ),
                        {
                            "type": "DistributedSlotSparseEmbeddingHash",
                            "bottom": "sparse",
                            "top": "embedding",
                            "sparse_embedding_hparam": {
                                "embedding_vec_size": 4,
                                "max_vocabulary_size_global": 8,
                                "combiner": "sum",
                            },
                        },
                    ]
                },
            )

            loader = HugeCTRLoader(graph_path, model_path, True, [sparse_model])
            loader.load_layer()
            with self.assertRaisesRegex(ValueError, "not aligned to embedding vector size"):
                loader.load_layer()

    def test_sparse_embedding_rejects_key_outside_hash_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = os.path.join(tmpdir, "graph.json")
            model_path = os.path.join(tmpdir, "dense.model")
            sparse_model = os.path.join(tmpdir, "sparse.model")
            os.mkdir(sparse_model)
            open(model_path, "wb").close()
            with open(os.path.join(sparse_model, "key"), "wb") as key_file:
                key_file.write(struct.pack("q", 99))
            with open(os.path.join(sparse_model, "emb_vector"), "wb") as vec_file:
                vec_file.write(struct.pack("2f", 1.0, 2.0))
            self.write_json(
                graph_path,
                {
                    "layers": [
                        self.make_data_layer(
                            dense_dim=0,
                            sparse_layers=[{"top": "sparse", "slot_num": 1, "nnz_per_slot": [1]}],
                        ),
                        {
                            "type": "DistributedSlotSparseEmbeddingHash",
                            "bottom": "sparse",
                            "top": "embedding",
                            "sparse_embedding_hparam": {
                                "embedding_vec_size": 2,
                                "max_vocabulary_size_global": 8,
                                "combiner": "sum",
                            },
                        },
                    ]
                },
            )

            loader = HugeCTRLoader(graph_path, model_path, True, [sparse_model])
            loader.load_layer()
            with self.assertRaisesRegex(ValueError, "outside hash table range"):
                loader.load_layer()


if __name__ == "__main__":
    unittest.main()
