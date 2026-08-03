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
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERT_PD_SCRIPTS = [
    REPO_ROOT / "samples" / "bst" / "utils" / "1_convert_pd.py",
    REPO_ROOT / "samples" / "din" / "utils" / "1_convert_pd.py",
]


def load_module(path):
    pandas_module = sys.modules.get("pandas")
    if pandas_module is None:
        sys.modules["pandas"] = make_fake_pandas()
    spec = importlib.util.spec_from_file_location(path.parent.parent.name + "_convert_pd", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        if pandas_module is None:
            del sys.modules["pandas"]
    return module


def make_fake_pandas():
    class FakeDataFrame(object):
        def __init__(self, rows):
            self.rows = rows

        @classmethod
        def from_dict(cls, records, orient):
            if orient != "index":
                raise ValueError("unexpected orient")
            return cls([records[index] for index in sorted(records)])

        def __getitem__(self, key):
            return [row[key] for row in self.rows]

    return types.SimpleNamespace(DataFrame=FakeDataFrame)


class ConvertPdValidationTest(unittest.TestCase):
    def write_records(self, directory, file_name, records):
        path = os.path.join(directory, file_name)
        with open(path, "w") as data_file:
            for record in records:
                data_file.write(record + "\n")
        return path

    def test_to_df_accepts_json_and_python_literal_records(self):
        for script_path in CONVERT_PD_SCRIPTS:
            with self.subTest(script_path=str(script_path)):
                module = load_module(script_path)
                with tempfile.NamedTemporaryFile("w", delete=False) as data_file:
                    data_file.write('{"asin": "json-record", "overall": 5.0}\n')
                    data_file.write("{'asin': 'literal-record', 'overall': 4.0}\n")
                    data_path = data_file.name
                try:
                    df = module.to_df(data_path)
                finally:
                    os.unlink(data_path)

                self.assertEqual(["json-record", "literal-record"], list(df["asin"]))
                self.assertEqual([5.0, 4.0], list(df["overall"]))

    def test_to_df_accepts_expected_review_and_meta_schema(self):
        for script_path in CONVERT_PD_SCRIPTS:
            with self.subTest(script_path=str(script_path)):
                module = load_module(script_path)
                with tempfile.TemporaryDirectory() as tmpdir:
                    reviews_path = self.write_records(
                        tmpdir,
                        "reviews_Electronics_5.json",
                        [
                            '{"reviewerID": "user", "asin": "item", "unixReviewTime": 123}',
                        ],
                    )
                    meta_path = self.write_records(
                        tmpdir,
                        "meta_Electronics.json",
                        [
                            '{"asin": "item", "categories": [["Electronics", "Camera"]]}',
                        ],
                    )

                    reviews_df = module.to_df(reviews_path)
                    meta_df = module.to_df(meta_path)

                self.assertEqual(["user"], list(reviews_df["reviewerID"]))
                self.assertEqual(["item"], list(meta_df["asin"]))

    def test_to_df_rejects_executable_input(self):
        for script_path in CONVERT_PD_SCRIPTS:
            with self.subTest(script_path=str(script_path)):
                module = load_module(script_path)
                with tempfile.TemporaryDirectory() as tmpdir:
                    marker_path = os.path.join(tmpdir, "eval_executed")
                    data_path = os.path.join(tmpdir, "records.json")
                    with open(data_path, "w") as data_file:
                        data_file.write("__import__('os').system('touch {}')\n".format(marker_path))

                    with self.assertRaisesRegex(ValueError, "Unable to parse record"):
                        module.to_df(data_path)
                    self.assertFalse(os.path.exists(marker_path))

    def test_to_df_rejects_malformed_records(self):
        cases = [
            ("records.json", '["not", "a", "dict"]', "dictionary record"),
            (
                "reviews_Electronics_5.json",
                '{"asin": "item", "unixReviewTime": 123}',
                "reviewerID",
            ),
            (
                "reviews_Electronics_5.json",
                '{"reviewerID": "user", "asin": "item", "unixReviewTime": true}',
                "unixReviewTime",
            ),
            (
                "meta_Electronics.json",
                '{"asin": "item", "categories": []}',
                "categories",
            ),
            (
                "meta_Electronics.json",
                '{"asin": "item", "categories": [["Electronics", ""]]}',
                "final category",
            ),
        ]
        for script_path in CONVERT_PD_SCRIPTS:
            module = load_module(script_path)
            for file_name, record, expected_message in cases:
                with self.subTest(script_path=str(script_path), file_name=file_name):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        data_path = self.write_records(tmpdir, file_name, [record])
                        with self.assertRaisesRegex(ValueError, expected_message):
                            module.to_df(data_path)


if __name__ == "__main__":
    unittest.main()
