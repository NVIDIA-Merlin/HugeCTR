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

import ast
import json
import os
import pickle
import pandas as pd


def validate_reviews_record(record, file_path, line_number):
    for field in ("reviewerID", "asin"):
        if not isinstance(record.get(field), str) or len(record[field]) == 0:
            raise ValueError(
                "{} line {} field '{}' must be a non-empty string".format(
                    file_path, line_number, field
                )
            )
    if isinstance(record.get("unixReviewTime"), bool) or not isinstance(
        record.get("unixReviewTime"), int
    ):
        raise ValueError(
            "{} line {} field 'unixReviewTime' must be an integer".format(
                file_path, line_number
            )
        )


def validate_meta_record(record, file_path, line_number):
    if not isinstance(record.get("asin"), str) or len(record["asin"]) == 0:
        raise ValueError(
            "{} line {} field 'asin' must be a non-empty string".format(
                file_path, line_number
            )
        )
    categories = record.get("categories")
    if not isinstance(categories, list) or len(categories) == 0:
        raise ValueError(
            "{} line {} field 'categories' must be a non-empty list".format(
                file_path, line_number
            )
        )
    category_path = categories[-1]
    if not isinstance(category_path, (list, tuple)) or len(category_path) == 0:
        raise ValueError(
            "{} line {} field 'categories' must contain non-empty category paths".format(
                file_path, line_number
            )
        )
    if not isinstance(category_path[-1], str) or len(category_path[-1]) == 0:
        raise ValueError(
            "{} line {} final category must be a non-empty string".format(
                file_path, line_number
            )
        )


def validate_record(record, file_path, line_number):
    if not isinstance(record, dict):
        raise ValueError("{} line {} must contain a dictionary record".format(file_path, line_number))
    file_name = os.path.basename(file_path)
    if file_name == "reviews_Electronics_5.json":
        validate_reviews_record(record, file_path, line_number)
    elif file_name == "meta_Electronics.json":
        validate_meta_record(record, file_path, line_number)


def parse_record(line, file_path, line_number):
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        try:
            record = ast.literal_eval(line)
        except (SyntaxError, ValueError) as error:
            raise ValueError(
                "Unable to parse record from {} line {}".format(file_path, line_number)
            ) from error
    validate_record(record, file_path, line_number)
    return record


def to_df(file_path):
    with open(file_path, "r") as fin:
        df = {}
        for i, line in enumerate(fin):
            df[i] = parse_record(line, file_path, i + 1)
        df = pd.DataFrame.from_dict(df, orient="index")
        return df


def main():
    reviews_df = to_df("../raw_data/reviews_Electronics_5.json")
    with open("../raw_data/reviews.pkl", "wb") as f:
        pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

    meta_df = to_df("../raw_data/meta_Electronics.json")
    meta_df = meta_df[meta_df["asin"].isin(reviews_df["asin"].unique())]
    meta_df = meta_df.reset_index(drop=True)
    with open("../raw_data/meta.pkl", "wb") as f:
        pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
