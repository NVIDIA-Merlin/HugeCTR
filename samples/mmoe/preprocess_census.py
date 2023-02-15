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

import pandas as pd

# import torch
# import tqdm
import sys
import numpy as np

write_dir = "data/census_parquet/"

df = pd.read_csv(
    "data/census-income.data",
    names=[
        "age",
        "class_of_worker",
        "industry_recode",
        "occupation_recode",
        "education",
        "wage_per_hour",
        "last_wk_enrolled",
        "marital_status",
        "major_industry_code",
        "major_occupation_code",
        "race",
        "hispanic",
        "sex",
        "labor_union",
        "reason_for_unemp",
        "emp_status",
        "capital_gains",
        "capital_losses",
        "dividends_from_stocks",
        "tax_filer_status",
        "region",
        "state",
        "household_status",
        "summary_of_household",
        "???",
        "migration_msa",
        "migration_reg",
        "migrtion_code-move_reg",
        "live_1_year",
        "migration_sumbelt",
        "num_coworkers",
        "family_members_under_18",
        "father_country_of_birth",
        "mother_country_of_birth",
        "country_of_birth",
        "citizenship",
        "self_employed",
        "veteran_admin",
        "veteran_benefits",
        "weeks_worked_in_year",
        "year",
        "income_50k",
    ],
)
df_test = pd.read_csv(
    "data/census-income.test",
    names=[
        "age",
        "class_of_worker",
        "industry_recode",
        "occupation_recode",
        "education",
        "wage_per_hour",
        "last_wk_enrolled",
        "marital_status",
        "major_industry_code",
        "major_occupation_code",
        "race",
        "hispanic",
        "sex",
        "labor_union",
        "reason_for_unemp",
        "emp_status",
        "capital_gains",
        "capital_losses",
        "dividends_from_stocks",
        "tax_filer_status",
        "region",
        "state",
        "household_status",
        "summary_of_household",
        "???",
        "migration_msa",
        "migration_reg",
        "migrtion_code-move_reg",
        "live_1_year",
        "migration_sumbelt",
        "num_coworkers",
        "family_members_under_18",
        "father_country_of_birth",
        "mother_country_of_birth",
        "country_of_birth",
        "citizenship",
        "self_employed",
        "veteran_admin",
        "veteran_benefits",
        "weeks_worked_in_year",
        "year",
        "income_50k",
    ],
)

# new label to determine if income is >50k
df["income_over_50k"] = (df["income_50k"].astype(str) != " - 50000.").astype(int)
df_test["income_over_50k"] = (df_test["income_50k"].astype(str) != " - 50000.").astype(int)

# new label if never married
df["never_married"] = (df["marital_status"].astype(str) == "Never married").astype(int)
df_test["never_married"] = (df_test["marital_status"].astype(str) == "Never married").astype(int)

collist = list(df.columns)
a, b, c, d = (
    collist.index("age"),
    collist.index("income_over_50k"),
    collist.index("class_of_worker"),
    collist.index("never_married"),
)
collist[a], collist[b] = collist[b], collist[a]
collist[c], collist[d] = collist[d], collist[c]
df = df[collist]

collist = list(df_test.columns)
a, b, c, d = (
    collist.index("age"),
    collist.index("income_over_50k"),
    collist.index("class_of_worker"),
    collist.index("never_married"),
)
collist[a], collist[b] = collist[b], collist[a]
collist[c], collist[d] = collist[d], collist[c]
df_test = df_test[collist]

collist = list(df.columns)
a, b, c, d = (
    collist.index("industry_recode"),
    collist.index("occupation_recode"),
    collist.index("age"),
    collist.index("???"),
)
collist[a], collist[c] = collist[c], collist[a]
collist[b], collist[d] = collist[d], collist[b]
df = df[collist]
df_test = df_test[collist]


temp = df.drop(columns=["income_50k", "marital_status"])
temp_test = df_test.drop(columns=["income_50k", "marital_status"])
df = temp
df_test = temp_test

temp = df.drop(
    columns=[
        "country_of_birth",
        "citizenship",
        "self_employed",
        "veteran_admin",
        "veteran_benefits",
        "weeks_worked_in_year",
        "father_country_of_birth",
        "mother_country_of_birth",
    ]
)

temp_test = df_test.drop(
    columns=[
        "country_of_birth",
        "citizenship",
        "self_employed",
        "veteran_admin",
        "veteran_benefits",
        "weeks_worked_in_year",
        "father_country_of_birth",
        "mother_country_of_birth",
    ]
)
df = temp
df_test = temp_test


hash_df = df.applymap(hash)
df = hash_df.astype(np.int32)
hash_df_test = df_test.applymap(hash)
df_test = hash_df_test.astype(np.int32)

df = df.astype({"income_over_50k": "float32", "never_married": "float32"})
df_test = df_test.astype({"income_over_50k": "float32", "never_married": "float32"})


CATEGORICAL_COLUMNS = list(df.columns)
CATEGORICAL_COLUMNS.remove("income_over_50k")
CATEGORICAL_COLUMNS.remove("never_married")

# compress values to between 0 and slot size
for col in CATEGORICAL_COLUMNS:
    u_list = list(set(df[col].unique()) | set(df_test[col].unique()))
    u_dict = {}

    for idx, x in enumerate(u_list):
        u_dict[x] = idx

    temp = df[col].apply(lambda x: u_dict[x])
    df[col] = temp.astype(np.int32)

    temp_test = df_test[col].apply(lambda x: u_dict[x])
    df_test[col] = temp_test.astype(np.int32)

df.to_parquet(write_dir + "train/0.parquet")
df_test.to_parquet(write_dir + "val/0.parquet")

idx = 2
with open(write_dir + "train/_metadata.json", "w") as f:
    f.write(
        '{"file_stats": [{"file_name": "0.parquet", "num_rows": 199523}], "conts": [], "cats": ['
    )
    for col in CATEGORICAL_COLUMNS:
        #        f.write("{\"col_name\": \"" + col + "\", \"index\": " + str(idx) + "}")
        f.write('{"col_name": "C' + str(idx - 1) + '", "index": ' + str(idx) + "}")
        idx = idx + 1
        if idx < 34:
            f.write(", ")
    f.write(
        '], "labels": [{"col_name": "income_over_50k", "index": 0}, {"col_name": "never_married", "index": 1}]}'
    )

idx = 2
with open(write_dir + "val/_metadata.json", "w") as f:
    f.write(
        '{"file_stats": [{"file_name": "0.parquet", "num_rows": 99762}], "conts": [], "cats": ['
    )
    for col in CATEGORICAL_COLUMNS:
        #        f.write("{\"col_name\": \"" + col + "\", \"index\": " + str(idx) + "}")
        f.write('{"col_name": "C' + str(idx - 1) + '", "index": ' + str(idx) + "}")
        idx = idx + 1
        if idx < 34:
            f.write(", ")
    f.write(
        '], "labels": [{"col_name": "income_over_50k", "index": 0}, {"col_name": "never_married", "index": 1}]}'
    )

with open(write_dir + "file_names.txt", "w") as f:
    f.write("1\n./data/census_parquet/train/0.parquet")

with open(write_dir + "file_names_val.txt", "w") as f:
    f.write("1\n./data/census_parquet/val/0.parquet")
