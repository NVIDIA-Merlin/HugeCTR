#!/usr/bin/env bash
#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

wget https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz
wget https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz

mkdir data
mv census-income.data.gz data/
mv census-income.test.gz data/

cd data
gunzip census-income.data.gz
gunzip census-income.test.gz

mkdir census_parquet
mkdir census_parquet/train
mkdir census_parquet/val
