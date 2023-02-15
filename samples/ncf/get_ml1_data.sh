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

wget -nc https://files.grouplens.org/datasets/movielens/ml-1m.zip -P data -q --show-progress
unzip -n data/ml-1m.zip -d data
rm data/ml-1m.zip
mv data/ml-1m/ratings.dat data/ml-1m/ratings.csv
sed -i 's/::/,/g' data/ml-1m/ratings.csv
sed -i '1s/^/userId,movieId,rating,timestamp\n/' data/ml-1m/ratings.csv
