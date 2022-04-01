#!/bin/bash
wget -nc https://files.grouplens.org/datasets/movielens/ml-1m.zip -P data -q --show-progress
unzip -n data/ml-1m.zip -d data
rm data/ml-1m.zip
mv data/ml-1m/ratings.dat data/ml-1m/ratings.csv
sed -i 's/::/,/g' data/ml-1m/ratings.csv
sed -i '1s/^/userId,movieId,rating,timestamp\n/' data/ml-1m/ratings.csv
