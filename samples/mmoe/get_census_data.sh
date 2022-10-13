#!/bin/bash

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
