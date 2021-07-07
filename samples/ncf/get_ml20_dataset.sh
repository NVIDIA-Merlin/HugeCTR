#!/bin/bash
wget -nc http://files.grouplens.org/datasets/movielens/ml-20m.zip -P data -q --show-progress
unzip -n data/ml-20m.zip -d data
rm data/ml-20m.zip
