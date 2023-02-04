#!/bin/bash
mkdir -p $1
g++ -fopenmp generate_embedding.cpp
./a.out $2 $3 $1