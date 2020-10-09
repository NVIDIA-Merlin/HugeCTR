#!/bin/bash

cp ../build/lib/hugectr.so ./
python3 data_reader_test.py
python3 data_reader_raw_test.py
python3 data_reader_parquet_test.py
python3 session_test.py

rm hugectr.so
cp ../build_multi/lib/hugectr.so ./
mpirun --allow-run-as-root -np 2 python3 multi_node_test.py
