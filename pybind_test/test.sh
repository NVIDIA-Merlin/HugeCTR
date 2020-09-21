#!/bin/bash

cp ../build/HugeCTR/pybind/hugectr.so ./
python3 data_reader_test.py
python3 data_reader_raw_test.py
python3 data_reader_parquet_test.py
python3 session_test.py
