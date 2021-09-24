#!/bin/bash
python 1_convert_pd.py
python 2_remap_id.py
python 3_padding.py
python 4_nvt_process.py