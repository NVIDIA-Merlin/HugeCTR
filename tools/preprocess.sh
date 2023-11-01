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

if [[ $# -lt 3 ]]; then
  echo "Usage: preprocess.sh [DATASET_NO.] [DST_DATA_DIR] [SCRIPT_TYPE] [SCRIPT_TYPE_SPECIFIC_ARGS...]"
  exit 2
fi

DST_DATA_DIR=$2

echo "Warning: existing $DST_DATA_DIR is erased"
rm -rf $DST_DATA_DIR

if [[ $3 == "nvt" ]]; then
  if [[ $# -ne 6 ]]; then
		echo "Usage: preprocess.sh [DATASET_NO.] [DST_DATA_DIR] nvt [IS_PARQUET_FORMAT] [IS_CRITEO_MODE] [IS_FEATURE_CROSSED]"
    exit 2
	fi
	echo "Preprocessing script: NVTabular"
else
	echo "Script type must be nvt"
	exit 2
fi

SCRIPT_TYPE=$3

echo "Getting the first few examples from the uncompressed dataset..."
mkdir -p $DST_DATA_DIR/train                         && \
mkdir -p $DST_DATA_DIR/val                           && \
head -n 5000000 day_$1 > $DST_DATA_DIR/day_$1_small
if [ $? -ne 0 ]; then
	echo "Warning: fallback to find original compressed data day_$1.gz..."
	echo "Decompressing day_$1.gz..."
	gzip -d -c day_$1.gz > day_$1
	if [ $? -ne 0 ]; then
		echo "Error: failed to decompress the file."
		exit 2
	fi
	head -n 5000000 day_$1 > $DST_DATA_DIR/day_$1_small
	if [ $? -ne 0 ]; then
		echo "Error: day_$1 file"
		exit 2
	fi
fi

echo "Counting the number of samples in day_$1 dataset..."
total_count=$(wc -l $DST_DATA_DIR/day_$1_small)
total_count=(${total_count})
echo "The first $total_count examples will be used in day_$1 dataset."

echo "Shuffling dataset..."
shuf $DST_DATA_DIR/day_$1_small > $DST_DATA_DIR/day_$1_shuf

train_count=$(( total_count * 8 / 10))
valtest_count=$(( total_count - train_count ))
val_count=$(( valtest_count * 5 / 10 ))
test_count=$(( valtest_count - val_count  ))

split_dataset()
{
	echo "Splitting into $train_count-sample training, $val_count-sample val, and $test_count-sample test datasets..."
	head -n $train_count $DST_DATA_DIR/$1 > $DST_DATA_DIR/train/train.txt          && \
	tail -n $valtest_count $DST_DATA_DIR/$1 > $DST_DATA_DIR/val/valtest.txt        && \
	head -n $val_count $DST_DATA_DIR/val/valtest.txt > $DST_DATA_DIR/val/val.txt   && \
	tail -n $test_count $DST_DATA_DIR/val/valtest.txt > $DST_DATA_DIR/val/test.txt

	if [ $? -ne 0 ]; then
		exit 2
	fi
}

echo "Preprocessing..."
if [[ $SCRIPT_TYPE == "nvt" ]]; then
	IS_PARQUET_FORMAT=1
	IS_CRITEO_MODE=$4
	FEATURE_CROSS_LIST_OPTION=""
	if [[ ( $IS_CRITEO_MODE -eq 0 ) && ( $6 -eq 1 ) ]]; then
		FEATURE_CROSS_LIST_OPTION="--feature_cross_list C1_C2,C3_C4"
		echo $FEATURE_CROSS_LIST_OPTION
	fi
  split_dataset day_$1_shuf
  python3 criteo_script/preprocess_nvt.py \
		--data_path $DST_DATA_DIR             \
		--out_path $DST_DATA_DIR              \
		--freq_limit 6                        \
		--device_limit_frac 0.5               \
		--device_pool_frac 0.5                \
		--out_files_per_proc 8                \
		--devices "0"                         \
		--num_io_threads 2                    \
        --parquet_format=$IS_PARQUET_FORMAT   \
		--criteo_mode=$IS_CRITEO_MODE         \
		$FEATURE_CROSS_LIST_OPTION
fi

if [ $? -ne 0 ]; then
	exit 2
fi

echo "All done!"


