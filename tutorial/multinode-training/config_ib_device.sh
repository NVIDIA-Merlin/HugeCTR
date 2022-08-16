#!/bin/bash

DEVICE_FOLDER="/dev/infiniband/"
IBDEVICES_STR=""
PREFIX="--device="
for file_name in ${DEVICE_FOLDER}/*; do
    temp_file=`basename $file_name`
    IBDEVICES_STR=$IBDEVICES_STR" "$PREFIX$temp_file
done 
export IBDEVICES=$IBDEVICES_STR
