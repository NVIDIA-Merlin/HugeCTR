#!/bin/bash
set -e

task_num=${1}
file_num=${2}

horovodrun -np ${task_num} python ${file_num}
