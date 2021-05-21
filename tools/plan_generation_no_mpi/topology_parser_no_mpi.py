# 
# Copyright (c) 2021, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


#!/usr/bin/env python3

from subprocess import run, PIPE
import numpy as np
import os
import json
#from mpi4py import MPI

def parse_conf(conf_file):
    with open(conf_file, "r") as f:
        conf = json.load(f)
        gpu_lists = conf["solver"]["gpu"]
        for layer in conf["layers"]:
            if layer["type"] == "LocalizedSlotSparseEmbeddingHash":
                plan_files = layer["plan_file"]
                break
    if isinstance(plan_files,list):
        assert(len(gpu_lists) == len(plan_files))
        return gpu_lists[0], plan_files[0]
    else:
        return gpu_lists, plan_files


def normalize_nv_link(lines):
    existed_link = {}
    for line in lines:
        fields = line.split("\t")[1:]
        for field in fields:
            if field != " X ":
                existed_link[field] = 1

    if len(existed_link) == 1:
        link = list(existed_link.keys())[0]
        if link.startswith("NV"):
            n = int(link[2:])
            if n > 2:
                result = []
                for line in lines:
                    result.append(line.replace(link, "NV1"))
                return result
    return lines
    

def filter_gpu(topology_lines, gpu_list):
    num_gpus = len(topology_lines)
    for gpu_id in gpu_list:
        assert(gpu_id < num_gpus)

    mat = np.empty((num_gpus, num_gpus+1), dtype = object)
    for i,line in enumerate(topology_lines):
        mat[i] = line.split("\t")[0:num_gpus+1]

    mat = mat[gpu_list, :]
    column_list = np.array(gpu_list) + 1;
    # the 0-th column is not used in get_topology_matrix() but should exist
    column_list = [0] + column_list.tolist();
    mat = mat[:, column_list]

    result = ["\t".join(row) for row in mat]
    return normalize_nv_link(result)


def get_topology_matrix(filename = "", gpu_list = None):
    if filename:
        with open(filename, "r") as file:
            lines = file.read().split('\n')
    else:
        process = run(["nvidia-smi", "topo", "-m"], stdout=PIPE, universal_newlines=True)
        process.check_returncode()
        lines = process.stdout.split('\n')

    topology_lines = [l for l in lines if l.find("GPU") == 0]
    if gpu_list is not None:
        topology_lines = filter_gpu(topology_lines, gpu_list)

    num_gpus = len(topology_lines)

    nvlink = False
    topology = []
    for line in topology_lines:
        if line.find("NV") >= 0:
            nvlink = True
        topology.append(line.split()[1:num_gpus+1])

    if not nvlink:
        return np.ones((num_gpus,num_gpus))

    topology_matrix = np.eye(num_gpus) * num_gpus

    for i in range(len(topology)):
        for j in range(len(topology[i])):
            item = topology[i][j]
            if item[:2] == "NV":
                topology_matrix[i,j] = int(item[2:])


    return topology_matrix


if __name__ == "__main__":
    topology_matrix = get_topology_matrix()
    print(topology_matrix)

    topology_matrix = get_topology_matrix("dgx1_topology.txt")
    print(topology_matrix)

    topology_matrix = get_topology_matrix("dgx1_topology.txt", [7,0,2,6])
    print(topology_matrix)
