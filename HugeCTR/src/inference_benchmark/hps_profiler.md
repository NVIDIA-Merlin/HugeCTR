<!--
# Copyright (c) 2023, NVIDIA CORPORATION.
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
-->

# HPS Profiler

A critical part of optimizing the inference performance of HPS
is being able to measure changes in performance as you experiment with
different optimization strategies and data distribution. The hps_profiler application performs benchmark tasks for the Hierarchical Parameter Server. The hps_profiler will be compiled and installed from the following instructions in [Build and install the HPS Profiler](#section-1).


The hps_profiler application generates inference requests to HPS and measures the throughput and latency of different components, such as embedding cache, Database Backend and Lookup session. To
get representative results, hps_profiler measures the throughput and
latency over the configurable iteration, and then repeats the measurements until it reaches a specified number of iterations. 
For example, if `--embedding_cache` is used the results will be show below:

```
$ hps_profiler --iterations 1000 --num_key 2000 --powerlaw --alpha 1.2 --config /hugectr/model/ps.json --table_size 630000 --warmup_iterations 100   --embedding_cache

...
*** Measurement Results ***
  The Benchmark of: Apply for workspace from the memory pool for Embedding Cache Lookup
Latencies [900 iterations] min = 0.000285ms, mean = 0.000384853ms, median = 0.000365ms, 95% = 0.000428ms, 99% = 0.000465ms, max = 0.009736ms, throughput = 2.73973e+06/s
The Benchmark of: Copy the input to workspace of Embedding Cache
Latencies [900 iterations] min = 0.010842ms, mean = 0.0117076ms, median = 0.011596ms, 95% = 0.012219ms, 99% = 0.016642ms, max = 0.027379ms, throughput = 86236.6/s
The Benchmark of: Deduplicate the input embedding key for Embedding Cache
Latencies [900 iterations] min = 0.019159ms, mean = 0.0272492ms, median = 0.027262ms, 95% = 0.028104ms, 99% = 0.029548ms, max = 0.052309ms, throughput = 36681.1/s
The Benchmark of: Lookup the embedding keys from Embedding Cache
Latencies [900 iterations] min = 0.178875ms, mean = 0.231377ms, median = 0.227815ms, 95% = 0.267493ms, 99% = 0.284738ms, max = 0.47672ms, throughput = 4389.53/s
The Benchmark of: Merge output from Embedding Cache
Latencies [900 iterations] min = 0.007656ms, mean = 0.00850756ms, median = 0.008434ms, 95% = 0.009117ms, 99% = 0.011863ms, max = 0.018697ms, throughput = 118568/s
The Benchmark of: Missing key synchronization insert into Embedding Cache
Latencies [900 iterations] min = 0.105163ms, mean = 0.15741ms, median = 0.153763ms, 95% = 0.192302ms, 99% = 0.208846ms, max = 0.402043ms, throughput = 6503.52/s
The Benchmark of: Native Embedding Cache Query API
Latencies [900 iterations] min = 0.021729ms, mean = 0.0227739ms, median = 0.02253ms, 95% = 0.023695ms, 99% = 0.025035ms, max = 0.043024ms, throughput = 44385.3/s
The Benchmark of: decompress/deunique output from Embedding Cache
Latencies [900 iterations] min = 0.011247ms, mean = 0.0121274ms, median = 0.011953ms, 95% = 0.013055ms, 99% = 0.014706ms, max = 0.022186ms, throughput = 83661/s
The Benchmark of: The hit rate of Embedding Cache
Occupancy [900 iterations] min = 0.719323, mean = 0.843972, median = 0.854749, 95% = 0.894188, 99% = 0.90276, max = 0.918169
```
<a id="section-1"></a>
## Build and install the HPS Profiler
To build HPS profiler from source, do the following:
2. Download the HugeCTR repository and the third-party modules that it relies on by running the following commands:
```shell
   $ git clone https://github.com/NVIDIA/HugeCTR.git
   $ cd HugeCTR
   $ git submodule update --init --recursive
```

2. Pull the NGC Docker and run it

Pull the container using the following command:

```shell
docker pull nvcr.io/nvidia/merlin/merlin-hugectr:23.03
```

Launch the container in interactive mode (mount the HugeCTR root directory into the container for your convenience) by running this command:

   ```shell
   docker run --gpus all --rm -it --cap-add SYS_NICE --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -u root -v $(pwd):/HugeCTR -w /HugeCTR -p 8888:8888 nvcr.io/nvidia/merlin/merlin-hugectr:23.03
   ```  

3. Here is an example of how you can build HPS Profiler  using the build options:
   ```shell
   $ mkdir -p build && cd build
   $ cmake -DCMAKE_BUILD_TYPE=Release -DSM="70;80" -DENABLE_INFERENCE=ON -DENABLE_PROFILER=ON .. # Target is NVIDIA V100 / A100 with Inference mode ON.
   $ make -j && make install
   ```
4. You will get `hps_profiler` under bin foler.

## Create a synthetic embedding table
The embedding generator is used to generate synthetic HugeCTR sparse model files that can be loaded into HugeCTR [HPS](https://nvidia-merlin.github.io/HugeCTR/main/hierarchical_parameter_server/index.html) for inference. To generate a HugeCTR embedding file, please refer to the [Model generator ](../../../tools/inference_test_scripts/README.md#Inference-test-scripts)

## Use the HPS Profiler to get the measurement results
1. Generate HPS json configuration file based on synthetic model file.
For configuration information about HPS, you can refer to [here](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_parameter_server.html#configuration). Here is an example:

```json
{
	"supportlonglong": true,
	"models": [{
			"model": "model_name",
			"sparse_files": ["The path of synthetic embedding files"],
			"dense_file": "",
			"network_file": "",
			"num_of_worker_buffer_in_pool": 2,
			"num_of_refresher_buffer_in_pool":1,
			"deployed_device_list":[0],
			"max_batch_size":1024,
			"default_value_for_each_table":[0.0],
			"cache_refresh_percentage_per_iteration":0.1,
			"hit_rate_threshold":1.0,
			"gpucacheper":0.9,
			"gpucache":true,
			"maxnum_des_feature_per_sample": 0,
			"maxnum_catfeature_query_per_table_per_sample" : [26],
			"embedding_vecsize_per_table" : [16]

		}
	]
}
```
*`NOTE`*: The product of the `max_batch_size` size and the `maxnum_catfeature_query_per_table_per_sample` needs to be greater than or equal to the `--num_key` option in the hps_profiler.

2. Add arguments to hps_profiler for benchmark
```
$ hps_profiler 
--config: required.
Usage: HPS_Profiler [options] 

Optional arguments:
-h --help                       shows help message and exits [default: false]
-v --version                    prints version information and exits [default: false]
--config                        The path of the HPS json configuration file [required]
--powerlaw                      Generate the queried key that  in each iteration based on the power distribution [default: false]
--table_size                    The number of keys in the embedded table [default: 100000]
--alpha                         Alpha of power distribution [default: 1.2]
--hot_key_percentage            Percentage of hot keys in embedding tables [default: 0.2]
--hot_key_coverage              The probability of the hot key in each iteration [default: 0.8]
--num_key                       The number of keys to query for each iteration [default: 1000]
--iterations                    The number of iterations of the test [default: 1000]
--warmup_iterations             Performance results in warmup stage will be discarded [default: 0]
--embedding_cache               Enable embedding cache profiler, including the performance of lookup, insert, etc. [default: false]
--database_backend              Enable database backend profiler, which is to get the lookup performance of VDB/PDB [default: false]
--refresh_embeddingcache        Enable refreshing embedding cache. If the embedding cache tool is also enabled, the refresh will be performed asynchronously [default: false]
--lookup_session                Enable lookup_session profiler, which is E2E profiler, including embedding cache and data backend query delay [default: false]
```

Measurement example of the HPS Lookup Session
```
$hps_profiler --iterations 1000 --num_key 2000 --powerlaw --alpha 1.2 --config /hugectr/Model_Samples/wdl/wdl_infer/model/ps.json --table_size 630000 --warmup_iterations 100   --lookup_session
...
*** Measurement Results ***
The Benchmark of: End-to-end lookup embedding keys for Lookup session
Latencies [900 iterations] min = 0.190813ms, mean = 0.243117ms, median = 0.238085ms, 95% = 0.283761ms, 99% = 0.346377ms, max = 0.511712ms, throughput= 4200.18/s
```

Measurement example of the HPS Data Backend
```
$hps_profiler --iterations 1000 --num_key 2000 --powerlaw --alpha 1.2 --config /hugectr/Model_Samples/wdl/wdl_infer/model/ps.json --table_size 630000 --warmup_iterations 100   --database_backend
...
*** Measurement Results ***
The Benchmark of: Lookup the embedding key from default HPS database Backend
Latencies [900 iterations] min = 0.075086ms, mean = 0.127312ms, median = 0.121235ms, 95% = 0.166826ms, 99% = 0.219295ms, max = 0.285409ms, throughput = 8248.44/s
```
*`NOTE`*:  
1. If the user add the `--powerlaw` option, the queried embedding key will be generated with the specified argument `--alpha = **`.
2. If the user add the `--hot_key_percentage=**` and `--hot_key_coverage=xx` options, the queried embedding key  will generate the number of `--table_size` * `--hot_key_percentage` keys with this probability of `--hot_key_percentage=**`. 
For example `--hot_key_percentage=0.01`,  `--hot_key_coverage=0.9` and `--table_size=1000`, then the first 1000*0.01=10 keys will appear in the request with a probability of 90%.
3. It is recommended that users make mutually exclusive selections of three components(`--embedding_cache`,`--database_backend` and `--lookup_session`) to ensure the most accurate performance. Because the measurement results of the lookup session will include the performance results of the database backend and embedding cache.
4. If enable the [static embedding table](https://github.com/NVIDIA-Merlin/HugeCTR/blob/main/docs/source/hugectr_parameter_server.md#inference-parameters-and-embedding-cache-configuration) in HPS json file, the hps_profiler does not support the refresh operation.
