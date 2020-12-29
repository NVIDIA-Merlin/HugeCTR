# Deploy HugeCTR Model on Triton TensorFlow Backend
The tutorial demonstrates how to deploy the trained HugeCTR model on Triton TensorFlow Backend.

## Save HugeCTR Model
After [building HugeCTR](../../../README.md#2-build-docker-image-and-hugectr), you can [train an example model](../../../README.md#4-train-an-example-dcn-model). It is easy to save the trained model to files in HugeCTR. The binary executable will save the trained models (i.e., dense model and sparse models) according to the `snapshot` and `snapshot_prefix` we set in json files. For example, if we set `snapshot` to be `10000` and `snapshot_prefix` to be `"./"`, we could get model files `_dense_10000.model` and `0_sparse_10000.model` after 10000 training iterations of an example Criteo model.

## Dump to TensorFlow and Export as graphdef
We have a [tutorial](../dump_to_tf) on dumping the saved model to TensorFlow, which currently supports Criteo and DCN models. To make things easier, we provide a [python script](./hugectr_tf/1/hugectr_tf.py) which creates the computational model graph, dumps the parameters in the model files to the graph and exports the graph to the formats that Triton TensorFlow Backend requires. You can convert the model files to `.graphdef` format by running:
```bash
cd hugectr_tf/1
python3 hugectr_tf.py --model <criteo|dcn> --json_file <*.json> --dense_model <.model> --sparse_models <.model> ...
```
Parameters:
+ `--model`: Specify model type, `criteo` or `dcn`.
+ `--json_file`: Specify the json file with which the model files are trained.
+ `--dense_model`: Specify the path of dense model.  
+ `--sparse_models`: Specify the path of sparse model(s) in the order of embedding(s) in json file.
We could get the `hugectr_tf.graphdef`, which can be deployed on Triton based on its TensorFlow Backend.

## Deploy on Triton
1. Exit the HugeCTR docker container:

2. Pull the NGC docker image of tritonserver:
```bash
docker pull nvcr.io/nvidia/tritonserver:20.09-py3
```

3. Start the triton inference server:
```bash
docker run --gpus=1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /absolute/path/to/triton_tf_deploy:/models nvcr.io/nvidia/tritonserver:20.09-py3 tritonserver --model-repository=/models
```
Please make sure to change `/absolute/path/to/triton_tf_deploy` according to the absolute path of this folder in your system.

After you start Triton you will see output on the console showing the server starting up and loading the model. When you see output like the following, Triton is ready to accept inference requests.
```bash
I1119 04:24:47.951902 1 model_repository_manager.cc:896] successfully loaded 'hugectr_tf' version 1
I1119 04:24:47.953012 1 grpc_server.cc:3897] Started GRPCInferenceService at 0.0.0.0:8001
I1119 04:24:47.953311 1 http_server.cc:2705] Started HTTPService at 0.0.0.0:8000
I1119 04:24:47.994344 1 http_server.cc:2724] Started Metrics Service at 0.0.0.0:8002
```

**NOTE**
  + Model Repository Layout <br>
  The layout of the model repository `hugectr_tf` should be like this:
  ```bash
  hugectr_tf
  ├── 1
  │   ├── hugectr_tf.graphdef
  │   └── hugectr_tf.py
  └── config.pbtxt
  ```
  
  + Configuration Protobuf <br>
  If you are deploying the Criteo model, you can use `config.pbtxt` here. For DCN model, you need to change the `input` of `config.pbtxt` like this:
  ```bash
  input [
    {
        name: "INPUT0"
        data_type: TYPE_INT64
        dims: [26]
    },
    {
        name: "INPUT1"
        data_type: TYPE_FP32
        dims: [13]
    }
  ]
  ```
## Make Inference Requests
You can use the Triton client SDK to make inference requests to the running Triton server and get the corresponding responses.
1. Pull the NGC docker image of triton client SDK:
```bash
docker pull nvcr.io/nvidia/tritonserver:20.09-py3-clientsdk
```
2. Start the docker container on the client:
```bash
docker run --rm -it -v /absolute/path/to/client/examples:/client/examples nvcr.io/nvidia/tritonserver:20.09-py3-clientsdk
```

We provide two python examples, i.e., `hugectr_tf_criteo_client.py` and  `hugectr_tf_dcn_client.py`, based on Python client libraries of Triton to send inference request and receiving response. You should create a folder on the clinet for these examples and mount the directory `/absolute/path/to/client/examples` into the container. Remember to modify the IP address in the scripts according to the running Triton server to succesfully make inference requests:
```bash
root@3e84dc39688f:/client/examples# python3 hugectr_tf_criteo_client.py
INPUT0: [      2     365    5840    8106   11025  144676  150012  151981  152540
  156528  156541  156678  156873  157293  158801  159375  430985  609983
  610294  612740  622307  622941  627045  671224  675962  946084  948795
  949311  960691 1208214 1208284 1212984 1215053 1215132 1479013 1479030
 1479060 1550398 1550501]
OUTPUT0: [-1.2869172]
```