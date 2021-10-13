# HugeCTR to ONNX Converter #

The HugeCTR to Open Neural Network Exchange (ONNX) converter (hugectr2onnx) is a python package that can convert HugeCTR models to ONNX. It can improve the compatibility of HugeCTR with other deep learning frameworks since ONNX serves as an open-source format for AI models.

To use the HugeCTR to ONNX converter, you need to prepare all HugeCTR model and graph configuration files. You'll have one binary dense model file that stores the weights of all the dense layers. The number of sparse model folders will be equal to the number of embedding tables, and each folder will contain embedding vectors and binary files for every corresponding key. All these model files will be saved automatically according to the `snapshot` and `snapshot_prefix` values in the `hugectr.Model.fit()` training API. A graph configuration JSON file is required to load these binary model files correctly, which can be derived with the `hugectr.Model.graph_to_json()` training API. 

For example:

```bash
wdl_model
├── wdl0_sparse_2000.model
│   ├── emb_vector
│   └── key
├── wdl1_sparse_2000.model
│   ├── emb_vector
│   └── key
├── wdl_dense_2000.model
└── wdl.json
```

The HugeCTR to ONNX converter will parse the graph configuration JSON file for each HugeCTR layer and construct an equivalent ONNX graph with ONNX operators. At the same time, the HugeCTR to ONNX converter will read the HugeCTR model files and upload the weights to the corresponding ONNX node if the HugeCTR layer has some weights. Sparse models don't necessarily have to be converted to ONNX since they are usually very large and the weight alignment can be easily compared to the dense model.

If you choose to convert both dense and sparse models to ONNX, the converted ONNX graph will take dense features and sparse keys as inputs and expose prediction results as the output. If you only convert the dense model to ONNX, then the converted ONNX graph will require dense features and sparse embedding vectors as the inputs. The following figure depicts the differences of the converted ONNX models in these two cases. The left diagram depicts what takes place when only converting a dense and sparse model. The red dotted frames represent lookup and reduction for sparse embedding vectors. The right diagram depicts what takes place when only converting a dense model.

<div align=center><img src ="readme_src/wdl_onnx.png" width="800"/></div>
<div align=center>Fig. 1: The ONNX Graph of a WDL Model</div>

<br/>

Refer to [HugeCTR Python Interface](../docs/python_interface.md) to get familiar with how to train and save your models. To learn how to train a HugeCTR model, convert it to ONNX, and make inference with ONNX Runtime, refer to the [HugeCTR to ONNX Demo Notebook](../notebooks/hugectr2onnx_demo.ipynb).

## Installing hugectr2onnx ##

You can install `hugectr2onnx` in one of the following ways:
* [Use Our NGC Container](#use-our-ngc-container)
* [Build from Source](#build-from-source)

### Use Our NGC Container ###

`hugectr2onnx` has already been installed on the `nvcr.io/nvidia/merlin/merlin-training:21.09` container. You can directly import this package by running the following command:

```python
import hugectr2onnx
```
    
### Build from Source ###

If you want to build `hugectr2onnx` from the souce code, run the following commands:

```shell
$ git clone https://github.com/NVIDIA/HugeCTR.git
$ cd HugeCTR/onnx_converter
$ python3 setup.py install
```

## Using the HugeCTR to ONNX Converter API ##

Use the **hugectr2onnx.converter.convert(\*args, \*\*kwargs)** API to convert HugeCTR models to ONNX and save them to a specified directory. This API requires graph configuration JSON and model files that were produced during training.

### Parameters ###

* **onnx_model_path (string)**: Path used to store the ONNX model.
* **graph_config (string)**: JSON file that contains the graph configurations for the HugeCTR model.
* **dense_model (string)**: File that contains the weights for the HugeCTR dense model.
* **convert_embedding (boolean)**: Indicates whether to convert the HugeCTR sparse models (optional).
* **sparse_models (List[str])**: Files for the HugeCTR sparse model (optional).
* **ntp_file (string)**: File that contains the non-trainable parameters for the HugeCTR model (optional).
* **graph_name (string)**: Graph name for the ONNX model (optional).

### Example ###

```python
hugectr2onnx.converter.convert(onnx_model_path = "wdl.onnx",
                            graph_config = "wdl.json",
                            dense_model = "wdl_dense_2000.model",
                            convert_embedding = True,
                            sparse_models = ["wdl0_sparse_2000.model", "wdl1_sparse_2000.model"])
```
