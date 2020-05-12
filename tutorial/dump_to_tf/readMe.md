# Dump Model To TensorFlow
A tutorial of dumping HugeCTR's model to TensorFlow.

## Model File Names
There are two kind of binary model files dumped from HugeCTR: sparse model file (embedding) and dense model file. After training, HugeCTR will:
- save each sparse model (embedding) to **\<prefix>\<sparse-index>\_sparse_\<iteration>.model**, for example, ```dcnmodel0_sparse_10000.model```.
- and save dense model to **\<prefix>\_dense_\<iteration>.model**, for example, ```dcnmodel_dense_10000.model```.

## Model Format 
+ Sparse model file <br>
HugeCTR supports multiple embeddings in one network. Each embedding corresponds to a sparse model file.

  1. Distributed embedding <br>
For distributed embedding, HugeCTR stores keys and embedding features in the following order:
      ```
      key0, embedding_feature0, 
      key1, embedding_feature1, 
      key2, embedding_feature2,
      ...
      ```
      Each pair of **<key, embedding_feature>** has size in bytes = **sizeof(TypeHashKey) + sizeof(float) \* embedding_vec_size**.

  2. Localized embedding <br>
For localized embedding, HugeCTR stores keys, slot ids and embedding features in the following order:
      ```
      key0, slot_id0, embedding_feature0,
      key1, slot_id1, embedding_feature1,
      key2, slot_id2, embedding_feature2,
      ...
      ```
      Each pair of **<key, slot_id, embedding_feature>** has size in bytes = **sizeof(TypeHashKey) + sizeof(TypeHashValueIndex) + sizeof(float) \* embedding_vec_size**.

+ Dense model file <br>
Dense model's weights will be stored in the order of layers in configuration file. All values are of type `float`.
  ```
  Weights in Layer0,
  Weights in Layer1, 
  Weights in Layer2,
  ...
  ```
  The [non-training parameters](../../docs/hugectr_user_guide.md#no-trained-parameters) will be saved to a json file, such as ```moving-mean``` and ```moving-var``` in BatchNorm layer. <br>

  So far, the following layers have parameters needed to be saved, and the parameters in each layer are stored in the order in which the variables appear:
  1. BatchNorm <br>
      ```
      gamma, beta
      ```

  2. InnerProduct <br>
      ```
      weight, bias
      ```

  3. MultiCross <br>
      ```
      for i in num_layers:
        weight, bias
      ```

  4. Multiply <br>
      ```
      weight
      ```

**NOTE** <br>
These binary model files only store the values described above, without any other identifiers or headers, which means you can parse the weights from the model file, or write initial values into model file in order.

## How To Dump Model Files To TensorFlow
To achieve this, the whole process has the following steps:<br>
1. Train with HugeCTR to get model files. 
2. According to model configuration json file, manually build the same computing-graph using TensorFlow.
3. Load weights from model files to initialize corresponding layers in TensorFlow.
4. Then you can save it as TensorFlow checkpoint, fine-tune that network, or do something you like.

## Demo 
Take criteo dataset and DCN model as an example to demonstrate the steps. 

### Requirements
+ Python >= 3.6
+ TensorFlow 1.x or TensorFlow 2.x
+ numpy
+ struct (python package)
+ json (python package)

You can use these commands to run this demo:
```
$ cd hugectr/tutorial/dump_to_tf
$ python3 main.py \
../../samples/dcn/criteo/sparse_embedding0.data \
../../samples/dcn/_dense_20000.model \
../../samples/dcn/0_sparse_20000.model
```

**Usage** 
```
python3 main.py dataset dense_model sparse_model0 sparse_model1 ...
```
Arguments: <br>
```dataset```: data file used in HugeCTR training <br>
```dense_model```: HugeCTR's dense model file <br>
```sparse_model```: HugeCTR's sparse model file(s). Specify sparse model(s) in the order of embedding(s) in model json file.

### Steps
1. Train with HugeCTR to get model files.<br>
Follow the [instructions](../../samples/dcn/README.md) to get binary model files. 

2. According to model configuration json file, manually build the same computing-graph using TensorFlow.<br>
As shown in [main.py](./main.py), use Tensorflow to build each layer according to [model json file](../../samples/dcn/dcn.json). TensorFlow layers equivalent to those used in HugeCTR can be found in [hugectr_layers.py](./hugectr_layers.py). <br><br>
For simplicity, the ```input keys``` are directly used as the ```row-index``` of embedding-table to look up ```embedding features```. Therefore input keys have shape ```[batchsize, slot_num, max_nnz_per_slot]```. 
    ```
    For example:
    input_keys has shape [batchsize = 2, slot_num = 3, max_nnz_per_slot = 4]
    [[[5 17 24 26]
      [3 0  -1 -1]
      [1 18 29 -1]]
      
     [[4 16 23 -1]
      [2 0  3  -1]
      [1 -1 -1 -1]]]
      
    -1 represents invalid key, and its embedding feature is 0.
    ```

3. Load weights from model files to initialize corresponding layers in TensorFlow. <br>
[dump.py](./dump.py) is used to parse parameters from binary model files. Each parameter is parsed in order as described above. The parsed values can be used to initialize parameters defined in TensorFlow's layers.

4. Save it as TensorFlow checkpoint. <br>
After completing the above steps, the computing-graph will be saved as TensorFlow chekcpoint. Then you can convert this checkpoint to other formats you need, such as ```.pb, .onnx```.


