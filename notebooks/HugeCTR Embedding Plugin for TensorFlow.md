 HugeCTR Embedding  Plugin for TensorFlow
===================================

This notebook introduces a TensorFlow (TF) plugin for the HugeCTR embedding layer, embedding_plugin, where users may benefit from both the computational efficiency of the HugeCTR embedding layer and the ease of use of TensorFlow (TF).
 
To use this notebook, we recommend that you use our dev.tfplugin.Dockerfile docker image. For additional information, see notebooks/README.md#1-requirements.

## Build HugeCTR

Before you can build the embedding_plugin, you must first build HugeCTR. You can do so by running the following commands:
```
$ cd hugectr
$ mkdir -p build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DSM=70 .. # Target is NVIDIA V100
$ make -j
```

A dynamic library is generated in the `lib/` directory that you'll have to load using TensorFlow. You can directly import hugectr.py, where we prepare the codes to load that dynamic library and wrap some operations for convenient usage, in your python script to be used with the embedding_plugin.

## Verify Accuracy 
To verify whether the embedding_plugin can obtain the correct result, you can generate synthetic data for testing purposes as shown below.

In HugeCTR, the corresponding dense shape of the input keys is [batch_size, slot_num, max_nnz], and 0 is a valid key. Therefore, -1 is used to denote invalid keys, which only occupy that position in the corresponding dense keys matrix.

The results from embedding_plugins and original TF ops are consistent in both first and second forward propagation, which means the embedding_plugin can get the same forward result and perform the same backward propagation as TF ops. Therefore, the embedding_plugin can obtain the correct results.

### DeepFM Demo

In this notebook, TF 2.x is used to build the DeepFM model.

#### Define Models with the Embedding_Plugin

To proceed, Kernel must be restarted.

The above cells wrap the embedding_plugin ops into a TF layer, and uses that layer to define a TF DeepFM model. Similarly, define an embedding layer with TF original ops, and define a DeepFM model with that layer. Because embedding_plugin supports model parallelism, the parameters of the original TF embedding layer are equally distributed to each GPU for a fair performance comparison.

#### Define Models with the Original TF Ops

Dataset is needed to use these models for training. Kaggle Criteo datasets provided by CriteoLabs is used as the training dataset. The original training set contains 45,840,617 examples. Each example contains a label (0 by default or 1 if the ad was clicked) and 39 features in which 13 of them are integer and the other 26 are categorial. Since TFRecord is suitable for the training process and the Criteo dataset is missing numerous values across the feature columns, preprocessing is needed. The original test set won't be used because it doesn't contain labels.

#### Process Dataset

1. Download the dataset from [http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).

2. Extract the dataset by running the following command:
   ```
   $ tar zxvf dac.tar.gz
   ```

3. Preprocess the datast and set missing values.
   Preprocessing functions are defined in preprocess.py. Open that file and check the codes.

4. Split the dataset by running the following commands:
   ```
   $ head -n 36672493 train.out.txt > train
   $ tail -n 9168124 train.out.txt > valtest
   $ head -n 4584062 valtest > val
   $ tail -n 4584062 valtest > test
   ```

5. Convert the dataset into a TFRecord file.
   Converting functions are defined in txt2tfrecord.py. Open that file and check the codes.
   After the data preprocessing is completed, *.tfrecord file(s) will be generated, which can be used for training. The training loop can now be configured to use the dataset and models to perform the training.

#### Define training loop and do training

In read_data.py, some preprocessing and TF data reading pipeline creation functions are defined.

## API signature

All embedding_plugin APIs are defined in hugectr.py.

This function is used to create resource manager, which manages resources used by embedding_plugin. 
**IMPORTANT**: This function can only be called once. It must be called before any other embedding_plugin API is called Currently, only key_type='int64', value_type='float' has been tested.

This function can be used to do forward propagation for distributed and localized embedding layers. It will use all input keys that are stored in the SparseTensor format as its input, and will convert those keys to the CSR format within this function. Therefore, its performance is not very satisfying.

This function can be used to do forward propagation for distributed and localized embedding layers. Its inputs has been previously converted to the CSR format. Therefore, no conversion will be conducted within this function. For example, if the embedding_plugin uses four GPUs to perform the computation, then four CSR sparse matrices will be needed for its inputs. Addtionally, four row_offsets are stacked together to form a single tensor and value_tensors.

This function is used to save the embedding_plugin parameters in the file.

This function is used to restore the embedding_plugin parameters from file.