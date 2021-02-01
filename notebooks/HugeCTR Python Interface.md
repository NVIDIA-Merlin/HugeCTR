HugeCTR Python Interface
===================================

In HugeCTR version 2.3, we've integrated the Python interface, which supports setting data source and model oversubscribing during training. This notebook explains how to access and use the HugeCTR Python interface.

## Table of Contents
* [Building the HugeCTR Python Interface](#building-the-hugectr-python-interface)
* [Wide & Deep Demo](#wide-&-deep-demo)
* [API Signatures](#api-signatures)

## Building the HugeCTR Python Interface ##
To build the HugeCTR Python interface: 

1. Enter the HugeCTR docker container and run the following commands:
   ```
   $ cd hugectr
   $ mkdir -p build && cd build
   $ cmake -DCMAKE_BUILD_TYPE=Release -DSM=70 .. # Target is NVIDIA V100
   $ make -j
   ```

   A dynamic link to the `hugectr.so` library is generated in the `hugectr/build/lib/` folder as shown here:
   ```
   !ls /hugectr/build/lib
   ```

2. Copy `hugectr.so` to the folder where you want to use the Python interface. 
   You can also install it to /usr/local/hugectr/lib and set the environment variable export to `PYTHONPATH=/usr/local/hugectr/lib:$PYTHONPATH` if you want to use the Python interface within the docker container environment.

3. Import HugeCTR and train your model using with Python as shown here:
   ```
   import hugectr
   ```

## Wide & Deep Demo

### Download and Preprocess Data

1. Download the Kaggle Criteo dataset using the following command:
   ```
   $ wget https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz
   ```

   For additional information, see [http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).
   
2. Extract the dataset using the following command:
   ```
   $ tar zxvf dac.tar.gz
   ```

3. Preprocess the data using  the following commands:
   ```
   $mkdir wdl_data
   $shuf train.txt > train.shuf.txt
   $python3 /hugectr/tools/criteo_script/preprocess.py --src_csv_path=train.shuf.txt --dst_csv_path=wdl_data/train.out.txt --normalize_dense=1 --feature_cross=1
   ```

4. Split the dataset using the following commands:
   ```
   head -n 36672493 wdl_data/train.out.txt > wdl_data/train && \\
   tail -n 9168124 wdl_data/train.out.txt > wdl_data/valtest && \\
   head -n 4584062 wdl_data/valtest > wdl_data/val && \\
   tail -n 4584062 wdl_data/valtest > wdl_data/test
   ```
5. Convert the dataset to the HugeCTR Norm dataset format by generating a file_list.*.txt and file_list.*.keyset as well as all training data (*.data) so that the features of the designated source can be employed during training such as model prefetch during training using the following commands:

   ```
   %%writefile criteo2hugectr.sh
   mkdir -p wdl_data_hugectr/wdl_data_bin && \
   cd wdl_data_hugectr && \
   cp /hugectr/build/bin/criteo2hugectr ./ &&
   ./criteo2hugectr /wdl_data/train wdl_data_bin/ file_list.txt 2 100
   ```
   
   ```
   !bash criteo2hugectr.sh
   ```

### Train from Scratch
We can train fom scratch and store the trained dense model and embedding tables in model files by doing the following: 

1. Create a JSON file for the W&D model. 
   **NOTE**: Please note that the solver clause no longer needs to be added to the JSON file when using the Python interface. Instead, you can configure the parameters using `hugectr.solver_parser_helper()` directly in the Python interface.

2. Write the Python script. 
   Ensure that the `repeat_dataset` parameter is set to `False` within the script, which indicates that the file list needs to be specified before submitting the sess.train() or sess.evaluation() calls. Additionally, be sure to create a write-enabled directory for storing the temporary files for model oversubscribing.

### Train from the Stored Model
Check the stored model files that will be used in the training. Dense model file embeddings should be passed to the respective model_file and embedding_files when calling `sess.solver_parser_helper()`. We will use the same JSON file and training data as the previous section. Also, all the other configurations for `solver_parser_helper` will also be the same.

## API Signatures
Here's list of all the API signatures within the HugeCTR Python interface that you need to get familiar with to successfully train your own model. As you can see from the above example, we've included Session, DataReader, ModelPrefetcher, and solver_parser_helper.

