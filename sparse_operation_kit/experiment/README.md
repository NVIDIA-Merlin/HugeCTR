# SOK Experiment

## API

* sok.init
* sok.lookup_sparse
* class sok.Variable
* class sok.DynamicVariable
* sok.filter_variables
* class sok.OptimizerWrapper
* class sok.optimizers.SGD
* [TODO] sok.lookup
* [TODO] [Forward Compatible] sok.Init
* [TODO] [Forward Compatible] class sok.All2AllDenseEmbedding
* [TODO] [Forward Compatible] class sok.Distributedembedding
* [TODO] [Forward Compatible] sok.split_embedding_variable_from_others
* [TODO] [Forward Compatible] class sok.OptimizerScope
* [TODO] [Forward Compatible] class sok.Saver

## Install

```bash
docker run --privileged --gpus=all -it --rm nvcr.io/nvidia/tensorflow:22.05-tf2-py3

# Update cmake (Required by 3g-embedding)
apt remove cmake
pip install --upgrade pip
pip install --upgrade cmake

cd hugectr/sparse_operation_kit
mkdir build
cd build
cmake -DSM=80 ..  # -DSM=70 for V100
make -j
make install
cp -r ../sparse_operation_kit /usr/local/lib/python3.8/dist-packages/
```
