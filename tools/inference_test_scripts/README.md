# Inference test scripts #
This module contains several scripts that helps user do inference related tests and benchmarks. 

## Model generator
The model generator is used to generate synthetic HugeCTR sparse model files that can be loaded into HugeCTR [HPS](https://nvidia-merlin.github.io/HugeCTR/main/hierarchical_parameter_server/index.html) for inference. To generate a HugeCTR model file, please provide the number of uniques keys you need in this model file and the embedding vector size.

Use the following command to create a HugeCTR model with 2,000,000 unqiue keys and embedding size 128:

```shell
bash generate_model.sh synthetic_sparse_model 2000000 128
```

The above script will create a sparse model folder called `synthetic_sparse_model` where you can find `emb_vector` and `key` file under it.

**Note that the file size of an embedding table with 2,000,000 uniques keys and embedding size 128 is:
2,000,000 * 128 * sizeof(float) + 2,000,000 * sizeof(int_64) = 1,040,000,000 bytes (about 1 GB).