# Inference test scripts #
This module contains several scripts that helps user do inference related tests and benchmarks. 

## Model generator
The model generator is used to generate synthetic HugeCTR sparse model files that can be loaded into HugeCTR [HPS](https://nvidia-merlin.github.io/HugeCTR/main/hierarchical_parameter_server/index.html) for inference. To generate a HugeCTR model file, please provide the number of uniques keys you need in this model file and the embedding vector size.

Use the following command to create a HugeCTR model with 2,000,000 unqiue keys and embedding size 128:

```shell
model_generator 2000000 128 synthetic_sparse_model
```

The above script will create a sparse model folder called `synthetic_sparse_model` where you can find `emb_vector` and `key` file under it.

**Note that the file size of an embedding table with 2,000,000 uniques keys and embedding size 128 is:
2,000,000 * 128 * sizeof(float) + 2,000,000 * sizeof(int_64) = 1,040,000,000 bytes (about 1 GB).

## Request generator
The request generator is used to generate JSON format inference requests that can be used to feed into the [Triton Perf Analyzer](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md) to do performance testing.

Use the following command to generate inference request with dense dimension 13, two embedding tables with dim 2 and 26 and number of unique keys 10000, 250000, respectively. The generated request is located in `test_request.json`.

```shell
request_generator --output_path test_request.json --dense_dim 13 --sparse_dims 2,26 --num_unique_keys 10000,250000 
```

And below is a simple command to run the Triton Perf Analyzer for your reference:

```shell
perf_analyzer -m your_model_name --collect-metrics -f perf_output.csv --verbose-csv --input-data test_request.json
```

Argument list:
* `--output_path`: required, the path of the outputed JSON requests.
* `--dense_dim`: required, the number of dense features.
* `--sparse_dims`: required, the number of sparse dims for each table, seperated by `,`.
* `--num_unique_keys`: required, the number of unique keys in each table, seperated by `,`.
* `--int64_key`: whether to use int64 key type or int32 key type. Default value is true (int64).
* `--dense_name`: the input name of dense part, should be consistent to your Triton config. Default is DES.
* `--sparse_name`: the input name of sparse part, should be consistent to your Triton config. Default is CATCOLUMN.
* `--rowoffset_name`: the input name of row offset, should be consistent to your Triton config. Default is ROWINDEX.
* `--num_batch`: the number of batches. Default is 10.
* `--batch_size`: the number of samples in each batch. Default is 64.
* `--powerlaw`: whether to use powerlaw distribution for the generated keys. Default is true.
* `--alpha`: alpha parameter in the powerlaw distribution. Higher the alpha, skewer the distribution. Default is 1.2. Will be ignored if `--powerlaw` is set to be false.
* `--hot_key_percentage`: to control the percentage of hot keys in emebedding tables when powerlaw distribution is NOT used. Default is 0.2. Will be ignored if `--powerlaw` is set to be true.
* `--hot_key_coverage`: to control the probablity of hot keys appears in total occurance. Default is 0.8. Will be ignored if `--powerlaw` is set to be true.