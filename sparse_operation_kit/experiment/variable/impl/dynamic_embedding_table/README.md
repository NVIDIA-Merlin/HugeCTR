# DynamicEmbeddingTable

`DynamicEmbeddingTable` is a **MOD** to the open-source, GPU-accelerated concurrent data structures, [cuCollections](https://github.com/NVIDIA/cuCollections).
It is dedicated to embedding storage and has many embedding-specific features, e.g.

- Fused lookup/update for variable embedding dimension, with ragged input data and output data
- Device-side on demand parameters initialization
- Fully sync-free API (also with dynamic memory growth), can specify cudaStream
- Transactional lookup/update, guarantee correctness for duplicate keys

Currently the `DynamicEmbeddingTable` provides 4 APIs

- lookup
- scatter_add
- remove
- eXport

![ragged.svg](/uploads/76512ad08a847f8db687c414469c1587/ragged.svg)

## Performance

Environment:

- NVIDIA A100-SXM4-80GB (400W)
- nvidia/cuda:11.4.2-devel-ubuntu20.04

Parameters:

- keys per action: 1M
- keys space: 50M
- Value dimension: 4

|API|Default initial capacity|Sufficient initial capacity|
|---|---|---|
|lookup|0.80 billion keys per second|1.56 billion keys per second|
|scatter_add|1.25 billion keys per second|3.89 billion keys per second|
|remove|1.27 billion keys per second|4.33 billion keys per second|

## Requirements

- CUDA 11.4.2 or higher
- C++17
- CMake 3.18 or higher
- Volta+

## Dependencies

`DynamicEmbeddingTable` depends on the following libraries:

- [libcu++](https://github.com/NVIDIA/libcudacxx)
- [CUB](https://github.com/thrust/cub)

## Building DynamicEmbeddingTable

To build the tests, benchmarks, and examples:

```bash
cd $DYNAMIC_EMBEDDING_TABLE_ROOT
mkdir -p build
cd build
cmake .. 
make
```

Binaries will be built into:

- `build/bin/`

## Contact

Gems Guo (gemsg@nvidia.com)
