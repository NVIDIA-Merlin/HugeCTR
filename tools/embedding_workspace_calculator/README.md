# Cal vocabulary_size_per_gpu and workspace_size_per_gpu_in_mb
This tool is used to help you configure proper `workspace_size_per_gpu_in_mb` required by [hugectr.SparseEmbedding](https://nvidia-merlin.github.io/HugeCTR/master/api/python_interface.html#sparseembedding). It will calculate least required `workspace_size_per_gpu_in_mb` based on:
1. `slot_size_array`: the cardinality array of input features.
2. `vvgpu`: gpu configuration.
3. `emb_vec_size`: the embedding vector size.
4. `optimizer`: optimizer type. Can be `adam`, `adagrad`, `momentumsgd`, `nesterov` and `sgd`
5. `optimzer update type`: optimizer update type. Can be `local`, `global`, `lazy_global`

The default configuration is:
```
vvgpu = [[0]]
emb_vec_size = 16
optimizer = "adam"
optimizer_update_type = "global"
slot_size_array = [39884, 39043, 17289, 7420, 20263, 3, 7120, 1543, 39884, 39043, 17289, 7420, 20263, 3, 7120, 1543, 63, 63, 39884, 39043, 17289, 7420, 20263, 3, 7120, 1543]
```
Please refer more details in [QAList.md#24](https://nvidia-merlin.github.io/HugeCTR/master/QAList.html#how-to-set-workspace-size-per-gpu-in-mb-and-slot-size-array)