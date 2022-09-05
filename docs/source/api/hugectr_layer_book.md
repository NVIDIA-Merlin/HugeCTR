# HugeCTR Layer Classes and Methods

```{contents}
---
depth: 2
local: true
backlinks: none
---
```

This document introduces different layer classes and corresponding methods in the Python API of HugeCTR. The description of each method includes its functionality, arguments, and examples of usage.

## Input Layer

```python
hugectr.Input()
```

`Input` layer specifies the parameters related to the data input. `Input` layer should be added to the Model instance first so that the following `SparseEmbedding` and `DenseLayer` instances can access the inputs with their specified names.

**Arguments**
* `label_dim`: Integer, the label dimension. 1 implies it is a binary label. For example, if an item is clicked or not. There is NO default value and it should be specified by users.

* `label_name`: String, the name of the label tensor to be referenced by following layers. There is NO default value and it should be specified by users.

* `dense_dim`: Integer, the number of dense (or continuous) features. If there is no dense feature, set it to 0. There is NO default value and it should be specified by users.

* `dense_name`: Integer, the name of the dense input tensor to be referenced by following layers. There is NO default value and it should be specified by users.

* `data_reader_sparse_param_array`: List[hugectr.DataReaderSparseParam], the list of the sparse parameters for categorical inputs. Each `DataReaderSparseParam` instance should be constructed with  `sparse_name`, `nnz_per_slot`, `is_fixed_length` and `slot_num`.
  * `sparse_name` is the name of the sparse input tensors to be referenced by following layers. There is NO default value and it should be specified by users.
  * `nnz_per_slot` is the maximum number of features for each slot for the specified spare input. The `nnz_per_slot` can be an `int` which means average nnz per slot so the maximum number of features per sample should be `nnz_per_slot * slot_num`. Or you can use List[int] to initialize `nnz_per_slot`, then the maximum number of features per sample should be `sum(nnz_per_slot)` and in this case, the length of the array `nnz_per_slot` should be the same with `slot_num`.
  * `is_fixed_length` is used to identify whether categorical inputs has the same length for each slot among all samples. If different samples have the same number of features for each slot, then user can set `is_fixed_length = True` and HugeCTR can use this information to reduce data transferring time.
  * `slot_num` specifies the number of slots used for this sparse input in the dataset.

**Example:**
```python
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 13, dense_name = "dense",
                        data_reader_sparse_param_array =
                            [hugectr.DataReaderSparseParam("data1", 1, True, 26)]))
```

```python
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 13, dense_name = "dense",
                        data_reader_sparse_param_array =
                            [hugectr.DataReaderSparseParam("wide_data", 2, True, 2),
                            hugectr.DataReaderSparseParam("deep_data", 2, True, 26)]))
```

## Sparse Embedding

**SparseEmbedding class**

```python
hugectr.SparseEmbedding()
```

`SparseEmbedding` specifies the parameters related to the sparse embedding layer. One or several `SparseEmbedding` layers should be added to the Model instance after `Input` and before `DenseLayer`.

**Arguments**
* `embedding_type`: The embedding type.
Specify one of the following values:
  * `hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash`
  * `hugectr.Embedding_t.LocalizedSlotSparseEmbeddingHash`
  * `hugectr.Embedding_t.LocalizedSlotSparseEmbeddingOneHot`

  For information about the different embedding types, see [Embedding Types Detail](./hugectr_layer_book.md#embedding-types-detail).
  This argument does not have a default value.
  You must specify a value.

* `workspace_size_per_gpu_in_mb`: Integer, the workspace memory size in megabyte per GPU.
This workspace memory must be big enough to hold all the embedding vocabulary and its corresponding optimizer state that is used during the training and evaluation.
To understand how to set this value, see [How to set workspace_size_per_gpu_in_mb and slot_size_array](../QAList.md#24-how-to-set-workspace_size_per_gpu_in_mb-and-slot_size_array).
This argument does not have a default value.
You must specify a value.

* `embedding_vec_size`: Integer, the embedding vector size.
This argument does not have a default value.
You must specify a value.

* `combiner`: String, the intra-slot reduction operation.
Specify `sum` or `mean`.
This argument does not have a default value.
You must specify a value.

* `sparse_embedding_name`: String, the name of the sparse embedding tensor.
This name is referenced by the following layers.
This argument does not have a default value.
You must specify a value.

* `bottom_name`: String, the number of the bottom tensor to consume with this sparse embedding layer.
Please note that the value should be a predefined sparse input name.
This argument does not have a default value.
You must specify a value.

* `slot_size_array`: List[int], specify the maximum key value from each slot.
It should be consistent with that of the sparse input.
This parameter is used in `LocalizedSlotSparseEmbeddingHash` and `LocalizedSlotSparseEmbeddingOneHot`.
The value you specify can help avoid wasting memory that is caused by an imbalanced vocabulary size.
For more information, see [How to set workspace_size_per_gpu_in_mb and slot_size_array](../QAList.md#24-how-to-set-workspace_size_per_gpu_in_mb-and-slot_size_array).
This argument does not have a default value.
You must specify a value.

* `optimizer`: OptParamsPy, the optimizer that is dedicated to this sparse embedding layer.
If you do not specify the optimizer for the sparse embedding, the sparse embedding layer adopts the same optimizer as dense layers.

## Embedding Types Detail
### DistributedSlotSparseEmbeddingHash Layer

The `DistributedSlotSparseEmbeddingHash` stores embeddings in an embedding table and gets them by using a set of integers or indices. The embedding table can be segmented into multiple slots or feature fields, which spans multiple GPUs and nodes. With `DistributedSlotSparseEmbeddingHash`, each GPU will have a portion of a slot. This type of embedding is useful when there's an existing load imbalance among slots and OOM issues.

**Important Notes**:

* In a single embedding layer, it is assumed that input integers represent unique feature IDs, which are mapped to unique embedding vectors.
All the embedding vectors in a single embedding layer must have the same size. If you want some input categorical features to have different embedding vector sizes, use multiple embedding layers.
* The input indices’ data type, `input_key_type`, is specified in the solver. By default,  the 32-bit integer (I32) is used, but the 64-bit integer type (I64) is also allowed even if it is constrained by the dataset type. For additional information, see [Solver](./python_interface.md#solver).

**Example:**
```python
model.add(hugectr.SparseEmbedding(
            embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
            workspace_size_per_gpu_in_mb = 23,
            embedding_vec_size = 1,
            combiner = 'sum',
            sparse_embedding_name = "sparse_embedding1",
            bottom_name = "input_data",
            optimizer = optimizer))
```

### LocalizedSlotSparseEmbeddingHash Layer

The `LocalizedSlotSparseEmbeddingHash` layer to store embeddings in an embedding table and get them by using a set of integers or indices. The embedding table can be segmented into multiple slots or feature fields, which spans multiple GPUs and nodes. Unlike the DistributedSlotSparseEmbeddingHash layer, with this type of embedding layer, each individual slot is located in each GPU and not shared. This type of embedding layer provides the best scalability.

**Important Notes**:

* In a single embedding layer, it is assumed that input integers represent unique feature IDs, which are mapped to unique embedding vectors.
All the embedding vectors in a single embedding layer must have the same size. If you want some input categorical features to have different embedding vector sizes, use multiple embedding layers.
* The input indices’ data type, `input_key_type`, is specified in the solver. By default, the 32-bit integer (I32) is used, but the 64-bit integer type (I64) is also allowed even if it is constrained by the dataset type. For additional information, see [Solver](./python_interface.md#solver).

Example:
```python
model.add(hugectr.SparseEmbedding(
            embedding_type = hugectr.Embedding_t.LocalizedSlotSparseEmbeddingHash,
            workspace_size_per_gpu_in_mb = 23,
            embedding_vec_size = 1,
            combiner = 'sum',
            sparse_embedding_name = "sparse_embedding1",
            bottom_name = "input_data",
            optimizer = optimizer))
```

### LocalizedSlotSparseEmbeddingOneHot Layer

The LocalizedSlotSparseEmbeddingOneHot layer stores embeddings in an embedding table and gets them by using a set of integers or indices. The embedding table can be segmented into multiple slots or feature fields, which spans multiple GPUs and nodes. This is a performance-optimized version of LocalizedSlotSparseEmbeddingHash for the case where NVSwitch is available and inputs are one-hot categorical features.

**Note**: LocalizedSlotSparseEmbeddingOneHot can only be used together with the Raw dataset format. Unlike other types of embeddings, LocalizedSlotSparseEmbeddingOneHot only supports single-node training and can be used only in a NVSwitch equipped system such as DGX-2 and DGX A100.
The input indices’ data type, `input_key_type`, is specified in the solver. By default, the 32-bit integer (I32) is used, but the 64-bit integer type (I64) is also allowed even if it is constrained by the dataset type. For additional information, see [Solver](./python_interface.md#solver).

Example:
```python
model.add(hugectr.SparseEmbedding(
            embedding_type = hugectr.Embedding_t.LocalizedSlotSparseEmbeddingOneHot,
            slot_size_array = [1221, 754, 8, 4, 12, 49, 2]
            embedding_vec_size = 128,
            combiner = 'sum',
            sparse_embedding_name = "sparse_embedding1",
            bottom_name = "input_data",
            optimizer = optimizer))
```

## Dense Layers

**DenseLayer class**

```python
hugectr.DenseLayer()
```

`DenseLayer` specifies the parameters related to the dense layer or the loss function. HugeCTR currently supports multiple dense layers and loss functions. Please **NOTE** that the final sigmoid function is fused with the loss function to better utilize memory bandwidth.

**Arguments**
* `layer_type`: The layer type to be used. The supported types include `hugectr.Layer_t.Add`, `hugectr.Layer_t.BatchNorm`, `hugectr.Layer_t.Cast`, `hugectr.Layer_t.Concat`, `hugectr.Layer_t.Dropout`, `hugectr.Layer_t.ELU`, `hugectr.Layer_t.FmOrder2`, `hugectr.Layer_t.FusedInnerProduct`, `hugectr.Layer_t.InnerProduct`, `hugectr.Layer_t.Interaction`, `hugectr.Layer_t.MultiCross`, `hugectr.Layer_t.ReLU`, `hugectr.Layer_t.ReduceSum`, `hugectr.Layer_t.Reshape`, `hugectr.Layer_t.Sigmoid`, `hugectr.Layer_t.Slice`, `hugectr.Layer_t.WeightMultiply`, `hugectr.Layer_t.ElementwiseMultiply`, `hugectr.Layer_t.GRU`, `hugectr.Layer_t.Scale`, `hugectr.Layer_t.FusedReshapeConcat`, `hugectr.Layer_t.FusedReshapeConcatGeneral`, `hugectr.Layer_t.Softmax`, `hugectr.Layer_t.PReLU_Dice`, `hugectr.Layer_t.ReduceMean`, `hugectr.Layer_t.Sub`, `hugectr.Layer_t.Gather`, `hugectr.Layer_t.BinaryCrossEntropyLoss`, `hugectr.Layer_t.CrossEntropyLoss` and `hugectr.Layer_t.MultiCrossEntropyLoss`. There is NO default value and it should be specified by users.

* `bottom_names`: List[str], the list of bottom tensor names to be consumed by this dense layer. Each name in the list should be the predefined tensor name. There is NO default value and it should be specified by users.

* `top_names`: List[str], the list of top tensor names, which specify the output tensors of this dense layer. There is NO default value and it should be specified by users.

* For details about the usage of each layer type and its parameters, please refer to [Dense Layers Usage](#dense-layers-usage).

## Dense Layers Usage

### FullyConnected Layer
The FullyConnected layer is a densely connected layer (or MLP layer). It is usually made of a `InnerProduct` layer and a `ReLU`.

Parameters:

* `num_output`: Integer, the number of output elements for the `InnerProduct` or `FusedInnerProduct` layer. The default value is 1.
* `weight_init_type`: Specifies how to initialize the weight array. The supported types include `hugectr.Initializer_t.Default`, `hugectr.Initializer_t.Uniform`, `hugectr.Initializer_t.XavierNorm`, `hugectr.Initializer_t.XavierUniform` and `hugectr.Initializer_t.Zero`. The default value is `hugectr.Initializer_t.Default`.
* `bias_init_type`: Specifies how to initialize the bias array for the `InnerProduct`, `FusedInnerProduct` or `MultiCross` layer. The supported types include `hugectr.Initializer_t.Default`, `hugectr.Initializer_t.Uniform`, `hugectr.Initializer_t.XavierNorm`, `hugectr.Initializer_t.XavierUniform` and `hugectr.Initializer_t.Zero`. The default value is `hugectr.Initializer_t.Default`.

Input and Output Shapes:

* input: (batch_size, *) where * represents any number of elements
* output: (batch_size, num_output)

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                            bottom_names = ["relu1"],
                            top_names = ["fc2"],
                            num_output=1024))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc2"],
                            top_names = ["relu2"]))
```

### FusedFullyConnected Layer

The FusedFullyConnected layer fuses a common case where FullyConnectedLayer and ReLU are used together to save memory bandwidth.

**Note**: This layer can only be used with Mixed Precision mode enabled.

* `num_output`: Integer, the number of output elements for the `InnerProduct` or `FusedInnerProduct` layer. The default value is 1.
* `weight_init_type`: Specifies how to initialize the weight array. The supported types include `hugectr.Initializer_t.Default`, `hugectr.Initializer_t.Uniform`, `hugectr.Initializer_t.XavierNorm`, `hugectr.Initializer_t.XavierUniform` and `hugectr.Initializer_t.Zero`. The default value is `hugectr.Initializer_t.Default`.
* `bias_init_type`: Specifies how to initialize the bias array for the `InnerProduct`, `FusedInnerProduct` or `MultiCross` layer. The supported types include `hugectr.Initializer_t.Default`, `hugectr.Initializer_t.Uniform`, `hugectr.Initializer_t.XavierNorm`, `hugectr.Initializer_t.XavierUniform` and `hugectr.Initializer_t.Zero`. The default value is `hugectr.Initializer_t.Default`.
Input and Output Shapes:

* input: (batch_size, *) where * represents any number of elements
* output: (batch_size, num_output)

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,
                            bottom_names = ["fc1"],
                            top_names = ["fc2"],
                            num_output=1024))
```

### MultiCross Layer

The MultiCross layer is a cross network where explicit feature crossing is applied across cross layers.

**Note**: This layer doesn’t currently support Mixed Precision mode.

Parameters:

* `num_layers`: Integer, number of cross layers in the cross network. It should be set as a positive number if you want to use the cross network. The default value is 0.
* `weight_init_type`: Specifies how to initialize the weight array. The supported types include `hugectr.Initializer_t.Default`, `hugectr.Initializer_t.Uniform`, `hugectr.Initializer_t.XavierNorm`, `hugectr.Initializer_t.XavierUniform` and `hugectr.Initializer_t.Zero`. The default value is `hugectr.Initializer_t.Default`.
* `bias_init_type`: Specifies how to initialize the bias array. The supported types include `hugectr.Initializer_t.Default`, `hugectr.Initializer_t.Uniform`, `hugectr.Initializer_t.XavierNorm`, `hugectr.Initializer_t.XavierUniform` and `hugectr.Initializer_t.Zero`. The default value is `hugectr.Initializer_t.Default`.

Input and Output Shapes:

* input: (batch_size, *) where * represents any number of elements
* output: same as input

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.MultiCross,
                            bottom_names = ["slice11"],
                            top_names = ["multicross1"],
                            num_layers=6))
```

### FmOrder2 Layer

TheFmOrder2 layer is the second-order factorization machine (FM), which models linear and pairwise interactions as dot products of latent vectors.

Parameters:

* `out_dim`: Integer, the output vector size. It should be set as a positive number if you want to use factorization machine. The default value is 0.

Input and Output Shapes:

* input: (batch_size, *) where * represents any number of elements
* output: (batch_size, out_dim)

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FmOrder2,
                            bottom_names = ["slice32"],
                            top_names = ["fmorder2"],
                            out_dim=10))
```

### WeightMultiply Layer

The Multiply Layer maps input elements into a latent vector space by multiplying each feature with a corresponding weight vector.

Parameters:

* `weight_dims`: List[Integer], the shape of the weight matrix (slot_dim, vec_dim) where vec_dim corresponds to the latent vector length for the `WeightMultiply` layer. It should be set correctly if you want to employ the weight multiplication. The default value is [].
* `weight_init_type`: Specifies how to initialize the weight array. The supported types include `hugectr.Initializer_t.Default`, `hugectr.Initializer_t.Uniform`, `hugectr.Initializer_t.XavierNorm`, `hugectr.Initializer_t.XavierUniform` and `hugectr.Initializer_t.Zero`. The default value is `hugectr.Initializer_t.Default`.

Input and Output Shapes:

* input: (batch_size, slot_dim) where slot_dim represents the number of input features
* output: (batch_size, slot_dim * vec_dim)

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.WeightMultiply,
                            bottom_names = ["slice32"],
                            top_names = ["fmorder2"],
                            weight_dims = [13, 10]),
                            weight_init_type = hugectr.Initializer_t.XavierUniform)
```

### ElementwiseMultiply Layer

The ElementwiseMultiply Layer maps two inputs into a single resulting vector by performing an element-wise multiplication of the two inputs.

Parameters: None

Input and Output Shapes:

* input: 2x(batch_size, num_elem)
* output: (batch_size, num_elem)

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ElementwiseMultiply,
                            bottom_names = ["slice1","slice2"],
                            top_names = ["eltmultiply1"])
```

### BatchNorm Layer

The BatchNorm layer implements a cuDNN based batch normalization.

Parameters:

* `factor`: Float, exponential average factor such as runningMean = runningMean*(1-factor) + newMean*factor for the `BatchNorm` layer. The default value is 1.
* `eps`: Float, epsilon value used in the batch normalization formula for the `BatchNorm` layer. The default value is 1e-5.
* `gamma_init_type`: Specifies how to initialize the gamma (or scale) array for the `BatchNorm` layer. The supported types include `hugectr.Initializer_t.Default`, `hugectr.Initializer_t.Uniform`, `hugectr.Initializer_t.XavierNorm`, `hugectr.Initializer_t.XavierUniform` and `hugectr.Initializer_t.Zero`. The default value is `hugectr.Initializer_t.Default`.
* `beta_init_type`: Specifies how to initialize the beta (or offset) array for the `BatchNorm` layer. The supported types include `hugectr.Initializer_t.Default`, `hugectr.Initializer_t.Uniform`, `hugectr.Initializer_t.XavierNorm`, `hugectr.Initializer_t.XavierUniform` and `hugectr.Initializer_t.Zero`. The default value is `hugectr.Initializer_t.Default`.

Input and Output Shapes:

* input: (batch_size, num_elem)
* output: same as input

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BatchNorm,
                            bottom_names = ["slice32"],
                            top_names = ["fmorder2"],
                            factor = 1.0,
                            eps = 0.00001,
                            gamma_init_type = hugectr.Initializer_t.XavierUniform,
                            beta_init_type = hugectr.Initializer_t.XavierUniform)
```

When training a model, each BatchNorm layer stores mean and variance in a JSON file using the following format:
“snapshot_prefix” + “_dense_” + str(iter) + ”.model”

Example: my_snapshot_dense_5000.model<br>

In the JSON file, you can find the batch norm parameters as shown below:
```json
    {
      "layers": [
        {
          "type": "BatchNorm",
          "mean": [-0.192325, 0.003050, -0.323447, -0.034817, -0.091861],
          "var": [0.738942, 0.410794, 1.370279, 1.156337, 0.638146]
        },
        {
          "type": "BatchNorm",
          "mean": [-0.759954, 0.251507, -0.648882, -0.176316, 0.515163],
          "var": [1.434012, 1.422724, 1.001451, 1.756962, 1.126412]
        },
        {
          "type": "BatchNorm",
          "mean": [0.851878, -0.837513, -0.694674, 0.791046, -0.849544],
          "var": [1.694500, 5.405566, 4.211646, 1.936811, 5.659098]
        }
      ]
    }
```
### LayerNorm Layer

The LayerNorm layer implements a layer normalization.

Parameters:

* `eps`: Float, epsilon value used in the batch normalization formula for the `LayerNorm` layer. The default value is 1e-5.
* `gamma_init_type`: Specifies how to initialize the gamma (or scale) array for the `LayerNorm` layer. The supported types include `hugectr.Initializer_t.Default`, `hugectr.Initializer_t.Uniform`, `hugectr.Initializer_t.XavierNorm`, `hugectr.Initializer_t.XavierUniform` and `hugectr.Initializer_t.Zero`. The default value is `hugectr.Initializer_t.Default`.
* `beta_init_type`: Specifies how to initialize the beta (or offset) array for the `LayerNorm` layer. The supported types include `hugectr.Initializer_t.Default`, `hugectr.Initializer_t.Uniform`, `hugectr.Initializer_t.XavierNorm`, `hugectr.Initializer_t.XavierUniform` and `hugectr.Initializer_t.Zero`. The default value is `hugectr.Initializer_t.Default`.

Input and Output Shapes:

* input: 2D: (batch_size, num_elem), 3D: (batch_size, seq_len, num_elem), 4D: (head_num, batch_size, seq_len, num_elem)
* output: same as input

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.LayerNorm,
                            bottom_names = ["slice32"],
                            top_names = ["fmorder2"],
                            eps = 0.00001,
                            gamma_init_type = hugectr.Initializer_t.XavierUniform,
                            beta_init_type = hugectr.Initializer_t.XavierUniform)
```

### Concat Layer

The Concat layer concatenates a list of inputs.

Parameters:
* `axis`:  Integer, the dimension to concat for the `Concat` layer. If the input is N-dimensional, 0 <= axis < N. The default value is 1.

Input and Output Shapes:

* input: 3D: {(batch_size, num_feas_0, num_elems_0), (batch_size, num_feas + 1, num_elems_1), ...} or 2D: {(batch_size, num_elems_0), (batch_size, num_elems_1), ...}
* output: 3D and axis=1: (batch_size, num_feas_0+num_feas_1+..., num_elems). 3D and axis=2: (batch_size, num_feas, num_elems_0+num_elems_1+...). 2D: (batch_size, num_elems_0+num_elems_1+...)

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,
                            bottom_names = ["reshape3","weight_multiply2"],
                            top_names = ["concat2"],
                            axis = 2))
```

### Reshape Layer

The Reshape layer reshapes a 3D input tensor into 2D shape.

Parameter:

* `leading_dim`: Integer, the innermost dimension of the output tensor. It must be the multiple of the total number of input elements. If it is unspecified, n_slots * num_elems (see below) is used as the default leading_dim.
* `time_step`: Integer, the second dimension of the 3D output tensor. It must be the multiple of the total number of input elements and must be defined with leading_dim.
* `selected`: Boolean, whether to use the selected mode for the `Reshape` layer. The default value is False.
* `selected_slots`: List[int], the selected slots for the `Reshape` layer. It will be ignored if `selected` is False. The default value is [].

Input and Output Shapes:

* input: (batch_size, n_slots, num_elems)
* output: (tailing_dim, leading_dim) where tailing_dim is batch_size * n_slots * num_elems / leading_dim

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                            bottom_names = ["sparse_embedding1"],
                            top_names = ["reshape1"],
                            leading_dim=416))
```

### Slice Layer

The Slice layer extracts multiple output tensors from a 2D input tensors.

Parameter:

* `ranges`: List[Tuple[int, int]], used for the Slice layer. A list of tuples in which each one represents a range in the input tensor to generate the corresponding output tensor. For example, (2, 8) indicates that 6 elements starting from the second element in the input tensor are used to create an output tensor. Note that the start index is inclusive and the end index is exclusive. The number of tuples corresponds to the number of output tensors. Ranges are allowed to overlap unless it is a reverse or negative range. The default value is [].

Input and Output Shapes:

* input: (batch_size, num_elems)
* output: {(batch_size, b-a), (batch_size, d-c), ....) where ranges ={[a, b), [c, d), …} and len(ranges) <= 5

Example:

You can apply the Slice layer to actually slicing a tensor. In this case, it must be explicitly added with Python API.
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Slice,
                            bottom_names = ["dense"],
                            top_names = ["slice21", "slice22"],
                            ranges=[(0,10),(10,13)]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.WeightMultiply,
                            bottom_names = ["slice21"],
                            top_names = ["weight_multiply1"],
                            weight_dims= [10,10]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.WeightMultiply,
                            bottom_names = ["slice22"],
                            top_names = ["weight_multiply2"],
                            weight_dims= [3,1]))
```

The Slice layer can also be employed to create copies of a tensor, which helps to express a branch topology in your model graph.
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Slice,
                            bottom_names = ["dense"],
                            top_names = ["slice21", "slice22"],
                            ranges=[(0,13),(0,13)]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.WeightMultiply,
                            bottom_names = ["slice21"],
                            top_names = ["weight_multiply1"],
                            weight_dims= [13,10]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.WeightMultiply,
                            bottom_names = ["slice22"],
                            top_names = ["weight_multiply2"],
                            weight_dims= [13,1]))
```

From HugeCTR v.3.3, the aforementioned, Slice layer based branching can be abstracted away. When the same tensor is referenced multiple times in constructing a model in Python, the HugeCTR parser can internally add a Slice layer to handle such a situation. Thus, the example below behaves as the same as the one above whilst simplifying the code.
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.WeightMultiply,
                            bottom_names = ["dense"],
                            top_names = ["weight_multiply1"],
                            weight_dims= [13,10]))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.WeightMultiply,
                            bottom_names = ["dense"],
                            top_names = ["weight_multiply2"],
                            weight_dims= [13,1]))
```

### Dropout Layer

The Dropout layer randomly zeroizes or drops some of the input elements.

Parameter:

* `dropout_rate`: Float, The dropout rate to be used for the `Dropout` layer. It should be between 0 and 1. Setting it to 0 indicates that there is no dropped element at all. The default value is 0.5.

Input and Output Shapes:

* input: (batch_size, num_elems)
* output: same as input

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                            bottom_names = ["relu1"],
                            top_names = ["dropout1"],
                            dropout_rate=0.5))
```

### ELU Layer

The ELU layer represents the Exponential Linear Unit.

Parameter:

* `elu_alpha`: Float, the scalar that decides the value where this ELU function saturates for negative values. The default value is 1.

Input and Output Shapes:

* input: (batch_size, *) where * represents any number of elements
* output: same as input

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ELU,
                            bottom_names = ["fc1"],
                            top_names = ["elu1"],
                            elu_alpha=1.0))
```

### ReLU Layer

The ReLU layer represents the Rectified Linear Unit.

Input and Output Shapes:

* input: (batch_size, *) where * represents any number of elements
* output: same as input

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                            bottom_names = ["fc1"],
                            top_names = ["relu1"]))
```

### Sigmoid Layer

The Sigmoid layer represents the Sigmoid Unit.

Input and Output Shapes:

* input: (batch_size, *) where * represents any number of elements
* output: same as input

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Sigmoid,
                            bottom_names = ["fc1"],
                            top_names = ["sigmoid1"]))
```
**Note**: The final sigmoid function is fused with the loss function to better utilize memory bandwidth, so do NOT add a Sigmoid layer before the loss layer.

### Interaction Layer

The interaction layer is used to explicitly capture second-order interactions between features.

Parameters: None

Input and Output Shapes:

* input: {(batch_size, num_elems), (batch_size, num_feas, num_elems)} where the first tensor typically represents a fully connected layer and the second is an embedding.
* output: (batch_size, output_dim) where output_dim = num_elems + (num_feas + 1) * (num_feas + 2 ) / 2 - (num_feas + 1) + 1

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Interaction,
                            bottom_names = ["layer1", "layer3"],
                            top_names = ["interaction1"]))
```

**Important Notes**:
There are optimizations that can be employed on the `Interaction` layer and the following `GroupFusedInnerProduct` layer during fp16 training. In this case, you should specify two output tensor names for the `Interaction` layer, and use them as the input tensors for the following `GroupFusedInnerProduct` layer. Please refer to the example of [GroupDenseLayer](#groupdenselayer) for the detailed usage.

### Add Layer

The Add layer adds up an arbitrary number of tensors that have the same size in an element-wise manner.

Parameters: None

Input and Output Shapes:

* input: Nx(batch_size, num_elems) where N is the number of input tensors
* output: (batch_size, num_elems)

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Add,
                            bottom_names = ["fc4", "reducesum1", "reducesum2"],
                            top_names = ["add"]))
```

### ReduceSum Layer

The ReduceSum Layer sums up all the elements across a specified dimension.

Parameter:

* `axis`:  Integer, the dimension to reduce for the `ReduceSum` layer. If the input is N-dimensional, 0 <= axis < N. The default value is 1.

Input and Output Shapes:

* input: (batch_size, ...) where ... represents any number of elements with an arbitrary number of dimensions
* output: Dimension corresponding to axis is set to 1. The others remain the same as the input.

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReduceSum,
                            bottom_names = ["fmorder2"],
                            top_names = ["reducesum1"],
                            axis=1))
```
#### GRU Layer

The GRU layer is Gated Recurrent Unit.

Parameters:

* `num_output`: Number of output elements.
* `batchsize`: Number of batchsize.
* `SeqLength`: Length of the sequence.
* `vector_size`: size of the input vector.
* `weight_init_type`: Specifies how to initialize the weight array. The supported types include `hugectr.Initializer_t.Default`, `hugectr.Initializer_t.Uniform`, `hugectr.Initializer_t.XavierNorm`, `hugectr.Initializer_t.XavierUniform` and `hugectr.Initializer_t.Zero`. The default value is `hugectr.Initializer_t.Default`.
* `bias_init_type`: Specifies how to initialize the bias array. The supported types include `hugectr.Initializer_t.Default`, `hugectr.Initializer_t.Uniform`, `hugectr.Initializer_t.XavierNorm`, `hugectr.Initializer_t.XavierUniform` and `hugectr.Initializer_t.Zero`. The default value is `hugectr.Initializer_t.Default`.

Input and Output Shapes:

* input: (1, batch_size*SeqLength*embedding_vec_size)
* output: (1, batch_size*SeqLength*embedding_vec_size)

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.GRU,
                            bottom_names = ["GRU1"],
                            top_names = ["conncat1"],
                            num_output=256,
                            batchsize=13,
                            SeqLength=20,
                            vector_size=20))
```

#### PReLUDice Layer

The PReLUDice layer represents the Parametric Rectified Linear Unit, which adaptively adjusts the rectified point according to distribution of input data.

Parameters:

* `elu_alpha`: A scalar that decides the value where this activation function saturates for negative values.
* `eps`: Epsilon value used in the PReLU/Dice formula.

Input and Output Shapes:

* input: (batch_size, *) where * represents any number of elements
* output: same as input

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.PReLU_Dice,
                            bottom_names = ["fc_din_i1"],
                            top_names = ["dice_1"],
                            elu_alpha=0.2, eps=1e-8))
```

#### Scale Layer

The Scale layer scales the input 2D tensor to specific size on the designate axis.

Parameters:

* `axis`: Along the designate axis to scale the tensor. The designate axis could be axis 0, 1.
* `factor `: scale factor.

Input and Output Shapes:

* input: (batch_size, num_elems)
* output: if axis = 0; (batch_size, num_elems * factor), if axis = 1; (batch_size * factor, num_elems)

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Scale,
                            bottom_names = ["item1"],
                            top_names = ["Scale_item"],
                            axis = 1, factor = 10))
```

#### FusedReshapeConcat Layer

The FusedReshapeConcat layer cross combines the input tensors and outputs item tensor, AD tensor.

Parameters: None

Input and Output Shapes:

* input: {(batch_size, num_feas + 1, num_elems_0), (batch_size, num_feas + 1, num_elems_1), ...}, the input tensors are embeddings.
* output: {(batch_size x num_feas, (num_elems_0 + num_elems_1 + ...)), (batch_size, (num_elems_0 + num_elems_1 + ...))}.

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedReshapeConcat,
                            bottom_names = ["sparse_embedding_good", "sparse_embedding_cate"],
                            top_names = ["FusedReshapeConcat_item_his_em", "FusedReshapeConcat_item"]))
```

#### FusedReshapeConcatGeneral Layer

The FusedReshapeConcatGeneral layer cross combines the input tensors and outputs item tensor, AD tensor.

Parameters: None

Input and Output Shapes:

* input: {(batch_size, num_feas, num_elems_0), (batch_size, num_feas, num_elems_1), ...}, the input tensors are embeddings.
* output: (batch_size x num_feas, (num_elems_0 + num_elems_1 + ...)).

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedReshapeConcatGeneral,
                            bottom_names = ["sparse_embedding_good", "sparse_embedding_cate"],
                            top_names = ["FusedReshapeConcat_item_his_em"]))
```

#### Softmax Layer

The Softmax layer computes softmax activations.

Parameter: None

Input and Output Shapes:

* input: (batch_size, num_elems)
* output: same as input

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Softmax,
                            bottom_names = ["reshape1"],
                            top_names = ["softmax_i"]))
```

#### Sub Layer

Inputs: x tensor, y tensor in same size.
Produce x - y in element wise manner.

Parameters: None

Input and Output Shapes:

* input: Nx(batch_size, num_elems) where N is the number of input tensors
* output: (batch_size, num_elems)

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Sub,
                            bottom_names = ["Scale_item1", "item_his1"],
                            top_names = ["sub_ih"]))
```

#### ReduceMean Layer

The ReduceMean Layer computes the mean of elements across a specified dimension.

Parameter:

* `axis`: The dimension to reduce. If the input is N-dimensional, 0 <= axis < N.

Input and Output Shapes:

* input: (batch_size, ...) where ... represents any number of elements with an arbitrary number of dimensions
* output: Dimension corresponding to axis is set to 1. The others remain the same as the input.

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReduceMean,
                            bottom_names = ["fmorder2"],
                            top_names = ["reducemean1"],
                            axis=1))
```

#### MatrixMutiply Layer

The MatrixMutiply Layer is a binary operation that produces a matrix output from two matrix inputs by performing matrix mutiplication.

Parameters: None

Input and Output Shapes:

* input: 2D: (m, n), (n, k) or 3D: (batch_size, m, n), (batch_size, n, k)
* output: 2D: (m, k) or 3D: (batch_size, m, k)

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.MatrixMutiply,
                            bottom_names = ["slice1","slice2"],
                            top_names = ["MatrixMutiply1"])
```
#### MultiHeadAttention Layer

The MultiHeadAttention Layer is a binary operation that produces a matrix output from two matrix inputs by performing matrix mutiplication.

Parameters: None

Input and Output Shapes:

* input: 4D: (head_num, batch_size, m, n), (head_num, batch_size, k, n)
* output: 4D: (head_num, batch_size, m, k)

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.MultiHeadAttention,
                            bottom_names = ["query","key"],
                            top_names = ["AttentionOut1"])
```

#### Gather Layer

The Gather layer gather multiple output tensor slices from an input tensors on the last dimension.

Parameter:

* `indices`: A list of indices in which each one represents an index in the input tensor to generate the corresponding output tensor. For example, [2, 8] indicates the second and eights tensor slice in the input tensor which are used to create an output tensor.

Input and Output Shapes:

* input: (batch_size, num_elems)
* output: (num_indices, num_elems)

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Gather,
                            bottom_names = ["reshape1"],
                            top_names = ["gather1"],
                            indices=[1,3,5]))
```

### BinaryCrossEntropyLoss

BinaryCrossEntropyLoss calculates loss from labels and predictions where each label is binary. The final sigmoid function is fused with the loss function to better utilize memory bandwidth.

Parameter:
* `use_regularizer`: Boolean, whether to use regulariers. THe default value is False.
* `regularizer_type`: The regularizer type for the `BinaryCrossEntropyLoss`, `CrossEntropyLoss` or `MultiCrossEntropyLoss` layer. The supported types include `hugectr.Regularizer_t.L1` and `hugectr.Regularizer_t.L2`. It will be ignored if `use_regularizer` is False. The default value is `hugectr.Regularizer_t.L1`.
* `lambda`: Float, the lambda value of the regularization term. It will be ignored if `use_regularier` is False. The default value is 0.

Input and Output Shapes:

* input: [(batch_size, 1), (batch_size, 1)] where the first tensor represents the predictions while the second tensor represents the labels
* output: (batch_size, 1)

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["add", "label"],
                            top_names = ["loss"]))
```

### CrossEntropyLoss

CrossEntropyLoss calculates loss from labels and predictions between the forward propagation phases and backward propagation phases. It assumes that each label is two-dimensional.

Parameter:

* `use_regularizer`: Boolean, whether to use regulariers. THe default value is False.
* `regularizer_type`: The regularizer type for the `BinaryCrossEntropyLoss`, `CrossEntropyLoss` or `MultiCrossEntropyLoss` layer. The supported types include `hugectr.Regularizer_t.L1` and `hugectr.Regularizer_t.L2`. It will be ignored if `use_regularizer` is False. The default value is `hugectr.Regularizer_t.L1`.
* `lambda`: Float, the lambda value of the regularization term. It will be ignored if `use_regularier` is False. The default value is 0.

Input and Output Shapes:

* input: [(batch_size, 2), (batch_size, 2)] where the first tensor represents the predictions while the second tensor represents the labels
* output: (batch_size, 2)

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.CrossEntropyLoss,
                            bottom_names = ["add", "label"],
                            top_names = ["loss"],
                            use_regularizer = True,
                            regularizer_type = hugectr.Regularizer_t.L2,
                            lambda = 0.1))
```

### MultiCrossEntropyLoss

MultiCrossEntropyLoss calculates loss from labels and predictions between the forward propagation phases and backward propagation phases. It allows labels in an arbitrary dimension, but all the labels must be in the same shape.

Parameter:

* `use_regularizer`: Boolean, whether to use regulariers. THe default value is False.
* `regularizer_type`: The regularizer type for the `BinaryCrossEntropyLoss`, `CrossEntropyLoss` or `MultiCrossEntropyLoss` layer. The supported types include `hugectr.Regularizer_t.L1` and `hugectr.Regularizer_t.L2`. It will be ignored if `use_regularizer` is False. The default value is `hugectr.Regularizer_t.L1`.
* `lambda`: Float, the lambda value of the regularization term. It will be ignored if `use_regularier` is False. The default value is 0.

Input and Output Shapes:

* input: [(batch_size, *), (batch_size, *)] where the first tensor represents the predictions while the second tensor represents the labels. * represents any even number of elements.
* output: (batch_size, *)

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.MultiCrossEntropyLoss,
                            bottom_names = ["add", "label"],
                            top_names = ["loss"],
                            use_regularizer = True,
                            regularizer_type = hugectr.Regularizer_t.L1,
                            lambda = 0.1
                            ))
```

## Embedding Collection

### About the HugeCTR embedding collection

Embedding collection is introduced in the v3.7 release.
The embedding collection enables you to use embeddings with different vector sizes, optimizers, and arbitrary table placement strategy.
Compared with the `hugectr.SparseEmbedding` class, the embedding collection has three key advantages:

1. The embedding collection can fuse embedding tables with different embedding vector sizes.
   The previous embedding can only fuse embedding tables with the same embedding vector size.
   The enhancement boosts both flexibility and performance.
2. The embedding collection extends the functionality of embedding by supporting the `concat` combiner and supporting different slot lookups on the same embedding table.
3. The embedding collection supports arbitrary embedding table placement, such as data parallel and model parallel.
You provide a plan JSON file and specify the table placement strategy that you want.

### Sample Notebook

The [HugeCTR Embedding Collection](../notebooks/embedding_collection.ipynb) sample notebook demonstrates the following:

* Introduces the API of the embedding collection.
* Introduces the embedding table placement strategy (ETPS) and how to configure ETPS in embedding collection.
* Shows how to use an embedding collection in a DLRM model with the Criteo dataset for training and evaluation.
The notebook shows two different ETPS as reference.

### Overview of using the HugeCTR embedding collection

To use an embedding collection, you need the following items:

* A list of `hugectr.EmbeddingTableConfig` objects that represent the embedding tables such as the size, maximum key, and an optional optimizer.

* A `hugectr.EmbeddingPlanner` object that uses the embedding table config objects to organize the lookup operations between the input data and the embedding tables.
The embedding planner also accepts a JSON-formatted plan file that describes the embedding table placement strategy.

After those items are in place, you run the `create_embedding_collection()` method on the embedding planner and you receive a
 `hugectr.EmbeddingCollection`.
Specify the embedding collection as an argument to `Model.add()` to use the embedding collection.

### EmbeddingTableConfig

The `hugectr.EmbeddingTableConfig` class enables you to specify the attributes of an embedding table.

Parameter:

* `table_id`: Integer, typically it is the index from the corresponding `slot_size_array` variable when you create a new embedding table.
* `max_vocabulary_size`: Integer, specifies the vocabulary size of this table.
If positive, then the value indicates the number of embedding vectors that this table contains.
If you specify the value incorrectly and exceed the value during training or evaluation, you will cause an overflow and receive an error.
If you do not know the exact size of the embedding table, you can specify `-1` to use a dynamic embedding table with a size that can be extended dynamically during training or evaluation.
* `ev_size`: Integer, specifies the embedding vector size that this embedding consists of.
* `min_key`: Integer, the minimum value of the input key.
* `max_key`: Integer, the maximum value of the input key.
* `opt_params`: Optional, `hugectr.Optimizer`, the optimizer you want to use for this embedding table.
If not specified, the embedding table uses the optimizer specified in `hugectr.Model`.

Example:

```python
# Create the embedding table.
slot_size_array = [203931, 18598, 14092, 7012, 18977, 4, 6385, 1245, 49,
                   186213, 71328, 67288, 11, 2168, 7338, 61, 4, 932, 15,
                   204515, 141526, 199433, 60919, 9137, 71, 34]
embedding_table_list = []
for i in range(len(slot_size_array):
    embedding_table_list.append(
        hugectr.EmbeddingTableConfig(
            table_id=i,
            max_vocabulary_size=slot_size_array[i],
            ev_size=128,
            min_key=0,
            max_key=slot_size_array[i]
        )
    )
```

### EmbeddingPlanner

Create a `hugectr.EmbeddingPlanner` instance to construct the lookup operation and create a `hugectr.EmbeddingCollection`.
The constructor for the embedding planner class does not accept any parameters.

#### embedding_lookup method

The `embedding_lookup` method enables you to specify the lookup operations between the input data and an embedding table.

Parameter:

* `table_config` : `hugectr.EmbeddingTableConfig`, the embedding table for the lookup operation.
* `bottom_name`: str, the bottom tensor name.
Specify a tensor that is compatible with the `data_reader_sparse_param_array` parameter of [`hugectr.Input`](#input-layer) in `hugectr.Model`.
* `top_name`: str, the output tensor name.
The shape of output tensor is (`<batch size>`, `1`, `<embedding vector size>`).
* `combiner`: str, specifies the combiner operation.
Specify `mean`, `sum`, or `concat`.

#### create_embedding_collection method

After configuring the embedding planner with the embedding table information by running `embedding_lookup()` for each embedding table, you can run the `create_embedding_collection()` method to create a `hugectr.EmbeddingCollection` instance.
You can use the `add()` method from `hugectr.Model` to use the embedding collection for training and evaluation.

Parameter:

* `plan_file`: str, specifies the path to a JSON file that describes the embedding table placement strategy.
Example:

```python
embedding_planner = hugectr.EmbeddingPlanner()
emb_vec_list = []
for i in range(len(slot_size_array)):
     embedding_planner.embedding_lookup(
        table_config=embedding_table_list[i],
        bottom_name="data{}".format(i),
        top_name="emb_vec{}".format(i),
        combiner="sum"
    )
embedding_collection = embedding_planner.create_embedding_collection("./plan.json")
```

### Plan File and Embedding Table Placement Strategy (ETPS)

### About ETPS and Benefits

In the recommendation system, the embedding table is usually so large that a single GPU is not able to hold all embedding tables.
One strategy for addressing the challenge is to use sharding to distribute the embedding tables across multiple GPUs.
We call this sharding strategy the embedding table placement strategy (ETPS).

ETPS can significantly boost the performance of embedding because different sharding strategies influence the communication between GPUs.
The optimal strategy is highly dependent on your dataset and your lookup operation.
If optimal performance is a concern, then configure an ETPS for the specific use case.

### Configuring ETPS and the Embedding Collection

HugeCTR provides a configurable ETPS interface so that you can adjust the embedding table placement strategy according your own use case.
You create a JSON file that describes the ETPS in all GPUs.
This file is called a _plan file_.

For example, consider the following scenario that has four embedding tables and five lookup operations:

```python
slot_size_array = [...]
embedding_table_list = []
for i in range(len(slot_size_array)):
    embedding_table_list.append(
        hugectr.EmbeddingTableConfig(
            table_id=i,
            max_vocabulary_size=slot_size_array[i],
            ev_size=128,
            min_key=0,
            max_key=slot_size_array[i]
        )
    )

embedding_planner = hugectr.EmbeddingPlanner()
embedding_planner.embedding_lookup(embedding_table_list[0], "data0", "emb_vec0", "sum") # lookup 0, table 0
embedding_planner.embedding_lookup(embedding_table_list[1], "data1", "emb_vec1", "sum") # lookup 1, table 1
embedding_planner.embedding_lookup(embedding_table_list[2], "data2", "emb_vec2", "sum") # lookup 2, table 2
embedding_planner.embedding_lookup(embedding_table_list[1], "data3", "emb_vec3", "sum") # lookup 3, table 1
embedding_planner.embedding_lookup(embedding_table_list[3], "data4", "emb_vec4", "sum") # lookup 4, table 3
```

The next step is to configure the ETPS through a plan file.
In the plan file, you can group several lookup operations together and to perform sharding.
You can specify the configuration in fine detail, down to the lookup operation, the GPU, and a portion of the embedding table.

The basic principle is one embedding table can only be sharded in a single way.
For example, if lookup `0` and lookup `3` take place on the same embedding table, then lookup `0` and lookup `3` should be grouped together and sharded in the same way.

#### Sample ETPS Plan File: Data Parallel

The following sample plan file shows how to use two GPUs and a data parallel ETPS in four embedding tables:

```json
[
  [
      {
          "local_embedding_list": [
              0, 1, 2, 3, 4
          ],
          "global_embedding_list": [
              [
                  0, 1, 2, 3, 4
              ],
              [
                  0, 1, 2, 3, 4
              ]
          ],
          "shards_count": 1,
          "shard_id": 0,
          "table_placement_strategy": "dp"
      }
  ],
  [
      {
          "local_embedding_list": [
              0, 1, 2, 3, 4
          ],
          "global_embedding_list": [
              [
                  0, 1, 2, 3, 4
              ],
              [
                  0, 1, 2, 3, 4
              ]
          ],
          "shards_count": 1,
          "shard_id": 0,
          "table_placement_strategy": "dp"
      }
  ]
]
```

The plan file consists of a list that describes the table placement strategy in each GPU, in order.
For each GPU, a list describes the multiple groups of sharded lookup operations.
Each group of sharded lookup operation is a dictionary with the following fields:

* `local_embedding_list`: a list of integers, specifies the lookup operations for the current GPU.
* `global_embedding_list`: a list of lists of integers, specifies the current group lookup operations in all GPUs.
* `shards_count`: an integer, specifies the total number of shards for the current group lookup operations.
* `shard_id`: an integer, the index of the current group lookup operations.
* `table_placement_strategy`: str, can be `mp` or `dp`. `mp` indicates model parallel and `dp` indicates data parallel.

#### Sample ETPS Plan File: Model Parallel and Data Parallel Combined

You can apply more complex strategies for ETPS.
The following sample plan file demonstrates how to shared lookups `0`, `1`, `2`, and `3` across two GPUs and lookup `4` is data parallel:

```json
[
  [
      {
          "local_embedding_list": [
              0,
              2
          ],
          "global_embedding_list": [
              [
                  0,
                  2
              ],
              [
                  1,
                  3
              ]
          ],
          "table_placement_strategy": "mp"
      },
      {
          "local_embedding_list": [
              4
          ],
          "global_embedding_list": [
              [
                  4
              ],
              [
                  4
              ]
          ],
          "table_placement_strategy": "dp"
      }
  ],
  [
      {
          "local_embedding_list": [
              1,
              3
          ],
          "global_embedding_list": [
              [
                  0,
                  2
              ],
              [
                  1,
                  3
              ]
          ],
          "table_placement_strategy": "mp"
      },
      {
          "local_embedding_list": [
              4
          ],
          "global_embedding_list": [
              [
                  4
              ],
              [
                  4
              ]
          ],
          "table_placement_strategy": "dp"
      }
  ]
]
```

## GroupDenseLayer

**DenseLayer class**
```python
hugectr.GroupDenseLayer()
```

`GroupDenseLayer` specifies the parameters related to a group of dense layers. HugeCTR currently supports only `GroupFusedInnerProduct`, which is comprised of multiple `FusedInnerProduct` layers. Please **NOTE** that the `FusedInnerProduct` layer only supports fp16.

**Arguments**
* `group_layer_type`: The layer type to be used. There is only one supported type, i.e., `hugectr.GroupLayer_t.GroupFusedInnerProduct`. There is NO default value and it should be specified by users.

* `bottom_name_list`: List[str], the list of bottom tensor names for the first dense layer in this group. Currently, the `FusedInnerProduct` layer at the head position can take one or two input tensors. There is NO default value and it should be specified by users.

* `top_name_list`: List[str], the list of top tensor names of each dense layer in the group. There should be only one name for each layer. There is NO default value and it should be specified by users.

* `num_outputs`: List[Integer], the number of output elements for each `FusedInnerProduct` layer in the group. There is NO default value and it should be specified by users.

* `last_act_type`: The activation type of the last `FusedInnerProduct` layer in the group. The supported types include `Activation_t.Relu` and `Activation_t.Non`. Except the last layer, the activation type of the other `FusedInnerProduct` layers in the group must be and will be automatically set as `Activation_t.Relu`, which do not allow any configurations. The default value is `Activation_t.Relu`.

**NOTE**: There should be at least two layers in the group, and the size of `top_name_list` and `num_outputs` should both be equal to the number of layers.

Example:

```python
model.add(hugectr.GroupDenseLayer(group_layer_type = hugectr.GroupLayer_t.GroupFusedInnerProduct,
                                  bottom_name = ["dense"],
                                  top_name_list = ["fc1", "fc2", "fc3"],
                                  num_outputs = [1024, 512, 256],
                                  last_act_type = hugectr.Activation_t.Relu))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Interaction,
                            bottom_names = ["fc3","sparse_embedding1"],
                            top_names = ["interaction1", "interaction1_grad"]))
model.add(hugectr.GroupDenseLayer(group_layer_type = hugectr.GroupLayer_t.GroupFusedInnerProduct,
                            bottom_name_list = ["interaction1", "interaction1_grad"],
                            top_name_list = ["fc4", "fc5", "fc6", "fc7", "fc8"],
                            num_outputs = [1024, 1024, 512, 256, 1],
                            last_act_type = hugectr.Activation_t.Non))
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                            bottom_names = ["fc8", "label"],
                            top_names = ["loss"]))
```
