# HugeCTR Layer Book
This document introduces different layer classes and corresponding methods in the Python API of HugeCTR. The description of each method includes its functionaliy, arguments, and examples of usage.
## Table of contents:
* [Input Layer](#input-layer)
* [Sparse Embedding Layers](#sparse-embedding)
  * [DistributedSlotSparseEmbeddingHash Layer](#distributedslotsparseembeddinghash-layer)
  * [LocalizedSlotSparseEmbeddingHash Layer](#localizedslotsparseembeddinghash-layer)
  * [LocalizedSlotSparseEmbeddingoneHot Layer](#localizedslotsparseembeddingonehot-layer)
* [Dense Layers](#dense-layers)
  * [FullyConnected Layer](#fullyconnected-layer)
  * [FusedFullyConnected Layer](#fusedfullyconnected-layer)
  * [MultiCross Layer](#multicross-layer)
  * [FmOrder2 Layer](#fmorder2-layer)
  * [WeightMultiply Layer](#weightmultiply-layer)
  * [ElementwiseMultiply Layer](#elementwisemultiply-layer)
  * [BatchNorm Layer](#batchnorm-layer)
  * [Concat Layer](#concat-layer)
  * [Reshape Layer](#reshape-layer)
  * [Slice Layer](#slice-layer)
  * [Dropout Layer](#dropout-layer)
  * [ELU Layer](#elu-layer)
  * [Interaction Layer](#interaction-layer)
  * [Add Layer](#add-layer)
  * [ReduceSum Layer](#reducesum-layer)
  * [BinaryCrossEntropyLoss](#binarycrossentropyloss)
  * [CrossEntropyLoss](#crossentropyloss)
  * [MultiCrossEntropyLoss](#multicrossentropyloss)

## Input Layer ##
```python
hugectr.Input()
```
`Input` layer specifies the parameters related to the data input. `Input` layer should be added to the Model instance first so that the following `SparseEmbedding` and `DenseLayer` instances can access the inputs with their specified names.

**Arguments**
* `label_dim`: Integer, the label dimension. 1 implies it is a binary label. For example, if an item is clicked or not. There is NO default value and it should be specified by users.

* `label_name`: String, the name of the label tensor to be referenced by following layers. There is NO default value and it should be specified by users.

* `dense_dim`: Integer, the number of dense (or continuous) features. If there is no dense feature, set it to 0. There is NO default value and it should be specified by users.

* `dense_name`: Integer, the name of the dense input tensor to be referenced by following layers. There is NO default value and it should be specified by users.

* `data_reader_sparse_param_array`: List[hugectr.DataReaderSparseParam], the list of the sparse parameters for categorical inputs. Each `DataReaderSparseParam` instance should be constructed with `hugectr.DataReaderSparse_t`, `max_feature_num`, `max_nnz` and `slot_num`. The supported types of `hugectr.DataReaderSparse_t` include `hugectr.DataReaderSparse_t.Distributed` and `hugectr.DataReaderSparse_t.Localized`. The maximum number of features per sample for the specified spare input can be specified by `max_feature_num`. For `max_nnz`, if it is set to 1, the dataset is specified as one-hot so that the memory consumption can be reduced. As for `slot_num`, it specifies the number of slots used for this sparse input in the dataset. The total number of categorical inputs is exactly the length of `data_reader_sparse_param_array`. There is NO default value and it should be specified by users.

* `sparse_names`: List[str], the list of names of the sparse input tensors to be referenced by following layers. The order of the names should be consistent with sparse parameters in `data_reader_sparse_param_array`. There is NO default value and it should be specified by users.

**Example:**
```python
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 13, dense_name = "dense",
                        data_reader_sparse_param_array = 
                            [hugectr.DataReaderSparseParam(hugectr.DataReaderSparse_t.Distributed, 30, 1, 26)],
                        sparse_names = ["data1"]))
```

```python
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 13, dense_name = "dense",
                        data_reader_sparse_param_array = 
                            [hugectr.DataReaderSparseParam(hugectr.DataReaderSparse_t.Distributed, 30, 1, 2),
                            hugectr.DataReaderSparseParam(hugectr.DataReaderSparse_t.Distributed, 30, 1, 26)],
                        sparse_names = ["wide_data", "deep_data"]))
```

## Sparse Embedding ##
**SparseEmbedding class**
```bash
hugectr.SparseEmbedding()
```
`SparseEmbedding` specifies the parameters related to the sparse embedding layer. One or several `SparseEmbedding` layers should be added to the Model instance after `Input` and before `DenseLayer`.

**Arguments**
* `embedding_type`: The embedding type to be used. The supported types include `hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash`, `hugectr.Embedding_t.LocalizedSlotSparseEmbeddingHash` and `hugectr.Embedding_t.LocalizedSlotSparseEmbeddingOneHot`. There is NO default value and it should be specified by users. For detail about different embedding types, please refer to [Embedding Types Detail](./hugectr_layer_book.md#embedding-types-detail).

* `max_vocabulary_size_per_gpu`: Integer, the maximum vocabulary size or cardinality across all the input features. There is NO default value and it should be specified by users.

* `embedding_vec_size`: Integer, the embedding vector size. There is NO default value and it should be specified by users.

* `combiner`: Integer, the intra-slot reduction operation (0=sum, 1=average). There is NO default value and it should be specified by users.

* `sparse_embedding_name`: String, the name of the sparse embedding tensor to be referenced by following layers. There is NO default value and it should be specified by users.

* `bottom_name`: String, the number of the bottom tensor to be consumed by this sparse embedding layer. Please note that it should be a predefined sparse input name. There is NO default value and it should be specified by users.

* `slot_size_array`: List[int], the cardinality array of input features. It should be consistent with that of the sparse input. If `max_vocabulary_size_per_gpu` is specified, this parameter is ignored. There is NO default value and it should be specified by users.

* `optimizer`: OptParamsPy, the optimizer dedicated to this sparse embedding layer. If the user does not specify the optimizer for the sparse embedding, it will adopt the same optimizer as dense layers. 

## Embedding Types Detail ##
### DistributedSlotSparseEmbeddingHash Layer ###
The `DistributedSlotSparseEmbeddingHash` stores embeddings in an embedding table and gets them by using a set of integers or indices. The embedding table can be segmented into multiple slots or feature fields, which spans multiple GPUs and nodes. With `DistributedSlotSparseEmbeddingHash`, each GPU will have a portion of a slot. This type of embedding is useful when there's an existing load imbalance among slots and OOM issues.

**Important Notes**:

* In a single embedding layer, it is assumed that input integers represent unique feature IDs, which are mapped to unique embedding vectors.
All the embedding vectors in a single embedding layer must have the same size. If you want some input categorical features to have different embedding vector sizes, use multiple embedding layers.
* The input indices’ data type, `input_key_type`, is specified in the solver. By default,  the 32-bit integer (I32) is used, but the 64-bit integer type (I64) is also allowed even if it is constrained by the dataset type. For additional information, see [Solver](./python_interface.md#solver).

**Example:**
```python
model.add(hugectr.SparseEmbedding(
            embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
            max_vocabulary_size_per_gpu = 5863985,
            embedding_vec_size = 1,
            combiner = 0,
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
            max_vocabulary_size_per_gpu = 5863985,
            embedding_vec_size = 1,
            combiner = 0,
            sparse_embedding_name = "sparse_embedding1",
            bottom_name = "input_data",
            optimizer = optimizer))
```

### LocalizedSlotSparseEmbeddingOneHot Layer
The LocalizedSlotSparseEmbeddingOneHot layer stores embeddings in an embedding table and gets them by using a set of integers or indices. The embedding table can be segmented into multiple slots or feature fields, which spans multiple GPUs and nodes. This is a performance-optimized version of LocalizedSlotSparseEmbeddingHash for the case where NVSwitch is available and inputs are one-hot categorical features.

**Note**: Unlike other types of embeddings, LocalizedSlotSparseEmbeddingOneHot only supports single-node training. LocalizedSlotSparseEmbeddingOneHot can be supported only in a NVSwitch equipped system such as DGX-2 and DGX A100.
The input indices’ data type, `input_key_type`, is specified in the solver. By default, the 32-bit integer (I32) is used, but the 64-bit integer type (I64) is also allowed even if it is constrained by the dataset type. For additional information, see [Solver](#solver).

Example:
```python
model.add(hugectr.SparseEmbedding(
            embedding_type = hugectr.Embedding_t.LocalizedSlotSparseEmbeddingOneHot, 
            slot_size_array = [1221, 754, 8, 4, 12, 49, 2]
            embedding_vec_size = 128,
            combiner = 0,
            sparse_embedding_name = "sparse_embedding1",
            bottom_name = "input_data",
            optimizer = optimizer))
```

## Dense Layers ##
**DenseLayer class**
```bash
hugectr.DenseLayer()
```
`DenseLayer` specifies the parameters related to the dense layer or the loss function. HugeCTR currently supports multiple dense layers and loss functions. Please **NOTE** that the final sigmoid function is fused with the loss function to better utilize memory bandwidth.

**Arguments**
* `layer_type`: The layer type to be used. The supported types include `hugectr.Layer_t.Add`, `hugectr.Layer_t.BatchNorm`, `hugectr.Layer_t.Cast`, `hugectr.Layer_t.Concat`, `hugectr.Layer_t.DotProduct`, `hugectr.Layer_t.Dropout`, `hugectr.Layer_t.ELU`, `hugectr.Layer_t.FmOrder2`, `hugectr.Layer_t.FusedInnerProduct`, `hugectr.Layer_t.InnerProduct`, `hugectr.Layer_t.Interaction`, `hugectr.Layer_t.MultiCross`, `hugectr.Layer_t.ReLU`, `hugectr.Layer_t.ReduceSum`, `hugectr.Layer_t.Reshape`, `hugectr.Layer_t.Sigmoid`, `hugectr.Layer_t.Slice`, `hugectr.Layer_t.WeightMultiply`, `hugectr.Layer_t.ElementwiseMultiply`, `hugectr.Layer_t.BinaryCrossEntropyLoss`, `hugectr.Layer_t.CrossEntropyLoss` and `hugectr.Layer_t.MultiCrossEntropyLoss`. There is NO default value and it should be specified by users.

* `bottom_names`: List[str], the list of bottom tensor names to be consumed by this dense layer. Each name in the list should be the predefined tensor name. There is NO default value and it should be specified by users.

* `top_names`: List[str], the list of top tensor names, which specify the output tensors of this dense layer. There is NO default value and it should be specified by users.

* For details about the usage of each layer type and its parameters, please refer to [Dense Layers Usage](#dense-layers-usage).

## Dense Layers Usage ##

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

### Concat Layer
The Concat layer concatenates a list of inputs.

Parameters: None

Input and Output Shapes:

* input: Nx(batch_size, *) where 2<=N<=4 and * represents any number of elements
* output: (batch_size, total_num_elems) where total_num_elems is the summation of N input elements

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,
                            bottom_names = ["reshape3","weight_multiply2"],
                            top_names = ["concat2"]))
```

### Reshape Layer
The Reshape layer reshapes a 3D input tensor into 2D shape.

Parameter:

* `leading_dim`: Integer, the innermost dimension of the output tensor. It must be the multiple of the total number of input elements. If it is unspecified, n_slots * num_elems (see below) is used as the default leading_dim.
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

* `ranges`: List[Tuple[int, int]], used for the Slice layer. A list of tuples in which each one represents a range in the input tensor to generate the corresponding output tensor. For example, (2, 8) indicates that 8 elements starting from the second element in the input tensor are used to create an output tensor. The number of tuples corresponds to the number of output tensors. Ranges are allowed to overlap unless it is a reverse or negative range. The default value is [].

Input and Output Shapes:

* input: (batch_size, num_elems)
* output: {(batch_size, b-a), (batch_size, d-c), ....) where ranges ={[a, b), [c, d), …} and len(ranges) <= 5

Example:
```python
model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Slice,
                            bottom_names = ["dense"],
                            top_names = ["slice21", "slice22"],
                            ranges=[(0,13),(0,13)]))
```

### Dropout Layer
The Dropout layer randomly zeroizes or drops some of the input elements.

Parameter:

* `dropout_rate`: Float, The dropout rate to be used for the `Dropout` layer. It should be between 0 and 1. Setting it to 1 indicates that there is no dropped element at all. The default value is 0.5.

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
