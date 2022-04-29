# Mixed Precision #
Mixed precision is a way to make an application run faster and use less memory. It uses both 16-bit and 32-bit floating-point types during model training.

SparseOperationKit follows the TensorFlow approach to enable mixed precision training. For more information, see the TensorFlow documentation about [mixed precision](https://tensorflow.google.cn/guide/mixed_precision?hl=en).

## TensorFlow 2.x ##
This section explains how to enable mixed-precision training with TF 2.x.

### enable mixed precision ###
To use mixed precision in your model training, you just need to add the following two lines of code to your training script and keep other parts untouched.
```python
policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)
```

### loss scaling ###
The `float16` data type has a narrow dynamic range compared to `float32`. As a result, `float16` might lead model training to underflow or overflow problems. Loss scaling is a technique to avoid numeric underflow. 

TensorFlow provides an optimizer wrapper for loss scaling. 
```python
optimizer = tf.keras.optimizers.Adam() # could be other optimizers as well
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
```

In the training loop, this optimizer wrapper should be used to calculate the scaled loss value. After the backward propagation and before updating trainable parameters, that wrapper should be used to adjust the gradients. This is necessary because the loss is scaled. As a result, when updating, the gradients must also be scaled accordingly.
```python
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(y, predictions)
        scaled_loss = optimizer.get_scaled_loss(loss) # this API should be called to get scaled loss
    emb_variables, other_variables =\
         sok.split_embedding_variable_from_others(model.trainable_variables)
    scaled_emb_gradients, scaled_other_gradients = tape.gradient(scaled_loss, 
                                            [emb_variables, other_variables])
    # use this API to scale embedding gradients back to correct value
    emb_gradients = optimizer.get_unscaled_gradients(scaled_emb_gradients) 
    # use this API to scale other gradients back to correct value
    other_gradients = optimizer.get_unscaled_gradients(scaled_other_gradients) 
    optimizer.apply_gradients(zip(emb_gradients, emb_variables),
                              experimental_aggregate_gradients=False)
    optimizer.apply_gradients(zip(other_gradients, other_variables))
    return loss
```

### example codes ###
You can find the mixed-precision example using TensorFlow 2.x in the  [`sparse_operation_kit/documents/tutorials/MixedPrecision`](https://github.com/NVIDIA/HugeCTR/tree/master/sparse_operation_kit/documents/tutorials/MixedPrecision) directory of the GitHub repository. Use the following command to launch it:
```shell
$ cd sparse_operation_kit/documents/tutorials/MixedPrecision/
$ python amp_tf2.py
```


## TensorFlow 1.15 ##
This section explains how to enable mixed-precision training with TF 1.15.

### enable mixed precision ###
To use mixed precision in your model training, you just need to add the following four lines of code to your training script and keep other parts untouched.
```python
from tensorflow.python.keras.engine import base_layer_utils
base_layer_utils.enable_v2_dtype_behavior()

policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
tf.keras.mixed_precision.experimental.set_policy(policy)
```

### loss scaling ###
Analogous to the TensorFlow 2.x approach, the `LossScaleOptimizer` wrapper should be used for loss scaling.
```python
optimizer = tf.keras.optimizers.Adam() # could be other optimizers as well
optimizer = sok.tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
```
Our TensorFlow 1.15 implementation of the `sok.tf.keras.mixed_precision.LossScaleOptimizer` is an optimized wrapper based on the `tf.keras.mixed_precision.experimental.LossScaleOptimizer` that exposes the same methods and arguments. Refer to the TensorFlow documentation about the [loss scale optimizer](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/mixed_precision/experimental/LossScaleOptimizer) for a detailed list and description of the all available options.

During the training loop, this optimizer wrapper should be used to calculate the scaled loss value, and to obtain the unscaled gradients before updating parameters.
```python
def train_step(x, y):
    predictions = model(x)
    loss = loss_object(y, predictions)
    # use this API to get scaled loss value
    scaled_loss = optimizer.get_scaled_loss(loss)
    scaled_gradients = tf.gradients(scaled_loss, model.trainable_variables)
    # distinguish embedding variables and other variabels
    # because they need different processing.
    emb_variables, other_variables =\
        sok.split_embedding_variable_from_others(model.trainable_variables)
    scaled_emb_gradients, scaled_other_gradients =\
        scaled_gradients[:len(emb_variables)], scaled_gradients[len(emb_variables):]
    # use this API to scale embedding gradients back to correct value
    emb_gradients = optimizer.get_unscaled_gradients(scaled_emb_gradients)
    # use this API to scale other gradients back to correct value
    other_gradients = optimizer.get_unscaled_gradients(scaled_other_gradients)
    other_gradients = [hvd.allreduce(grad) for grad in other_gradients]
    emb_train_op = optimizer.apply_gradients(zip(emb_gradients, emb_variables))
    other_train_op = optimizer.apply_gradients(zip(other_gradients, other_variables))

    with tf.control_dependencies([emb_train_op, other_train_op]):
        return tf.identify(loss)
```

### example codes ###
You can find the mixed-precision example using TensorFlow 1.15 in the[`sparse_operation_kit/documents/tutorials/MixedPrecision`](https://github.com/NVIDIA/HugeCTR/tree/master/sparse_operation_kit/documents/tutorials/MixedPrecision) directory of the GitHub repository. Use the following command to launch it:
```shell
$ cd sparse_operation_kit/documents/tutorials/MixedPrecision/
$ mpiexec --allow-run-as-root -np <gpu-number> python amp_tf1.py
```