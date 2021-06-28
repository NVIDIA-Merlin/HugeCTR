# Usage Samples for embedding plugin #
---
This folder contains several script files, each of which is used to demonstrate how to use the plugin APIs of one version.

## General introduction ##
1. `fprop` & `fprop_experimental` can be placed inner the scope of `tf.distribute.MirroredStrategy` to define DNN model. They are located in `libembedding_plugin_v2.so`, and `import hugectr_tf_ops_v2` should be used in training script.
    - `fprop_experimental` takes CSR format as its inputs. In terms of converting to CSR format, please refer to `format_processing.py` for more details. In some cases, this version can get better performance than `fprop`, but it is not stable.
    - `fprop` takes COO format as its inputs. Please refer to `format_processing.py` for more details about converting to COO format.
2. `fprop_v3` & `fprop_v4` can **Not** be placed inner the scope of `tf.distribute.MirroredStrategy` to define DNN model. The whole DNN model must be seperated into two sub-models. They are located in `libembedding_plugin.so`, and `import hugectr_tf_ops` should be used in training script. `fprop_v3` and `fprop_v4` will be **Deprecated** in near future, please update to `fprop` or `fprop_experimental`.
    - `fprop_v3` takes CSR format as its inputs. Please refer to `format_processing.py` for more details about converting to CSR format.
    - `fprop_v4` takes COO format as its inputs. Please refer to `format_processing.py` for more details about converting to COO format.
