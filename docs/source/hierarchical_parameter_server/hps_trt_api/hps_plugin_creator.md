# HPS Plugin Creator

```{contents}
---
depth: 2
local: true
backlinks: none
---
```

The HPS plugin has plugin creator class, `HpsPluginCreator`, with the registration name `HPS_TRT`.

The parameters are defined below and consists of the following attributes:

| Type     | Parameter                | Description
|----------|--------------------------|--------------------------------------------------------
|`string`  |`ps_config_file`          |The configuration JSON file for HPS.
|`string`  |`model_name`              |The name of the model.
|`int32`   |`table_id`                |The index for the embedding table.
|`int32`   |`emb_vec_size`            |The embedding vector size.

Refer to the [HPS configuration](../hps_database_backend.md#configuration) documentation for details about writing  the `ps_config_file`.

```{important}
Add a trailing null character, `'\0'`, when you configure the `ps_config_file` and `model_name` with TensorRT Python APIs.
This requirement is due to limitations of the supported plugin field types.
```

Refer to the following Python code for an example of using the trailing null characters:

```python
import tensorrt as trt
import numpy as np
ps_config_file = trt.PluginField("ps_config_file", np.array(["hps_conf.json\0"], dtype=np.string_), trt.PluginFieldType.CHAR)
model_name = trt.PluginField("model_name", np.array(["demo_model\0"], dtype=np.string_), trt.PluginFieldType.CHAR)
```