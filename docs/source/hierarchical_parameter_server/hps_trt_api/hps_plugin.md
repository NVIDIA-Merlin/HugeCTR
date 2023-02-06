# HPS Plugin

```{contents}
---
depth: 2
local: true
backlinks: none
---
```

The HPS plugin has plugin class, `HpsPlugin`, with the registration name `HPS_TRT`.

The HPS plugin accepts one input.
The input data type must be `int32`.
The input shape must be `(batch_size, num_keys_per_sample)`.

The HPS plugin generates one output.
The output data type is `float32`.
The output shape is `(batch_size, num_keys_per_sample, embedding_vector_size)`.

This plugin works for network with graph node named `HPS_TRT`. This is also the plugin name that should be used when getting the `HpsPluginCreator` from the Plugin Registry.