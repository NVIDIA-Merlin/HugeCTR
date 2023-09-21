# HPS Plugin for Torch

```{contents}
---
depth: 2
local: true
backlinks: none
---
```

#### LookupLayer class

This is a wrapper class for HPS lookup layer, which basically performs the same function as ``torch.nn.Embedding``. It inherits `torch.nn.Module`.

```python
hps_torch.LookupLayer.__init__
```
**Arguments**
* `ps_config_file`: String. The JSON configuration file for HPS initialization.

* `model_name`: String. The name of the model that has embedding tables.

* `table_id`: Integer. The index of the embedding table for the model specified by `model_name`.

* `emb_vec_size`: Integer. The embedding vector size for the embedding table specified by `model_name` and `table_id`.


```python
hps_torch.LookupLayer.forward
```
**Arguments**
* `keys`: Tensor of ``torch.int32`` or ``torch.int64``.

**Returns**
* `vectors`: Tensor of `torch.float32`.