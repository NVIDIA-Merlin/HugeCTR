"""
 Copyright (c) 2023, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import torch
import os

install_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lib/libhps_torch.so"))
torch.ops.load_library(install_path)


class LookupLayer(torch.nn.Module):
    """
    Abbreviated as ``hps_torch.LookupLayer(*args, **kwargs)``.

    This is a wrapper class for HPS lookup layer, which basically performs
    the same function as ``torch.nn.Embedding``.

    Parameters
    ----------
    ps_config_file: str
            The JSON configuration file for HPS initialization.
    model_name: str
            The name of the model that has embedding tables.
    table_id: int
            The index of the embedding table for the model specified by
            model_name.
    emb_vec_size: int
            The embedding vector size for the embedding table specified
            by model_name and table_id.

    Examples
    --------
    .. code-block:: python
        import torch
        import hps_torch
        lookup_layer = hps_torch.LookupLayer(ps_config_file = args.ps_config_file,
                                      model_name = args.model_name,
                                      table_id = args.table_id,
                                      emb_vec_size = args.embed_vec_size)
        keys = torch.randint(0, 100, (16, 10), dtype=torch.int64).cuda()
        vectors = lookup_layer(keys)
    """

    def __init__(self, ps_config_file: str, model_name: str, table_id: int, emb_vec_size: int):
        super().__init__()
        self.ps_config_file = ps_config_file
        self.model_name = model_name
        self.table_id = table_id
        self.emb_vec_size = emb_vec_size

    def forward(self, keys):
        """
        The forward logic of this wrapper class.

        Parameters
        ----------
        keys:
                Keys are stored in Tensor. The supported data types are ``torch.int32`` and ``torch.int64``.

        Returns
        -------
        vectors: ``torch.Tensor`` of float32
                the embedding vectors for the input keys.
        """
        vectors = torch.ops.hps_torch.hps_embedding_lookup(
            keys, self.ps_config_file, self.model_name, self.table_id, self.emb_vec_size
        )
        return vectors
