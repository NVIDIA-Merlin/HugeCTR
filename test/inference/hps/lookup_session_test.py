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

import numpy as np
import pandas as pd
import hugectr
from hugectr.inference import HPS, ParameterServerConfig, InferenceParams
import sys


key = [
    1,
    278211,
    693281,
    942339,
    961915,
    976114,
    983029,
    1001594,
    1001623,
    1007956,
    1009214,
    1009260,
    1235430,
    1315962,
    1388251,
    1388257,
    1390426,
    1398023,
    1398084,
    1398088,
    1399012,
    1399026,
    1648645,
    1817619,
    2061099,
    2129335,
    2138480,
    2138556,
]


def hps_dlpack(model_name, embedding_file_list, data_file, enable_cache, cache_type):
    batch_size = 1
    CATEGORICAL_COLUMNS = ["C1_C2", "C3_C4"] + [
        "C" + str(x) for x in range(1, 27)
    ]  # ["C1_C2","C3_C4"]+["C" + str(x) for x in range(1, 27)]
    emb_size = [
        278018,
        415262,
        249058,
        19561,
        14212,
        6890,
        18592,
        4,
        6356,
        1254,
        52,
        226170,
        80508,
        72308,
        11,
        2169,
        7597,
        61,
        4,
        923,
        15,
        249619,
        168974,
        243480,
        68212,
        9169,
        75,
        34,
    ]
    shift = np.insert(np.cumsum(emb_size), 0, 0)[:-1]

    ps_config = ParameterServerConfig(
        emb_table_name={"hps_demo": ["sparse_embedding0", "sparse_embedding1"]},
        embedding_vec_size={"hps_demo": [1, 16]},
        max_feature_num_per_sample_per_emb_table={"hps_demo": [2, 26]},
        inference_params_array=[
            InferenceParams(
                model_name="hps_demo",
                max_batchsize=10,
                hit_rate_threshold=1.0,
                dense_model_file="",
                sparse_model_files=embedding_file_list,  # ["/wdl_train/wdl0_sparse_2000.model","/wdl_train/wdl1_sparse_2000.model" ],
                deployed_devices=[0],
                use_gpu_embedding_cache=enable_cache,
                cache_size_percentage=0.5,
                embedding_cache_type=cache_type,
                i64_input_key=True,
            )
        ],
    )

    # 2. Initialize the HPS object
    hps = HPS(ps_config)

    # 3. Look up vectors from native hps
    print("[HUGECTR][INFO] Native hps lookup API test")
    # test_df=pd.read_csv("/wdl_train/infer_test.csv",sep=',')
    test_df = pd.read_csv(data_file, sep=",")
    cat_input = list((test_df.head(int(batch_size))[CATEGORICAL_COLUMNS] + shift).values.flatten())
    embedding1 = hps.lookup(cat_input[0:2], "hps_demo", 0).reshape(batch_size, 2, 1)
    embedding2 = hps.lookup(cat_input[2:], "hps_demo", 1).reshape(batch_size, 26, 16)
    print(
        "[HUGECTR][INFO] The shape of vectors that lookup from native hps interface for embedding table 1 of wdl: {}, the vectors: {}".format(
            embedding1[0].shape, embedding1
        )
    )
    print(
        "[HUGECTR][INFO] The shape of vectors that lookup from native hps interface for embedding table 2 of wdl: {}, the vectors: {}\n\n".format(
            embedding2[0].shape, embedding2
        )
    )
    return embedding1, embedding2


if __name__ == "__main__":
    model_name = sys.argv[1]
    embedding_file_list = str(sys.argv[2]).split(",")
    print(embedding_file_list)
    data_file = sys.argv[3]
    d1, d2 = hps_dlpack(
        model_name,
        embedding_file_list,
        data_file,
        True,
        hugectr.inference.EmbeddingCacheType_t.Dynamic,
    )
    u1, u2 = hps_dlpack(
        model_name, embedding_file_list, data_file, True, hugectr.inference.EmbeddingCacheType_t.UVM
    )
    s1, s2 = hps_dlpack(
        model_name,
        embedding_file_list,
        data_file,
        True,
        hugectr.inference.EmbeddingCacheType_t.Static,
    )
    diff = s2.reshape(1, 26 * 16) - d2.reshape(1, 26 * 16)
    if diff.mean() > 1e-3:
        raise RuntimeError(
            "Too large mse between Static cache and dynamic cache lookup result: {}".format(
                diff.mean()
            )
        )
        sys.exit(1)
    else:
        print(
            "The lookup results of Static cache are consistent with Dynamic cache, mse: {}".format(
                diff.mean()
            )
        )
    diff = u2.reshape(1, 26 * 16) - d2.reshape(1, 26 * 16)
    if diff.mean() > 1e-3:
        raise RuntimeError(
            "The lookup results of UVM cache are consistent with Dynamic cache: {}".format(
                diff.mean()
            )
        )
        sys.exit(1)
    else:
        print(
            "Pytorch dlpack on cpu  results are consistent with native HPS lookup api, mse: {}".format(
                diff.mean()
            )
        )
    # hps_dlpack(model_name, network_file, dense_file, embedding_file_list, data_file, False)
