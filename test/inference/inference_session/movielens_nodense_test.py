from hugectr.inference import InferenceParams, CreateInferenceSession
import pandas as pd
import numpy as np
import sys


# from mpi4py import MPI
def movie_inference(
    model_name, network_file, dense_file, embedding_file_list, data_file, enable_cache
):
    CATEGORICAL_COLUMNS = ["userId", "movieId"]
    LABEL_COLUMNS = ["rating"]
    emb_size = [162542, 56586]
    shift = np.insert(np.cumsum(emb_size), 0, 0)[:-1]
    result = [
        0.8336379528045654,
        0.24868586659431458,
        0.4039016664028168,
        0.9553083777427673,
        0.6617599725723267,
        0.5613522529602051,
        0.16344544291496277,
        0.537512481212616,
        0.5185080766677856,
        0.2947561740875244,
    ]

    test_df = pd.read_parquet(data_file)
    config_file = network_file
    row_ptrs = list(range(0, 21))
    dense_features = []
    test_df[CATEGORICAL_COLUMNS].astype(np.int64)
    embedding_columns = list((test_df.head(10)[CATEGORICAL_COLUMNS] + shift).values.flatten())

    # create parameter server, embedding cache and inference session
    inference_params = InferenceParams(
        model_name=model_name,
        max_batchsize=64,
        hit_rate_threshold=1.0,
        dense_model_file=dense_file,
        sparse_model_files=embedding_file_list,
        device_id=0,
        use_gpu_embedding_cache=enable_cache,
        cache_size_percentage=0.9,
        i64_input_key=True,
        use_mixed_precision=False,
    )
    inference_session = CreateInferenceSession(config_file, inference_params)
    output1 = inference_session.predict(dense_features, embedding_columns, row_ptrs)
    miss1 = np.mean((np.array(output1) - np.array(result)) ** 2)
    inference_session.refresh_embedding_cache()
    output2 = inference_session.predict(dense_features, embedding_columns, row_ptrs)
    miss2 = np.mean((np.array(output2) - np.array(result)) ** 2)
    print("Movielens model(no dense input) inference result should be {}".format(result))
    miss = max(miss1, miss2)
    if enable_cache:
        if miss > 0.0001:
            raise RuntimeError(
                "Movielens model(no dense input) inference using GPU cache, prediction error is greater than threshold: {}, error is {}".format(
                    0.0001, miss
                )
            )
            sys.exit(1)
        else:
            print(
                "[HUGECTR][INFO] Movielens model(no dense input) inference using GPU cache, prediction error is less  than threshold:{}, error is {}".format(
                    0.0001, miss
                )
            )
    else:
        if miss > 0.0001:
            raise RuntimeError(
                "[HUGECTR][INFO] Movielens model(no dense input) inference without GPU cache, prediction error is greater than threshold:{}, error is {}".format(
                    0.0001, miss
                )
            )
            sys.exit(1)
        else:
            print(
                "[HUGECTR][INFO] Movielens model(no dense input) inference without GPU cache, prediction error is less than threshold: {}, error is {}".format(
                    0.0001, miss
                )
            )


if __name__ == "__main__":
    model_name = sys.argv[1]
    network_file = sys.argv[2]
    dense_file = sys.argv[3]
    embedding_file_list = str(sys.argv[4]).split(",")
    print(embedding_file_list)
    data_file = sys.argv[5]
    movie_inference(model_name, network_file, dense_file, embedding_file_list, data_file, True)
    movie_inference(model_name, network_file, dense_file, embedding_file_list, data_file, False)
