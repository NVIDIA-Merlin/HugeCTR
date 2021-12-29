import hugectr
from hugectr.inference import InferenceParams, CreateInferenceSession
import pandas as pd
import numpy as np
import sys
#from mpi4py import MPI
def wdl_inference(model_name, network_file, dense_file, embedding_file_list, data_file,enable_cache):
    CATEGORICAL_COLUMNS=["C" + str(x) for x in range(1, 27)]+["C1_C2","C3_C4"]
    CONTINUOUS_COLUMNS=["I" + str(x) for x in range(1, 14)]
    LABEL_COLUMNS = ['label']
    emb_size = [202546, 18795, 14099, 6889, 18577, 4, 6349, 1247, 48, 186730, 71084, 66832, 11, 2158, 7415, 61, 4, 923, 15, 202617, 143251, 198823, 61025, 9057, 73, 34, 225812, 354963]
    shift = np.insert(np.cumsum(emb_size), 0, 0)[:-1]
    result=[0.05634006857872009, 0.04185676947236061, 0.007268941029906273, 0.10255379974842072, 0.14059557020664215, 0.011040309444069862, 0.005499477963894606, 0.24404558539390564, 0.012491216883063316, 0.005486942362040281]

    test_df=pd.read_csv(data_file)
    config_file = network_file
    row_ptrs = list(range(0,21))+list(range(0,261))
    dense_features =  list(test_df[CONTINUOUS_COLUMNS].values.flatten())
    test_df[CATEGORICAL_COLUMNS].astype(np.int64)
    embedding_columns = list((test_df[CATEGORICAL_COLUMNS]+shift).values.flatten())
    
    hash_map_database = hugectr.inference.VolatileDatabaseParams()
    rocksdb_database = hugectr.inference.PersistentDatabaseParams(path="/hugectr/test/utest/wdl_test_files/rocksdb")

    # create parameter server, embedding cache and inference session
    inference_params = InferenceParams(model_name = model_name,
                                max_batchsize = 64,
                                hit_rate_threshold = 1.0,
                                dense_model_file = dense_file,
                                sparse_model_files = embedding_file_list,
                                device_id = 0,
                                use_gpu_embedding_cache = enable_cache,
                                cache_size_percentage = 0.9,
                                i64_input_key = True,
                                use_mixed_precision = False,
                                number_of_worker_buffers_in_pool=4,
                                number_of_refresh_buffers_in_pool=1,
                                deployed_devices= [0],
                                default_value_for_each_table= [0.0,0.0],
                                volatile_db=hash_map_database,
                                persistent_db=rocksdb_database)
    inference_session = CreateInferenceSession(config_file, inference_params)
    # predict for the first time
    output1 = inference_session.predict(dense_features, embedding_columns, row_ptrs)
    miss1 = np.mean((np.array(output1) - np.array(result)) ** 2)
    # refresh emebdding cache, void operation since there is no update for the parameter server
    inference_session.refresh_embedding_cache()
    # predict for the second time
    output2 = inference_session.predict(dense_features, embedding_columns, row_ptrs)
    miss2 = np.mean((np.array(output2) - np.array(result)) ** 2)
    print("WDL multi-embedding table inference result should be {}".format(result))
    miss = max(miss1, miss2)
    if enable_cache:
        if miss > 0.0001:
            raise RuntimeError("WDL multi-embedding table inference using GPU cache, prediction error is greater than threshold: {}, error is {}".format(0.0001, miss))
            sys.exit(1)
        else:
            print("[HUGECTR][INFO] WDL multi-embedding table inference using GPU cache, prediction error is less  than threshold:{}, error is {}".format(0.0001, miss))
    else:
        if miss > 0.0001:
            raise RuntimeError("[HUGECTR][INFO] WDL multi-embedding table inference without GPU cache, prediction error is greater than threshold:{}, error is {}".format(0.0001, miss))
            sys.exit(1)
        else:
            print("[HUGECTR][INFO] WDL multi-embedding table inference without GPU cache, prediction error is less than threshold: {}, error is {}".format(0.0001, miss))

if __name__ == "__main__":
    model_name = sys.argv[1]
    network_file = sys.argv[2]
    dense_file = sys.argv[3]
    embedding_file_list = str(sys.argv[4]).split(',')
    print(embedding_file_list)
    data_file = sys.argv[5]
    #wdl_inference(model_name, network_file, dense_file, embedding_file_list, data_file, True, hugectr.Database_t.RocksDB)
    wdl_inference(model_name, network_file, dense_file, embedding_file_list, data_file, True)
    wdl_inference(model_name, network_file, dense_file, embedding_file_list, data_file, False)
