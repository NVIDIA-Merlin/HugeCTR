import sys
from hugectr.inference import InferenceParams, CreateInferenceSession
from mpi4py import MPI

def dcn_inference(config_file, model_name, data_path, use_gpu_embedding_cache):
  # read data from file
  data_file = open(data_path)
  labels = [int(item) for item in data_file.readline().split(' ')]
  dense_features = [float(item) for item in data_file.readline().split(' ')]
  embedding_columns = [int(item) for item in data_file.readline().split(' ')]
  row_ptrs = [int(item) for item in data_file.readline().split(' ')]
  # create parameter server, embedding cache and inference session
  inference_params = InferenceParams(model_name = model_name,
                                  max_batchsize = 4096,
                                  hit_rate_threshold = 0.6,
                                  dense_model_file = "/hugectr/test/utest/_dense_10000.model",
                                  sparse_model_files = ["/hugectr/test/utest/0_sparse_10000.model"],
                                  device_id = 0,
                                  use_gpu_embedding_cache = use_gpu_embedding_cache,
                                  cache_size_percentage = 0.9,
                                  i64_input_key = False)
  inference_session = CreateInferenceSession(config_file, inference_params)
  # make prediction and calculate accuracy
  output = inference_session.predict(dense_features, embedding_columns, row_ptrs)
  accuracy = calculate_accuracy(labels, output)
  if use_gpu_embedding_cache:
    print("[HUGECTR][INFO] Use gpu embedding cache, prediction number samples: {}, accuracy: {}".format(len(labels), accuracy))
  else:
    print("[HUGECTR][INFO] Use cpu parameter server, prediction number samples: {}, accuracy: {}".format(len(labels), accuracy))

def calculate_accuracy(labels, output):
  num_samples = len(labels)
  flags = [1 if ((labels[i] == 0 and output[i] <= 0.5) or (labels[i] == 1 and output[i] > 0.5)) else 0 for i in range(num_samples)]
  correct_samples = sum(flags)
  return float(correct_samples)/float(num_samples)
    
if __name__ == "__main__":
  config_file = sys.argv[1]
  model_name = sys.argv[2]
  data_path = sys.argv[3]
  dcn_inference(config_file, model_name, data_path, True)
  dcn_inference(config_file, model_name, data_path, False)
