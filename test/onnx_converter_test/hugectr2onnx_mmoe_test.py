import hugectr
from mpi4py import MPI

from hugectr.inference import InferenceParams, CreateInferenceSession
import hugectr2onnx
import onnxruntime as ort
from utils import read_samples_for_mmoe, compare_array_approx
import numpy as np

from hugectr.inference import InferenceModel, InferenceParams
import numpy as np


graph_config = "/onnx_converter/graph_files/mmoe.json"
dense_model = "/onnx_converter/hugectr_models/mmoe_dense_2000.model"
sparse_models = ["/onnx_converter/hugectr_models/mmoe0_sparse_2000.model"]
onnx_model_path = "/onnx_converter/onnx_models/mmoe.onnx"
data_file = "./val/0.parquet"
batch_size = 16384
num_batches = 1
data_source = "./file_names_val.txt"

hugectr2onnx.converter.convert(onnx_model_path, graph_config, dense_model, True, sparse_models, "")

label, dense, keys = read_samples_for_mmoe(data_file, batch_size * num_batches, slot_num=32)
sess = ort.InferenceSession(onnx_model_path)
res = sess.run(
    output_names=[sess.get_outputs()[0].name],
    input_feed={sess.get_inputs()[0].name: dense, sess.get_inputs()[1].name: keys},
)
res = res[0].reshape(batch_size * num_batches)

inference_params = InferenceParams(
    model_name="mmoe",
    max_batchsize=batch_size,
    hit_rate_threshold=1.0,
    dense_model_file="/onnx_converter/hugectr_models/mmoe_dense_2000.model",
    sparse_model_files=["/onnx_converter/hugectr_models/mmoe0_sparse_2000.model"],
    device_id=0,
    use_gpu_embedding_cache=False,
    cache_size_percentage=1.0,
    i64_input_key=False,
    use_mixed_precision=False,
    use_cuda_graph=True,
)
inference_session = CreateInferenceSession(
    "/onnx_converter/graph_files/mmoe.json", inference_params
)
slot_size_array = [
    91,
    73622,
    17,
    1425,
    3,
    24,
    15,
    5,
    10,
    2,
    3,
    6,
    8,
    133,
    114,
    1675,
    6,
    6,
    51,
    38,
    8,
    47,
    10,
    9,
    10,
    3,
    4,
    7,
    5,
    2,
    52,
    9,
]
predictions = inference_session.predict(
    num_batches, data_source, hugectr.DataReaderType_t.Parquet, hugectr.Check_t.Non, slot_size_array
)
predictions = np.array(predictions).T[0]

compare_array_approx(res, predictions, "mmoe", 1e-2, 1e-1)
