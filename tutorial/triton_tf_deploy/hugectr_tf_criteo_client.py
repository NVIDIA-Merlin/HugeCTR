from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import numpy as np

model_name = "hugectr_tf"
keys = [      2,     365,    5840,    8106,   11025,  144676,  150012,  151981,
           152540,  156528,  156541,  156678,  156873,  157293,  158801,  159375,
           430985,  609983,  610294,  612740,  622307,  622941,  627045,  671224,
           675962,  946084,  948795,  949311,  960691, 1208214, 1208284, 1212984,
          1215053, 1215132, 1479013, 1479030, 1479060, 1550398, 1550501]


with httpclient.InferenceServerClient("10.19.226.179:8000") as client:
    input0_data = np.array(keys).astype(np.int64)
    inputs = [
        httpclient.InferInput("INPUT0", input0_data.shape,
                              np_to_triton_dtype(input0_data.dtype)),
    ]

    inputs[0].set_data_from_numpy(input0_data)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    result = response.get_response()
    print("INPUT0: {}".format(input0_data))
    print("OUTPUT0: {}".format(response.as_numpy("OUTPUT0")))

