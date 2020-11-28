from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

import numpy as np

model_name = "hugectr_tf"
dense =[0.01612903, 0.00555074,   0.000777, 0.00930233, 0.00431766, 0.00263806,
        0.05597723,  0.0295858, 0.05496536,        0.2,        0.2, 0.00490196,
        0.00431034]

keys = [      32,     1488,  160036,  375021,  549216,  549416,  550554,  561592,
           562203,  587430,  616088,  945152,  954332,  954601,  958552, 1140402,
          1268018, 1269708, 1273761, 1274951, 1504094, 1599234, 1599247, 1619112,
          1679079, 1700863]


with httpclient.InferenceServerClient("10.19.226.179:8000") as client:
    input0_data = np.array(keys).astype(np.int64)
    input1_data = np.array(dense).astype(np.float32)
    inputs = [
        httpclient.InferInput("INPUT0", input0_data.shape,
                              np_to_triton_dtype(input0_data.dtype)),
        httpclient.InferInput("INPUT1", input1_data.shape,
                              np_to_triton_dtype(input1_data.dtype)),
    ]

    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)

    result = response.get_response()
    print("INPUT0: {}".format(
        input0_data))
    print("OUTPUT0: {}".format(response.as_numpy("OUTPUT0")))

