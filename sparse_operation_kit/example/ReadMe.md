# How to use SOK

### Requirements
this folder list how to use SOK to train a recsys

### items

sok_dump_load  sok_incremental_dump  sok_train_hps_inference
1. **SOK train and hps inference**  
   This test is located in the <mark>sok_train_hps_inference</mark> folder. Please read the ReadMe inside the folder to complete the test. <mark>sok_train.py</mark> is used for multi-GPU training of a recsys model using SOK, and it exports the trained model, which can then be converted into a format usable by HPS. <mark>hps_use_sok_weight_inference.py</mark> uses the weights converted by sok_train.py for inference.  
   If you want to deploy to the Triton server, please refer to the example of HPS's Triton backend deployment:[Example of HPS's Triton backend deployment](https://github.com/NVIDIA-Merlin/HugeCTR/blob/main/hps_tf/notebooks/hps_tensorflow_triton_deployment_demo.ipynb)  
2. **SOK dump load test**  
   This test is located in the <mark>sok_dump_load</mark> folder.This example demonstrates the use of SOK's dump and load functionalities. The process in the example includes dumping the weight, setting the weight to zero, loading the weight, and then comparing the weight values before and after. This allows for verifying the correctness of the dump and load operations.   

3. **SOK incremental model dump test** 
   This test is located in the <mark>sok_incremental_dump</mark> folder.This example demonstrates the use of SOK's incremental model dump functionalities. Currently, SOK can export the keys and values of an incremental model to a numpy array. If you want to integrate with an inference framework, you may need to customize an interface to accept numpy arrays.  
   For HPS, we can accomplish this by modifying and warp the existing API at [HPS kafka c++ API](https://github.com/NVIDIA-Merlin/HugeCTR/blob/main/HugeCTR/include/hps/kafka_message.hpp#L60) 
