# SOK train HPS inferece example

### Requirements
It is recommended to use the nvcr image for testing. Currently, the Merlin image available at <mark>nvcr.io/nvidia/merlin/merlin-tensorflow:nightly</mark> includes both SOK and HPS installed. However, there has not been a recent release of HugeCTR, so the code is not up-to-date. If you want to use the latest environment, you can choose the <mark>nvcr.io/nvidia/tensorflow:24.03-tf2-py3</mark> image and install SOK and HPS for testing.

### How to test
Follow these steps to test:

1. **SOK train**  
   horovodrun -np 2 python sok_train.py

2. **HPS inference**  
   python hps_use_sok_weight_inference.py

