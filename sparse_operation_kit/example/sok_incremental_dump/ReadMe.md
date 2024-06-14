# SOK incremental_dump

### Requirements

SOK incremental_model_dump API is released recently , and merlin image don't have newest SOK code.  
so you should chooes the <mark>nvcr.io/nvidia/tensorflow:24.03-tf2-py3</mark> image and install SOK and HPS for testing.

### How to test
Follow these steps to test:

1. **SOK DUMP LOAD**  
   horovodrun -np 4 python incremental_dump_test.py 

### Note

1.  This is not a model example , it is SOK's unit test , it is easy check incremental_dump_test correctness.  
2.  The SOK incremental_model_dump currently cannot push data to the HPS Kafka service without modifying the HPS code (HPS does not currently offer a Python interface to receive numpy arrays, but support could be planned for the future). Currently, clients obtain incremental keys and values (numpy arrays) at the Python level and then send them to their custom Kafka service.
 
