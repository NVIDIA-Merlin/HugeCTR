from hugectr import Check_t, DataReaderSparse_t, DataReaderType_t, DataReaderSparseParam, HeapEx, DataReaderWorker64, DataReader64, ResourceManager

num_files = 20
label_dim = 2
dense_dim = 64
slot_num = 10
num_records = 2048 * 2
max_nnz = 30
vocabulary_size = 511
# configurations for data_reader_worker
file_list_name = "./data_reader_test_data/data_reader_file_list.txt"
prefix = "./data_reader_test_data/temp_dataset_"
CHK = Check_t.Sum
def data_reader_worker_test():
  num_devices = 1
  batchsize = 2048;
  param = DataReaderSparseParam(DataReaderSparse_t.Distributed, max_nnz * slot_num, max_nnz, slot_num)
  params = []
  params.append(param)
  buffer_length = max_nnz
  csr_heap = HeapEx(1, num_devices, batchsize, label_dim+dense_dim, params)
  data_reader = DataReaderWorker64(0, 1, csr_heap, file_list_name, buffer_length, CHK, params)
  print(data_reader)
  data_reader.read_a_batch()
  print(data_reader)

def data_reader_simple_test():
  batchsize = 2048
  numprocs = 1
  pid = 0
  device_list = [0,1]
  vvgpu = []
  for i in range(numprocs):
    vvgpu.append(device_list)
  gpu_resource_group = ResourceManager.create(vvgpu, 0)
  param = DataReaderSparseParam(DataReaderSparse_t.Distributed, max_nnz * slot_num, max_nnz, slot_num)
  params = []
  params.append(param)
  data_reader = DataReader64(batchsize, label_dim, dense_dim, params, gpu_resource_group, 12)
  data_reader.create_drwg_norm(file_list_name, CHK)
  data_reader.read_a_batch_to_device()
  data_reader.read_a_batch_to_device()
  data_reader.read_a_batch_to_device()
  print("Finish reading three batches to device")

if __name__ == "__main__":
  data_reader_worker_test()
  data_reader_simple_test()
