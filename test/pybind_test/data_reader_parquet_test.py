from hugectr import Check_t, DataReaderSparse_t, DataReaderSparseParam, HeapEx, ParquetDataReaderWorker64, DataReader64, ResourceManager, cuda_memory_resource, device_memory_resource

slot_size = [
    39884406, 39043,    17289,    7420,     20263,  3,     7120, 1543, 63,
    38532951, 2953546,  403346,   10,       2208,   11938, 155,  4,    976,
    14,       39979771, 25641295, 39664984, 585935, 12972, 108,  36]
max_nnz = 1
slot_num = 26
label_dim = 1
dense_dim = 13
CHK = Check_t.Sum
prefix = "./data_reader_parquet_test_data/"
file_list_name = "./data_reader_parquet_test_data/data_reader_parquet_file_list.txt"
#using CVector = std::vector<std::unique_ptr<cudf::column>>;

def data_reader_worker_parquet_distributed_test():
  device_id = 0
  pool_alloc_size = 256 * 1024 * 1024
  dev = [device_id]
  mr = cuda_memory_resource()
  #  setup a CSR heap
  num_devices = 1
  batchsize = 128
  param = DataReaderSparseParam(DataReaderSparse_t.Distributed, max_nnz * slot_num, max_nnz, slot_num)
  params = []
  params.append(param)
  buffer_length = max_nnz
  csr_heap = HeapEx(1, num_devices, batchsize, label_dim + dense_dim, params)
  slot_offset = [0 for _ in range(len(slot_size))]
  for i in range(1, len(slot_size)):
    slot_offset[i] = slot_offset[i-1] + slot_size[i-1]
  # setup a data reader
  data_reader = ParquetDataReaderWorker64(0, 1, csr_heap, file_list_name, buffer_length,
                                          params, slot_offset, mr)
  # call read a batch
  data_reader.read_a_batch()
  print("************Parquet data reader workr successfully read a batch to device******************")

def data_reader_parquet_distributed_test():
  batchsize = 1024
  numprocs = 1
  device_list = [0, 1, 2, 3]
  vvgpu = []
  for i in range(numprocs):
    vvgpu.append(device_list)
  resource_manager = ResourceManager.create(vvgpu, 0)
  param = DataReaderSparseParam(DataReaderSparse_t.Distributed, max_nnz * slot_num, max_nnz, slot_num)
  params = []
  params.append(param)

  slot_offset = [0 for _ in range(len(slot_size))]
  for i in range(1, len(slot_size)):
    slot_offset[i] = slot_offset[i-1] + slot_size[i-1]

  data_reader = DataReader64(batchsize, label_dim, dense_dim, params, resource_manager, len(device_list));
  data_reader.create_drwg_parquet(file_list_name, slot_offset, True)
  data_reader.read_a_batch_to_device()
  data_reader.read_a_batch_to_device()
  print("**********data reader successfully read parquet format batches to device********************")


if __name__ == "__main__":
  data_reader_worker_parquet_distributed_test()
  data_reader_parquet_distributed_test()
