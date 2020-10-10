from hugectr import Check_t, DataReaderSparse_t, DataReaderSparseParam, HeapEx, DataReader64, DataReaderWorkerRaw64, MmapOffsetList, ResourceManager

slot_size = [39884406, 39043,    17289,    7420,     20263,  3,     7120, 1543, 63,
             38532951, 2953546,  403346,   10,       2208,   11938, 155,  4,    976,
             14,       39979771, 25641295, 39664984, 585935, 12972, 108,  36]

slot_offset = [0,        39884406, 39923449,  39940738,  39948158,  39968421,  39968424,  39975544, 39977087,
               39977150, 78510101, 81463647,  81866993,  81867003,  81869211,  81881149,  81881304, 81881308,
               81882284, 81882298, 121862069, 147503364, 187168348, 187754283, 187767255, 187767363]
num_samples = 131072 * 12
max_nnz = 1
slot_num = 26
label_dim = 1
dense_dim = 13
CHK = Check_t.Sum
file_name = "./train_data.bin"
float_label_dense = False

def data_reader_worker_raw_test():
  num_devices = 1
  batchsize = 2048
  param = DataReaderSparseParam(DataReaderSparse_t.Distributed, max_nnz * slot_num, 1, slot_num)
  params = []
  params.append(param)

  csr_heap = HeapEx(1, num_devices, batchsize, label_dim+dense_dim, params)
  file_offset_list = MmapOffsetList(file_name, num_samples, (label_dim + dense_dim + slot_num) * 4, batchsize, False, 1)
  data_reader = DataReaderWorkerRaw64(0, 1, file_offset_list, csr_heap, file_name, params,
                                      slot_offset, label_dim, float_label_dense)
  print(data_reader)
  data_reader.read_a_batch()
  print(data_reader)


def data_reader_raw_test():
  batchsize = 131072
  numprocs = 1
  pid = 0
  vvgpu = []
  device_list = [0, 1]
  for i in range(numprocs):
    vvgpu.append(device_list)
  gpu_resource_group = ResourceManager.create(vvgpu, 0)
  param = DataReaderSparseParam(DataReaderSparse_t.Localized, max_nnz * slot_num, 1, slot_num)
  params = []
  params.append(param)
  data_reader = DataReader64(batchsize, label_dim, dense_dim, params, gpu_resource_group, 1, True, False)
  data_reader.create_drwg_raw(file_name, num_samples, slot_offset, float_label_dense, True, True)
  current_batchsize = data_reader.read_a_batch_to_device()
  print("current batchsize: {}".format(current_batchsize))
  #data_reader.get_label_tensors()
  #data_reader.get_value_tensors()
  #data_reader.get_row_offsets_tensors()
  #data_reader.get_label_tensors()
  #data_reader.get_dense_tensors()
  #data_reader.get_value_tensors()
  #data_reader.get_row_offsets_tensors()
  
  current_batchsize = data_reader.read_a_batch_to_device()
  print("current batchsize: {}".format(current_batchsize))
  #data_reader.get_label_tensors()
  #data_reader.get_value_tensors()
  #data_reader.get_row_offsets_tensors()

  current_batchsize = data_reader.read_a_batch_to_device()
  print("current batchsize: {}".format(current_batchsize))
  #data_reader.get_value_tensors()
  #data_reader.get_row_offsets_tensors()
  #data_reader.get_value_tensors()
  #data_reader.get_row_offsets_tensors()
  
  current_batchsize = data_reader.read_a_batch_to_device()
  print("current batchsize: {}".format(current_batchsize))
  #data_reader.get_label_tensors()
  #data_reader.get_value_tensors()
  #data_reader.get_row_offsets_tensors()


if __name__ == "__main__":
  data_reader_worker_raw_test()
  data_reader_raw_test()
