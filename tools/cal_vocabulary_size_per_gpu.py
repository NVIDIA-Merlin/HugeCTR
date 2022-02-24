import math
def cal_vocabulary_size_per_gpu_for_distributed_slot(total_vocabulary_size, num_gpus):
  return math.ceil(total_vocabulary_size / num_gpus)
 
def cal_vocabulary_size_per_gpu_for_localized_slot(slot_size_array, num_gpus):
  vocal_size_per_gpu = [0 for _ in range(num_gpus)]
  for i in range(len(slot_size_array)):
    vocal_size_per_gpu[i%num_gpus] += slot_size_array[i]
  return math.ceil(max(vocal_size_per_gpu))

if __name__ == "__main__":
  vvgpu = [[0, 1, 2, 3], [4, 5, 6, 7]]
  num_gpus = sum([len(local_gpu) for local_gpu in vvgpu])
  # DistributedSlotSparseEmbedding
  total_vocabulary_size = 210000
  voc_size_per_gpu_distributed = cal_vocabulary_size_per_gpu_for_distributed_slot(total_vocabulary_size, num_gpus) 
  # LocalizedSlotSparseEmbedding
  slot_size_array = [10000, 20000, 50000, 30000, 20000, 70000, 10000]
  voc_size_per_gpu_localized = cal_vocabulary_size_per_gpu_for_localized_slot(slot_size_array, num_gpus)

  print(f"DistributedSlotSparseEmbedding, total_vocabulary_size: {total_vocabulary_size}, num_gpus: {num_gpus}, voc_size_per_gpu_distributed: {voc_size_per_gpu_distributed}")
  print(f"LocalizedSlotSparseEmbedding, slot_size_array: {slot_size_array}, num_gpus: {num_gpus}, voc_size_per_gpu_distributed: {voc_size_per_gpu_localized}")
   
