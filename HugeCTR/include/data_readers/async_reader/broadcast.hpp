#include <cuda_runtime.h>

namespace HugeCTR {

void broadcast(float** dev_pointers, const bool* dev_p2p_accessible, int batch_size_floats,
               int num_dests, int src_id, cudaStream_t stream);
}