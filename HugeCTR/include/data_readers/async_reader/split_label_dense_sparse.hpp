#include <tensor2.hpp>

namespace HugeCTR {

template <typename DenseType, typename SparseType>
void split_3_way(Tensor2<float> label_tensor_per_dev, Tensor2<DenseType> dense_tensor_per_dev,
                 Tensor2<SparseType> sparse_tensor, Tensor2<int> label_dense_sparse_buffer,
                 size_t local_idx_start, size_t local_idx_end,
                 cudaStream_t stream);
}
