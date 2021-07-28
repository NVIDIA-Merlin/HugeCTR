#include <tensor2.hpp>

namespace HugeCTR {

template <typename DenseType, typename SparseType>
void split_3_way(Tensor2<float> label_tensor, Tensor2<DenseType> dense_tensor,
                 Tensor2<SparseType> sparse_tensor, Tensor2<int> label_dense_sparse_buffer,
                 cudaStream_t stream);
}
