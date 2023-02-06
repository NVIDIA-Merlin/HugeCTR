#include <tensor2.hpp>

namespace HugeCTR {

template <typename DenseType, typename SparseType>
void split_3_way_feat_major(Tensor2<float> label_tensor, Tensor2<DenseType> dense_tensor,
                            Tensor2<SparseType*> sparse_tensors,
                            Tensor2<int> label_dense_sparse_tensor, Tensor2<int> bucket_ids,
                            Tensor2<int> bucket_positions, Tensor2<int> max_hotnesses,
                            cudaStream_t stream, bool is_dense_float = false);
}  // namespace HugeCTR
