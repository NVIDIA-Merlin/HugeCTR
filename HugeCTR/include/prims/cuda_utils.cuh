#include <math_constants.h>
#include <stdint.h>
#include <cuml/common/utils.hpp>

namespace MLCommon {

/** helper macro for device inlined functions */
#define DI inline __device__

template <typename ReduceLambda>
DI void myAtomicReduce(__half *address, __half val, ReduceLambda op) {
  //float *address_f = address;
  float val_f = val;
  unsigned int *address_as_uint = (unsigned int *)address;
  unsigned int old = *address_as_uint, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_uint, assumed,
                    __float_as_uint(op(val_f, __uint_as_float(assumed))));
  } while (assumed != old);
}

}  // namespace MLCommon