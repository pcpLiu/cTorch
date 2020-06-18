from cffi import FFI

import sys
import os
curr_dir = os.path.abspath(os.path.dirname(__file__))
lib_dir = os.path.join(curr_dir, 'lib')
include_dir = os.path.join(curr_dir, 'include')
sys.path.append(lib_dir)
sys.path.append(include_dir)

ffibuilder = FFI()

ffibuilder.set_source(
    "_c_torch_lib",
    r"""
        #include <cTorch/c_torch.h>
    """,
    libraries=['cTorch']
)

ffibuilder.cdef(
    """
    typedef struct CTorchOperator {
    CTH_OP_ID op_id;                        /* Operator ID */
    List(CTorchTensor) * in_bound_tensors;  /* List of input tensors. It includes
                                                inputs, weight and arguments. */
    List(CTorchTensor) * out_bound_tensors; /* List of output tensors */
    bool is_sharded;                        /* If op is a sharded one */
    } CTorchOperator;
    """
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
