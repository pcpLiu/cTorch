set(ops_cuda_all
    operators/cuda/op_list_cuda.c
    operators/cuda/op_abs_cuda.cu
    operators/cuda/op_acos_cuda.cu
    operators/cuda/op_add_cuda.cu
    operators/cuda/op_asin_cuda.cu
    operators/cuda/op_atan2_cuda.cu
    operators/cuda/op_atan_cuda.cu
    operators/cuda/op_ceil_cuda.cu
    operators/cuda/op_cos_cuda.cu
    operators/cuda/op_cosh_cuda.cu
    operators/cuda/op_div_cuda.cu
    operators/cuda/op_erf_cuda.cu
    operators/cuda/op_erfc_cuda.cu
    operators/cuda/op_erfinv_cuda.cu
    operators/cuda/op_exp_cuda.cu
    operators/cuda/op_expm1_cuda.cu
    operators/cuda/op_floor_cuda.cu
    operators/cuda/op_fmod_cuda.cu
    operators/cuda/op_log10_cuda.cu
    operators/cuda/op_log1p_cuda.cu
    operators/cuda/op_log2_cuda.cu
    operators/cuda/op_log_cuda.cu
    operators/cuda/op_mul_cuda.cu
    operators/cuda/op_pow_cuda.cu
    operators/cuda/op_remainder_cuda.cu
    operators/cuda/op_round_cuda.cu
    operators/cuda/op_rsqrt_cuda.cu
    operators/cuda/op_sin_cuda.cu
    operators/cuda/op_sinh_cuda.cu
    operators/cuda/op_sqrt_cuda.cu
    operators/cuda/op_tan_cuda.cu
    operators/cuda/op_tanh_cuda.cu
    operators/cuda/op_trunc_cuda.cu
)

# Remove disabled operators
foreach(disabled_op ${ops_disabled})
    list(FILTER ${ops_cuda_all}
        EXCLUDE
        REGEX ^src/operators/cuda/${disabled_op}_cuda.c$
    )
endforeach()

################################################################################
#
# Export
#
set(ops_cuda
    ${ops_cuda_all}
    PARENT_SCOPE
)
