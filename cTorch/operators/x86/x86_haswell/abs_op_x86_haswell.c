#include <immintrin.h>
#include <math.h>

#include "src/operators/abs_op.h"

/*
  Computes the element-wise absolute value of the given input tensor.
*/
void abs_op_x86_haswell(CTHOperator *op) {
  FORCE_INPUT_OUTPUT_TSR_NUM_EQ(op);
  CTHListItem(CTHTensor) *in = op->in_bound_tensors->head;
  CTHListItem(CTHTensor) *out = op->out_bound_tensors->head;
  CTH_TENSOR_DATA_TYPE dtype = in->data->meta_info->data_type;

  for (int i = 0; i < op->in_bound_tensors->size; i++) {
    // get data type bits length
    switch (dtype) {
    case CTH_TENSOR_DATA_TYPE_INT_8: {
      // vectorize process
      int8_t *in_mem, *out_mem;
      in_mem = in->data->values.val_int8;
      out_mem = out->data->values.val_int8;
      int64_t n_loop = floor(in->data->meta_info->n_elements / 16);
      while (n_loop > 0) {
        __m128i in_16, out_16;
        in_16 = _mm_lddqu_si128((__m128i *)in_mem);
        out_16 = _mm_lddqu_si128((__m128i *)out_mem);
        out_16 = _mm_abs_epi8(in_16);
        _mm_store_si128((__m128i *)out_mem, out_16);
        in_mem += 16;
        out_mem += 16;
        n_loop--;
      }
      // process rest
      int64_t rest = in->data->meta_info->n_elements % 16;
      while (rest > 0) {
        *out_mem = abs(*in_mem);
        in_mem++;
        out_mem++;
        rest--;
      }
    } break;
    case CTH_TENSOR_DATA_TYPE_FLOAT_16:
      break;
    case CTH_TENSOR_DATA_TYPE_FLOAT_32:
      break;
    default:
      FAIL_EXIT("Unsupported data type.");
      break;
    }

    in = in->next_item;
    out = out->next_item;
  }
}

void _shift_abs() {}
