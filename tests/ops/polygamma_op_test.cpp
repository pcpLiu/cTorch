// #include "cTorch/c_torch.h"
// #include "tests/test_util.h"
// #include "tests/torch_util.hpp"
// #include "gtest/gtest.h"

// /**
//  * @note We make the precision (1e-3). Otherwise, it often fails due to
//  * precision issues
//  */

// torch::Tensor _polygamma_torch(torch::Tensor py_tensor) {
//   return py_tensor.polygamma();
// }

// void test_polygamma(CTH_BACKEND backend, CTH_TENSOR_DATA_TYPE data_type,
//                     float min, float max) {
//   tensor_dim_t dims[] = {100, 100};
//   tensor_dim_t n_dim = sizeof(dims) / sizeof(dims[0]);
//   CTorchNode *op_node = create_dummy_op_node_unary(CTH_OP_ID_polygamma, dims,
//                                                    n_dim, data_type, min,
//                                                    max);
//   CTorchOperator *op = op_node->conent.op;

//   if (backend == CTH_BACKEND_DEFAULT) {
//     op_polygamma_cpu(op);
//   }

//   sample_print(data_type,
//                array_at(CTorchTensor)(op->in_bound_tensors, 0)->values,
//                array_at(CTorchTensor)(op->out_bound_tensors, 0)->values, 2);

//   if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_16 ||
//       data_type == CTH_TENSOR_DATA_TYPE_FLOAT_32) {
//     _ele_wise_equal_unary_pytorch(op, float, EXPECT_EQ_PRECISION, 1e-3,
//                                   _polygamma_torch);
//   } else if (data_type == CTH_TENSOR_DATA_TYPE_FLOAT_64) {
//     _ele_wise_equal_unary_pytorch(op, double, EXPECT_EQ_PRECISION, 1e-3,
//                                   _polygamma_torch);
//   } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_16) {
//     _ele_wise_equal_unary_pytorch(op, int16_t, EXPECT_EQ_PRECISION, 1e-3,
//                                   _polygamma_torch);
//   } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_32) {
//     _ele_wise_equal_unary_pytorch(op, int32_t, EXPECT_EQ_PRECISION, 1e-3,
//                                   _polygamma_torch);
//   } else if (data_type == CTH_TENSOR_DATA_TYPE_INT_64) {
//     _ele_wise_equal_unary_pytorch(op, int64_t, EXPECT_EQ_PRECISION, 1e-3,
//                                   _polygamma_torch);
//   }
// }

// TEST(cTorchPolygammaOpTest, testFloat16Default) {
//   test_polygamma(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_16, 1.0,
//                  100.0);
// }

// TEST(cTorchPolygammaOpTest, testFloat32Default) {
//   test_polygamma(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_32, 1.0,
//                  100.0);
// }

// TEST(cTorchPolygammaOpTest, testFloat64Default) {
//   test_polygamma(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_FLOAT_64, 1.0,
//                  100.0);
// }

// TEST(cTorchPolygammaOpTest, testInt16Default) {
//   test_polygamma(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_16, 1.0,
//   100.0);
// }

// TEST(cTorchPolygammaOpTest, testInt32Default) {
//   test_polygamma(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_32, 1.0,
//   100.0);
// }

// TEST(cTorchPolygammaOpTest, testInt64Default) {
//   test_polygamma(CTH_BACKEND_DEFAULT, CTH_TENSOR_DATA_TYPE_INT_64, 1.0,
//   100.0);
// }
