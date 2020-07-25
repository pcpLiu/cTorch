// #include "ctorch/c_torch.h"
// #include "gtest/gtest.h"

// /*
//   a -> b -> c
//             ^
//   d --------|
// */
// TEST(cTorchGraphTest, test_get_input_nodes) {
//   CTHNode a = {
//       .node_type = CTH_NODE_TYPE_DATA,
//       .exe_status = CTH_NODE_EXE_STATUS_CLEAN,
//       .inbound_nodes = NULL,
//       .outbound_nodes = NewCTHList(ListTypeName(CTHNode)),
//   };
//   CTHNode b = {
//       .node_type = CTH_NODE_TYPE_DATA,
//       .exe_status = CTH_NODE_EXE_STATUS_CLEAN,
//       .inbound_nodes = NewCTHList(ListTypeName(CTHNode)),
//       .outbound_nodes = NewCTHList(ListTypeName(CTHNode)),
//   };
//   CTHNode c = {
//       .node_type = CTH_NODE_TYPE_DATA,
//       .exe_status = CTH_NODE_EXE_STATUS_CLEAN,
//       .inbound_nodes = NewCTHList(ListTypeName(CTHNode)),
//       .outbound_nodes = NewCTHList(ListTypeName(CTHNode)),
//   };
//   CTHNode d = {
//       .node_type = CTH_NODE_TYPE_DATA,
//       .exe_status = CTH_NODE_EXE_STATUS_CLEAN,
//       .inbound_nodes = NULL,
//       .outbound_nodes = NewCTHList(ListTypeName(CTHNode)),
//   };
// }
