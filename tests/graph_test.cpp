// #include "ctorch/c_torch.h"
// #include "gtest/gtest.h"

// /*
//   a -> b -> c
//             ^
//   d --------|
// */
// TEST(cTorchGraphTest, test_get_input_nodes) {
//   CTorchNode a = {
//       .node_type = CTH_NODE_TYPE_DATA,
//       .exe_status = CTH_NODE_EXE_STATUS_CLEAN,
//       .inbound_nodes = NULL,
//       .outbound_nodes = NewCTHList(ListTypeName(CTorchNode)),
//   };
//   CTorchNode b = {
//       .node_type = CTH_NODE_TYPE_DATA,
//       .exe_status = CTH_NODE_EXE_STATUS_CLEAN,
//       .inbound_nodes = NewCTHList(ListTypeName(CTorchNode)),
//       .outbound_nodes = NewCTHList(ListTypeName(CTorchNode)),
//   };
//   CTorchNode c = {
//       .node_type = CTH_NODE_TYPE_DATA,
//       .exe_status = CTH_NODE_EXE_STATUS_CLEAN,
//       .inbound_nodes = NewCTHList(ListTypeName(CTorchNode)),
//       .outbound_nodes = NewCTHList(ListTypeName(CTorchNode)),
//   };
//   CTorchNode d = {
//       .node_type = CTH_NODE_TYPE_DATA,
//       .exe_status = CTH_NODE_EXE_STATUS_CLEAN,
//       .inbound_nodes = NULL,
//       .outbound_nodes = NewCTHList(ListTypeName(CTorchNode)),
//   };
// }
