HAEDER_DEF = """
typedef struct CTorchTensorMeta {
  uint8_t value_size_of;          /* Element size */
  CTH_TENSOR_DATA_TYPE data_type; /* Value data type */
  tensor_dim_t n_dim;             /* Number of dimensions */
  tensor_dim_t *dims;             /* Dimension array */
  tensor_size_t n_elements;       /* Number of elements */
  uint16_t align_size;            /* Alignment size of this storage */
  CTH_TENSOR_TYPE type;           /* Tensor type: normal or params */
  bool is_sharded;   /* If this tensor is a sharding piece of another tensor */
  char *tensor_name; /* For CTH_TENSOR_TYPE_PARAM type node, this is parameter
                        name. As for other types, this is an optiona field and
                        could be null */
} CTorchTensorMeta;


"""