################################################################################
#
# Pull MKL lib for tests
#
mkdir third_party/intel_mkl
aws s3 sync s3://ctorch.github/intel_mkl third_party/intel_mkl

