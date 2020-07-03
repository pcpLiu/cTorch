################################################################################
#
# Pull MKL lib for tests
#
cd third_party/
aws s3 cp s3://ctorch.github/intel_mkl.zip .
unzip intel_mkl.zip
