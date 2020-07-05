################################################################################
#
# Pull MKL lib for tests
#
cd third_party/
mkdir intel_mkl && cd intel_mkl

if [ "$RUNNER_OS" == "Linux" ]; then
    aws s3 cp s3://ctorch.github/intel_mkl/linux.zip .
    unzip linux.zip
else
    aws s3 cp s3://ctorch.github/intel_mkl/mac.zip .
    unzip mac.zip
fi
