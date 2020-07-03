################################################################################
#
# Install cli
#

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

################################################################################
#
# CP credential
#

if [ ! -d "~/.aws" ]; then
    mkdir ~/.aws
fi
cp ci/aws_credential ~/.aws/credentials



################################################################################
#
# Pull MKL lib for tests
#
mkdir third_party/intel_mkl
aws s3 sync s3://ctorch.github/intel_mkl third_party/intel_mkl

