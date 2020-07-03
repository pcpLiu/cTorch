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
sudo aws configure set aws_access_key_id AKIATGUE5JHV5TX52MJP
sudo aws configure set aws_secret_access_key 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


################################################################################
#
# Pull MKL lib for tests
#
mkdir third_party/intel_mkl
sudo aws s3 sync s3://ctorch.github/intel_mkl third_party/intel_mkl

