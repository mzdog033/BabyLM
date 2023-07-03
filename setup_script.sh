%%shell

# Remove previous installation if it exists
!mkdir model_folder
!pip uninstall -y lm-eval
!rm -rf evaluation-pipeline/

# Install evaluation-pipeline
!git clone -b colab https://github.com/babylm/evaluation-pipeline &> /dev/null
%cd evaluation-pipeline/
!pip install -e ".[colab]"

# Install other necessary packages
!pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
!pip install sentencepiece==0.1.94
!pip install transformers

# Unpack dataset
!unzip filter_data.zip
