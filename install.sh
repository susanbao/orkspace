#!/bin/bash
#cd ./orkspace
sudo apt install build-essential;
echo "****************************************************************************************************************************************"
echo "Complete essential build for DINO"
echo "****************************************************************************************************************************************"

cd ./DINO
ls
pip install -r requirements.txt
pip install opencv-python
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
pip install pandas
pip install seaborn
pip install ipdb
apt install unzip
apt install zip
pip3 install -U scikit-learn

# for ViT
conda install -y -c conda-forge tensorboard
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
pip install ml-collections
pip install wandb
pip install sockets
echo "****************************************************************************************************************************************"
echo "Complete installing lib for python"
echo "****************************************************************************************************************************************"

cd models/dino/ops
python setup.py build install
sh ./make.sh
echo "****************************************************************************************************************************************"
echo "Complete installing setup!!!"
echo "****************************************************************************************************************************************"

git config --global user.email "susannju@163.com"
git config --global user.name "susanbao"

# cd /workspace/orkspace/nuimages
# ls
# mkdir val2017
# mkdir train2017
# ln -s /workspace/samples /workspace/orkspace/nuimages/val2017/
# ln -s /workspace/samples /workspace/orkspace/nuimages/train2017/
# ln -s /workspace/coco /workspace/orkspace/
# echo "****************************************************************************************************************************************"
# echo "Complete soft link data set"
# echo "****************************************************************************************************************************************"


# cd /workspace/orkspace/DINO
# mkdir ckpts
# cd ./ckpts
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1cJ6WWAYrYjK2xGpv_x1-EqKMeNNW5R0h' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1cJ6WWAYrYjK2xGpv_x1-EqKMeNNW5R0h" -O checkpoint0023_4scale.pth && rm -rf /tmp/cookies.txt
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1eeAHgu-fzp28PGdIjeLe-pzGPMG2r2G_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1eeAHgu-fzp28PGdIjeLe-pzGPMG2r2G_" -O checkpoint0011_4scale.pth && rm -rf /tmp/cookies.txt
# echo "****************************************************************************************************************************************"
# echo "Download example checkpoint checkpoint0023_4scale.pth"
# echo "****************************************************************************************************************************************"

