## This code is modified from Hengyuan Hu's repository.
## https://github.com/hengyuan-hu/bottom-up-attention-vqa

## Script for downloading data

# GloVe Vectors
wget -P data http://nlp.stanford.edu/data/glove.6B.zip
unzip data/glove.6B.zip -d data/glove
rm data/glove.6B.zip

# Questions
wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
unzip data/v2_Questions_Train_mscoco.zip -d data
rm data/v2_Questions_Train_mscoco.zip

wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip data/v2_Questions_Val_mscoco.zip -d data
rm data/v2_Questions_Val_mscoco.zip

wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip data/v2_Questions_Test_mscoco.zip -d data
rm data/v2_Questions_Test_mscoco.zip

# Annotations
wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip data/v2_Annotations_Train_mscoco.zip -d data
rm data/v2_Annotations_Train_mscoco.zip

wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip data/v2_Annotations_Val_mscoco.zip -d data
rm data/v2_Annotations_Val_mscoco.zip

# Image Features
wget -P data https://imagecaption.blob.core.windows.net/imagecaption/trainval.zip
wget -P data https://imagecaption.blob.core.windows.net/imagecaption/test2014.zip
wget -P data https://imagecaption.blob.core.windows.net/imagecaption/test2015.zip
unzip data/trainval.zip -d data
unzip data/test2014.zip -d data
unzip data/test2015.zip -d data
rm data/trainval.zip
rm data/test2014.zip
rm data/test2015.zip

# Download Pickle caches for the pretrained model from
# https://drive.google.com/file/d/1m5pL9gOkcnLZ_NuANmnDFIcil3NQVmZc/view?usp=sharing
# and extract pkl files under data/cache/.
mkdir -p data/cache
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1m5pL9gOkcnLZ_NuANmnDFIcil3NQVmZc' -O data/cache/cache.pkl.tgz
tar xvf data/cache/cache.pkl.tgz -C data/cache/
rm data/cache/cache.pkl.tgz

