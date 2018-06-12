## This code is from Hengyuan Hu's repository.
## https://github.com/hengyuan-hu/bottom-up-attention-vqa

## Script for downloading data

# VQA Input Images

wget -P data http://msvocds.blob.core.windows.net/coco2014/train2014.zip
unzip data/train2014.zip -d data/
rm data/train2014.zip

wget -P data http://msvocds.blob.core.windows.net/coco2014/val2014.zip
unzip data/val2014.zip -d data/
rm data/val2014.zip

wget -P data http://msvocds.blob.core.windows.net/coco2015/test2015.zip
unzip data/test2015.zip -d data/
rm data/test2015.zip

