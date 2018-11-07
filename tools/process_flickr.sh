## This code is modified from Hengyuan Hu's repository.
## https://github.com/hengyuan-hu/bottom-up-attention-vqa

## Process data
## Notice that 10-100 adaptive bottom-up attention features are used.

python3 tools/create_dictionary.py --task flickr
python3 tools/adaptive_detection_features_converter.py --task flickr