#script to extract ImageNet dataset
# ILSVRC2012_img_train.tar (about 138 GB)
# ILSVRC2012_img_val.tar (about 6.3 GB)
# make sure ILSVRC2012_img_train.tar & ILSVRC2012_img_val.tar in your current directory
#
#  Adapted from:
#  https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md
#  https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4
# 
#  imagenet/train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#  imagenet/val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
#
#
# Make imagnet directory
#
#wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

mkdir imagenet
#
# Extract the training data:
#
# Create train directory; move .tar file; change directory
mkdir ImageNet-ILSVRC2012/train && cd ImageNet-ILSVRC2012/train


# && rm -f ILSVRC2012_img_train.tar
cd ILSVRC2012_img_train
#
# At this stage imagenet/train will contain 1000 compressed .tar files, one for each category
#
# For each .tar file: 
#   1. create directory with same name as .tar file
#   2. extract and copy contents of .tar file into directory
#   3. remove .tar file
find . -name "*.tar" | while read NAME ; do mkdir -p "../ImageNet-ILSVRC2012/train/${NAME%.tar}"; tar -xvf "${NAME}" -C "../ImageNet-ILSVRC2012/train/${NAME%.tar}"; done
#
# This results in a training directory like so:
#
#  imagenet/train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#
# Change back to original directory
cd ../
#
# Extract the validation data and move images to subfolders:
#
# Create validation directory; move .tar file; change directory; extract validation .tar; remove compressed file
mkdir ImageNet-ILSVRC2012/val && tar -xvf ILSVRC2012_img_val.tar -C "ImageNet-ILSVRC2012/val/"

cd ImageNet-ILSVRC2012/val/
# && rm -f ILSVRC2012_img_val.tar
# get script from soumith and run; this script creates all class directories and moves images into corresponding directories
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
#
# This results in a validation directory like so:
#
#  imagenet/val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
#
#
# Check total files after extract
#
#  $ find train/ -name "*.JPEG" | wc -l
#  1281167
#  $ find val/ -name "*.JPEG" | wc -l
#  50000
#
