#!/bin/bash
# cd scratch place
mkdir DeepFashion
  
# Download zip dataset from Google Drive
filename='train.zip'
fileid='1lQZOIkO-9L0QJuk_w1K8-tRuyno-KvLK'
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm ./cookie
  
# Unzip
unzip -q ${filename}
rm ${filename}
  
# cd out
cd