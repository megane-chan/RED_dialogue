#/usr/bin/env bash

mkdir -v out 2>/dev/null

python3 -c "import pkg_resources; pkg_resources.require(open('requirements.txt',mode='r'))" &>/dev/null || \
    echo "Installing packages"
    pip3 install -q -r reqirements.txt 

time python3 swda_BERT_train.py 


# ./run.sh &> "v1_ep5_$(date "+%m%d_%H%M")