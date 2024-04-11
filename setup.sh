#!/bin/bash

# this is here to help set everything up easily when using a fresh runpod

pip install -r requirements.txt  &

git submodule init && git submodule update && chmod 777 data/LongHealth/serve.sh &

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh \
    | bash && apt-get install git-lfs && git lfs pull &

# clean up all the processes when we kill this
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT