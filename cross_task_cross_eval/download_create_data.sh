# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

mkdir data
cd data
wget https://github.com/allenai/gooaq/raw/main/data/gooaq.jsonl
gdown 1X5GoVi_OcRxahXH1pRW7TSesZUeMH3ss
wget https://nlp.cs.washington.edu/ambigqa/data/ambignq_light.zip

tar xvzf linkso.tar.gz
unzip ambignq_light

python create_cross_task_data.py