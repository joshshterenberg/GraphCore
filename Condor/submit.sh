#!/bin/bash

cd /eos/user/j/jshteren/OCToPi

if [ ! -d "venv" ]; then
    python -m venv venv
fi
. venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install pytorch-metric-learning
python -m pip install torch_geometric
python -m pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

python -m pip install uproot
python -m pip install matplotlib
python -m pip install numpy
python -m pip install scipy

python /eos/user/j/jshteren/OCToPi/mlp.py
python /eos/user/j/jshteren/OCToPi/gnn.py

