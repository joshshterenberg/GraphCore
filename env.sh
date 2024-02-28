if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
. venv/bin/activate
python3 -m pip install --upgrade pip
#python3 -m pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#python3 -m pip install torch torchvision torchaudio
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install pytorch-metric-learning
python3 -m pip install torch_geometric
python3 -m pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.2.0+cu121.html

python3 -m pip install uproot
python3 -m pip install matplotlib
python3 -m pip install numpy
python3 -m pip install scipy
#python3 -m pip install cuml
#python3 -m pip install collections #part of python3 by default
#python3 -m pip install multiprocessing #part of python3 by default
