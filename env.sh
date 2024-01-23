if [ ! -d "venv" ]; then
    python -m venv venv
fi
. venv/bin/activate
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install pytorch-metric-learning
python -m pip install torch_geometric
python -m pip install torch-spline-conv

python -m pip install uproot
python -m pip install matplotlib
python -m pip install numpy 

