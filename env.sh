if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
. venv/bin/activate
python3 -m pip install --upgrade pip

dversion="2.2.1+cu121"
version=$(python3 -m pip show torch | grep Version | awk '{print $2}')
if [ -z "$version" ]; then
	python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
	if [ "$version" == "$dversion" ]; then
		echo "Good version installed"
	else
		python3 -m pip uninstall -y torch torchvision torchaudio
		python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	fi
fi


python3 -m pip install pytorch-metric-learning
python3 -m pip install torch_geometric
python3 -m pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.2.0+cu121.html

python3 -m pip install uproot
python3 -m pip install matplotlib
python3 -m pip install numpy
python3 -m pip install scipy
python3 -m pip install onnx onnxruntime
python3 -m pip install skl2onnx
