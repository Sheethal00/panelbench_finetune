#!/bin/bash

# Exit immediately if any command fails
set -e

echo "============================================="
echo "🚀 Starting YOLOX CPU Installation Workflow"
echo "============================================="

# Step 1: Set up system requirements & virtual environment
echo " -> Setting up virtual environment..."
sudo apt update && sudo apt install -y git build-essential
python3.11 -m venv yolox_env
source yolox_env/bin/activate

# Step 2: Install PyTorch for CPU
echo " -> Installing CPU-specific PyTorch..."
pip cache purge
pip3 install torch torchvision torchaudio

# Step 2b: Install this pipeline's own extra deps (mlflow, dotenv, pillow, pyyaml)
# into the same venv, since that's where `yolox` ends up living.
if [ -f "../requirements-extra.txt" ]; then
  echo " -> Installing panelbench_finetune extra requirements..."
  pip install -r ../requirements-extra.txt
fi

# Step 3: Clone the repository and enter directory
echo " -> Cloning YOLOX repository..."
if [ ! -d "YOLOX" ]; then
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
fi
cd YOLOX

# Step 4: Patch legacy dependency failure (onnx-simplifier's setup.py calls
# `git describe` for a version string, gets "unknown", and modern setuptools
# rejects that as an invalid PEP 440 version). onnx-simplifier is only
# needed for the later ONNX export step, not for training -- safe to drop
# here and install separately on whichever machine runs the export.
echo " -> Patching onnx-simplifier dependency..."
sed -i '/onnx-simplifier/d' setup.py
sed -i '/onnx-simplifier/d' requirements.txt

# Step 5: Install YOLOX bypassing build isolation (so its setup.py can see
# the torch we just installed into this same venv)
echo " -> Compiling and installing YOLOX..."
pip install -v -e . --no-build-isolation

echo "============================================="
echo "✅ Installation Completed! Starting Verification..."
echo "============================================="

# Step 6: Verification Checks
echo " -> Verification A (Version Check):"
python3 -c 'import yolox; print("   🎉 Success! YOLOX Version:", yolox.__version__)'

echo " -> Verification B (Core Utilities Check):"
python3 -c 'from yolox.utils import postprocess; print("   🎉 Success! Core utilities verified.")'

echo " -> Verification C (Downloading Weights & Running Demo):"
if [ ! -f "../yolox_nano.pth" ]; then
echo "    Downloading yolox_nano.pth..."
wget -O ../yolox_nano.pth https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_nano.pth
fi

python3 tools/demo.py image -n yolox-nano -c ../yolox_nano.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --device cpu --save_result

echo "============================================="
echo "🏁 Setup entirely verified and operational!"
echo "============================================="
