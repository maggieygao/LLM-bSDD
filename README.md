1. Clone the Repository
bashgit clone https://github.com/maggieygao/LLM-bSDD/.git
cd bsdd-vlm-evaluation
2. Python Environment Setup
Option A: Using Conda (Recommended)
bash# Create new conda environment
conda create -n bsdd-vlm python=3.10

# Activate environment
conda activate bsdd-vlm

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
Option B: Using venv
bash# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
3. Install Dependencies
bash# Install required packages
pip install -r requirements.txt
requirements.txt:
txtpandas>=2.0.0
openpyxl>=3.1.0
requests>=2.31.0
transformers>=4.36.0
accelerate>=0.25.0
torch>=2.1.0
ifcopenshell>=0.7.0
4. Verify CUDA Installation
python# Run this in Python to verify GPU access
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

Expected output:
```
CUDA available: True
CUDA version: 11.8
GPU device: NVIDIA GeForce RTX 3060
