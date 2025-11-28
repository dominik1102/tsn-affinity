import torch
import sys
import subprocess

print("=== PYTHON ===")
print(sys.version)
print()

print("=== TORCH ===")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Built with CUDA:", torch.version.cuda)
print()

print("=== GPU INFO (nvidia-smi) ===")
try:
    out = subprocess.check_output(["nvidia-smi"]).decode()
    print(out)
except Exception as e:
    print("nvidia-smi not found or GPU not available:", e)
