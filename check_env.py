import torch
import pennylane as qml
import numpy as np
import pandas as pd

# Kiểm tra PyTorch có nhận GPU không
print(f"PyTorch version: {torch.__version__}")
is_cuda_available = torch.cuda.is_available()
print(f"CUDA available: {is_cuda_available}")
if is_cuda_available:
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU name: {torch.cuda.get_device_name(0)}")

# Kiểm tra phiên bản PennyLane
print(f"PennyLane version: {qml.__version__}")

# Thoát khỏi Python
exit()
