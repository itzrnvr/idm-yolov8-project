"""Quick GPU probe — verifies colab run can allocate a T4 and CUDA is live."""
import torch
print("CUDA:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
