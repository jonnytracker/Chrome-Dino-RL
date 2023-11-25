import torch

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")
