import torch

# Load the checkpoint file
ckpt_path = "data/ckpt/usleep/alternative_big_sleep.ckpt"
checkpoint = torch.load(
    ckpt_path, map_location=torch.device("cpu")
)  # Load on CPU to avoid GPU issues

# Print the keys in the checkpoint
print("Checkpoint Keys:", checkpoint.keys())

# If it contains a model state dictionary, print available layers
if "state_dict" in checkpoint:
    print("\nModel State Dict Keys:")
    for key in checkpoint["state_dict"]:
        print(key)
