"""Sanity check: PyTorch CUDA, Warp and MuJoCo Warp on this machine."""
import torch, warp as wp, mujoco, mujoco_warp, mjlab
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", torch.cuda.get_device_capability(0))
print("warp", wp.__version__, "mujoco", mujoco.__version__)
wp.init()
print("warp devices", wp.get_cuda_devices())
x = torch.randn(1000, 1000, device="cuda"); print("matmul ok", float((x @ x).sum()) != 0)
