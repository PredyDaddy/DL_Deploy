import torch

# 检查CUDA是否可用
print('CUDA available: ', torch.cuda.is_available())

# 输出CUDA设备数量
print('Number of CUDA devices: ', torch.cuda.device_count())

# 输出当前设备的名称和能力
print('Current device: ', torch.cuda.get_device_name(0))
print('Capability of current device: ', torch.cuda.get_device_capability(0))
