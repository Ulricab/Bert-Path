import torch
gpus = torch.cuda.device_count()
print(gpus)
print(torch.cuda.is_available())
print(torch.cuda.current_device())