from __future__ import print_function
import torch
import numpy as np
import torch.cuda


# 将Torch张量转化为Numpy数组
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

# 查看numpy数组的值如何变化。
a.add_(1)
print(a)
print(b)

# 将NumPy数组转换为Torch张量
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)


#CUDA张量，使用该.to方法可以将张量移动到任何设备上。
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
print(1111)