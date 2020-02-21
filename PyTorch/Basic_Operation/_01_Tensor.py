from __future__ import print_function
import torch

# Tensors与NumPy的ndarrays类似，此外，Tensors还可以在GPU上使用以加速计算。

# 构造一个未初始化的5x3矩阵
x = torch.empty(5, 3)
print(x)

# 构造一个随机初始化的矩阵
x = torch.rand(5, 3)
print(x)

# 构造一个填充零且dtype long的矩阵
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 直接从数据构造张量
x = torch.tensor([5.5, 3])
print(x)

# 或基于现有张量创建张量。这些方法将重用输入张量的属性，例如dtype，除非用户提供新值
x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)  # override dtype!
print(x)  # result has the same size
print(x.size())

# 加法
y = torch.rand(5, 3)
print(y)
print(x + y)
print(torch.add(x, y))
# 加法：提供输出张量作为参数
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
# 加法：就地
y.add_(x)
print(y)


#调整大小：如果要调整张量的大小/形状，可以使用torch.view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())


#使用.item()将该值作为Python数字获取
x = torch.randn(1)
print(x)
print(x.item())