"""
PyTorch的神经网络的核心就是autograd包，而autograd提供了所有张量函数的自动求导。而且autograd是一个“define-by-run”的架构，
它的后向传播是由代码的运行来决定的。还有一个概念是"define-and-run"，前者和后者的区别应该类似动态图和静态图的区别。

torch.Tensor是autograd包的核心类。如果你将属性.requires_grad设为True，它将会追踪所有对张量的操作。
当你完成了计算(注：应该指前向传播)后，你可以调用.backward()来自动计算所有的梯度。计算结果会累计到.grad属性内。

若想停止追踪一个张量，你可以调用.detach()来将其从计算历史中移除，且其未来的计算也不会被追踪
(注：应该就是说计算梯度时跳过这个张量)。

为了防止追踪历史(以及使用内存)，你还可以将代码包装在with torch.no_grad():中。这在评估模型时可能会尤为有用，
因为模型可能有requires_grad = True的不需要梯度的可训练变量。

还有一个对autograd的实现很重要的类：Function。

Tensor和Function相互链接，构成了一个非循环图，这个图编码了整个计算过程。每个张量有一个.grad_fn属性，
它指向了创建这个张量的Function(除非这个张量是由用户创建的，它的grad_fn值为None)。

如果你想导数，你可以使用Tensor类调用.backward()。如果Tensor是个标量(也就是说，这个张量只有一个元素)，
你不必为backward()指定任何参数。但若它有更多元素的话，你就要指定一个形状匹配的张量作为梯度参数。
"""
import torch

# 创建一个向量，并将它的requires_grad属性设为True来追踪它的计算历史
x = torch.ones(2, 2, requires_grad=True)
print(x)

# 进行张量计算
y = x + 2
print(y)

# y是由于操作而创建的，因此具有grad_fn。
print(y.grad_fn)

# 进行更多操作 y
z = y * y * 3
out = z.mean()
print(z, out)

# .requires_grad_(…)可以改变张量的requires_grad值。如果未指定，该值则默认为False
