import numpy as np
import tensorflow as tf

# 首先，我们定义数据，进行基本的归一化操作。
X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

"""
现在，我们希望通过对该数据进行线性回归，即使用线性模型 y = ax + b 来拟合上述数据，此处 a 和 b 是待求的参数。
接下来，我们使用梯度下降方法来求线性模型中两个参数 a 和 b 的值 3。

对于多元函数 f(x) 求局部极小值，梯度下降 的过程如下：

    初始化自变量为 x_0 ， k=0

    迭代进行下列步骤直到满足收敛条件：

        求函数 f(x) 关于自变量的梯度 ▽f(x_k)

        更新自变量： x_{k+1} = x_{k} - γ*▽f(x_k) 。这里 γ是学习率（也就是梯度下降一次迈出的 “步子” 大小）

        k ＜—— k+1

接下来，我们考虑如何使用程序来实现梯度下降方法，
求得线性回归的解 min_{a, b} * L(a, b) = sum_{i=1}^n(ax_i + b - y_i)^2 。
"""
# NumPy 下的线性回归
# np.dot() 是求内积， np.sum() 是求和

a, b = 0, 0
num_epoch = 10000
learning_rate = 1e-3
for e in range(num_epoch):
    # 手动计算损失函数关于自变量（模型参数）的梯度
    y_pred = a * X + b
    grad_a, grad_b = (y_pred - y).dot(X), (y_pred - y).sum()

    # 更新参数
    a, b = a - learning_rate * grad_a, b - learning_rate * grad_b

print(a, b)

"""
使用常规的科学计算库实现机器学习模型有两个痛点：
    1.经常需要手工求函数关于参数的偏导数。如果是简单的函数或许还好，
    但一旦函数的形式变得复杂（尤其是深度学习模型），
    手工求导的过程将变得非常痛苦，甚至不可行。
    2.经常需要手工根据求导的结果更新参数。这里使用了最基础的梯度下降方法，
    因此参数的更新还较为容易。但如果使用更加复杂的参数更新方法（例如 Adam 或者 Adagrad），
    这个更新过程的编写同样会非常繁杂。
"""

# TensorFlow 下的线性回归
"""
TensorFlow 的 即时执行模式 5 与上述 NumPy 的运行方式十分类似，然而提供了更快速的
运算（GPU 支持）、自动求导、优化器等一系列对深度学习非常重要的功能。
以下展示了如何使用 TensorFlow 计算线性回归。
可以注意到，程序的结构和前述 NumPy 的实现非常类似。
这里，TensorFlow 帮助我们做了两件重要的工作：
    使用 tape.gradient(ys, xs) 自动计算梯度；
    使用 optimizer.apply_gradients(grads_and_vars) 自动更新模型参数。
"""
X = tf.constant(X)
y = tf.constant(y)

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

print(a, b)
"""
在这里，我们使用了前文的方式计算了损失函数关于参数的偏导数。
同时，使用 tf.keras.optimizers.SGD(learning_rate=1e-3) 声明了一个梯度下降 优化器 （Optimizer），
其学习率为 1e-3。优化器可以帮助我们根据计算出的求导结果更新模型参数，从而最小化某个特定的损失函数，
具体使用方式是调用其 apply_gradients() 方法。

注意到这里，更新模型参数的方法 optimizer.apply_gradients() 需要提供参数 grads_and_vars，
即待更新的变量（如上述代码中的 variables ）及损失函数关于这些变量的偏导数（如上述代码中的 grads ）。
具体而言，这里需要传入一个 Python 列表（List），列表中的每个元素是一个 （变量的偏导数，变量） 对。
比如上例中需要传入的参数是 [(grad_a, a), (grad_b, b)] 。我们通过 grads = tape.gradient(loss, variables) 
求出 tape 中记录的 loss 关于 variables = [a, b] 中每个变量的偏导数，也就是 grads = [grad_a, grad_b]，
再使用 Python 的 zip() 函数将 grads = [grad_a, grad_b] 和 variables = [a, b] 拼装在一起，就可以组合出所需的参数了。
"""
