import tensorflow as tf

"""
TensorFlow 使用 张量 （Tensor）作为数据的基本单位。
TensorFlow 的张量在概念上等同于多维数组，
我们可以使用它来描述数学中的标量（0 维数组）、向量（1 维数组）、
矩阵（2 维数组）等各种量，示例如下：
"""
# 定义一个随机数（标量）
random_float = tf.random.uniform(shape=())

# 定义一个有2个元素的零向量
zero_vector = tf.zeros(shape=(2))

# 定义两个2×2的常量矩阵
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])

# 张量的重要属性是其形状、类型和值。可以通过张量的 shape 、 dtype 属性和 numpy() 方法获得。例如：
# 查看矩阵A的形状、类型和值
print(A.shape)  # 输出(2, 2)，即矩阵的长和宽均为2
print(A.dtype)  # 输出<dtype: 'float32'>
print(A.numpy())  # 输出[[1. 2.]
#      [3. 4.]]

# 对A和B进行一些操作
C = tf.add(A, B)  # 计算矩阵A和B的和
D = tf.matmul(A, B)  # 计算矩阵A和B的乘积
print(C.numpy())
print(D.numpy())
