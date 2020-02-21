"""
使用SVM完成男生女生身高体重的分类问题，并完成训练，且可预测
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 准备数据
rand1 = np.array([[155, 48], [159, 50], [164, 53], [168, 56], [172, 60]])  # 女生数据
rand2 = np.array([[152, 53], [156, 55], [160, 56], [172, 64], [176, 65]])  # 男生数据

# 准备标签
label = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]])

# 数据处理
data = np.vstack((rand1, rand2))
data = np.array(data, dtype='float32')

# svm 所有的数据都要有label
# 0为女生，1位男生
# 监督学习 0为负样本 1为正样本

svm = cv2.ml.SVM_create()  # 创建了一个支持向量机（ml 机器学习模块）

# 属性设置
svm.setType(cv2.ml.SVM_C_SVC)  # 设置svm类型
svm.setKernel(cv2.ml.SVM_LINEAR)  # 设置svm内核为线性
svm.setC(0.01)  # 目前不了解这个表示什么

# 开始训练
svm.train(data, cv2.ml.ROW_SAMPLE, label)

# 预测
pt_data = np.vstack([[167, 55], [162, 57]])
pt_data = np.array(pt_data, dtype='float32')
print(pt_data)
(par1, par2) = svm.predict(pt_data)
print(par1, par2)
