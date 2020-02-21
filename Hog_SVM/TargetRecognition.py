"""
Hog+SVM实现小狮子的识别
步骤
    1、准备样本
    2、训练
    3、test 预测
pos文件夹下为正样本 包含所检测的目标
neg文件夹下为负样本 不包含所检测的目标

步骤
    1 参数 2 hog 3 svm 4computer hog 5 label 6 train 7 pred 8 draw
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

PosNum = 5548  # 正样本的个数
NegNUm = 2146
winSize = (176, 144)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nBin = 9

# 2 创建一个hog并传入参数
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBin)

# 3 svm
svm = cv2.ml.SVM_create()

# 4 计算hog
featureNum = int(((176 - 16) / 8 + 1) * ((144 - 16) / 8 + 1) * 4 * 9)  # 12582
featureArray = np.zeros(((PosNum + NegNUm), featureNum), np.float32)
labelArray = np.zeros(((PosNum + NegNUm), 1), np.int32)

for i in range(0, PosNum):
    fileName = r'E:/testimage/body/Pos/image' + str(i + 1) + '.jpg'
    print(fileName)
    img = cv2.imread(fileName)
    hist = hog.compute(img,(8,8))
    for j in range(0, featureNum):  # hog特征的装载
        featureArray[i, j] = hist[j]
    labelArray[i, 0] = 1  # 正样本

for i in range(0, NegNUm):
    fileName = 'E:/testimage/body/Neg/image' + str(i + 1) + '.jpg'
    print(i)
    img = cv2.imread(fileName)
    hist = hog.compute(img, (8, 8))
    for j in range(0, featureNum):  # hog特征的装载
        featureArray[i+PosNum, j] = hist[j]
    labelArray[i + PosNum, 0] = -1

svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(0.01)

# 6 train
ret = svm.train(featureArray, cv2.ml.ROW_SAMPLE, labelArray)
# 7 检测
alpha = np.zeros((1), np.float32)
rho = svm.getDecisionFunction(0, alpha)
print(rho)
print(alpha)
alphaArray = np.zeros((1,1), np.float32)
supportVArray = np.zeros((1, featureNum), np.float32)  # 支持向量机数组
resultArray = np.zeros((1,featureNum), np.float32)
alphaArray[0,0] = alpha
resultArray = -1*alphaArray*supportVArray

# detect
myDetect = np.zeros((12583), np.float32)
for i in range(0, 12582):
    myDetect[i] = resultArray[0, i]
myDetect[12582] = rho[0]

# 构建hog
myHog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBin)
myHog.setSVMDetector(myDetect)



fileName3 = r'C:\Users\CYM\Desktop\image20.jpg'
imageSrc = cv2.imread(fileName3, 1)
objs = myHog.detectMultiScale(imageSrc, 0, (8, 8), (8, 8), 1.05, 2)
x = int(objs[0][0][0])
y = int(objs[0][0][1])
w = int(objs[0][0][2])
h = int(objs[0][0][3])
cv2.rectangle(imageSrc, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imshow('dst', imageSrc)

cv2.waitKey(0)
cv2.destroyAllWindows()
