"""
基于Haar+Adaboost人脸识别
"""
# 加载两个XML文件
# 加载两个图片
# 灰度处理图片
# 检测
# 对检测出来的特征框出来
import cv2
import numpy as np

# 加载两个XML文件
face_xml = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
face_xml.load('E:/testimage/haarcascade_frontalface_alt_tree.xml')
eye_xml = cv2.CascadeClassifier('E:/testimage/haarcascade_frontalface_alt_tree.xml')

# 加载图片并灰度处理
filename = r'E:\testimage\tree.jpg'
img = cv2.imread(filename)
cv2.imshow('src',img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测
faces = face_xml.detectMultiScale(gray, 1.3, 5)  # 1.3为缩放系数  目标大小最低不能小于5个像素
print('face=', len(faces))  # 有多少个人脸

# 画方框
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_face = gray[y:y + h, x:x + w]  # 找到的灰度人脸的范围
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_xml.detectMultiScale(roi_face)
    print('eye = ', len(eyes))
    for (e_x, e_y, e_w, e_h) in eyes:
        cv2.rectangle(roi_color, (e_x, e_y), (e_x + e_w, e_y + e_h), (0, 255, 0), 2)
cv2.imshow('dst',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
