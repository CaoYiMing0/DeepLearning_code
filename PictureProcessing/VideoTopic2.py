# 视频分解为图片
import numpy as np
import cv2
file = r'E:\testimage\body\Scenery.flv'
cap = cv2.VideoCapture(file)
isOpend = cap.isOpened # 判断是否打开
print(isOpend)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(fps,width,height)
i = 0
while(isOpend):
    i = i + 1
    (flag,frame) = cap.read()  #读取每一张图片 flag：是否读取成功  frame：图片的内容
    frameInfo = frame.shape
    height2 = frameInfo[0]
    width2 = frameInfo[1]
    matScale = np.float32([[0.5,0,0],[0,0.5,0]])
    dst = cv2.warpAffine(frame,matScale,(int(width2/2),int(height2/2)))
    dst2 = dst[0:144,0:176]
    fileName = 'image' +str(i)+'.jpg'
    print(fileName)
    if flag == True:
        cv2.imwrite(fileName,dst2,[cv2.IMWRITE_JPEG_QUALITY,100])
    else:
        break
print('end!')
