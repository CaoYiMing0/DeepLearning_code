# 视频分解为图片

import cv2
file = r'E:\testimage\body\shahar_wave2.avi'
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
    print(frame.size)
    dst = frame[0:144,0:176]
    fileName = 'image' +str(i)+'.jpg'
    print(fileName)
    if flag == True:
        cv2.imwrite(fileName,dst,[cv2.IMWRITE_JPEG_QUALITY,100])
    else:
        break
print('end!')
