import cv2
face_xml = cv2.CascadeClassifier('E:/testimage/svm.xml')
#face_xml.load('E:/testimage/svm.xml')
hog =cv2.HOGDescriptor(face_xml)
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
videoFileName = r'E:\FDownload\lanqiu.flv'
cap = cv2.VideoCapture(videoFileName)
while True:
    cv2.HOGDescriptor()
    ret, img = cap.read()
    if not ret:  # 没读到当前帧，结束
        break
    rects, _ = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('Detect', img)
    key = cv2.waitKey(30)  # 每一帧间隔30ms
    if key == 27:  # 按下ESC键，退出
        flag =1
        break