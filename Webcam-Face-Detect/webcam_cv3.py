import cv2
import sys
import datetime as dt
from time import sleep

faceCascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")
video_capture = cv2.VideoCapture('2017-08-07 18-21-34_1.avi', 0)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        break

    ret, frame = video_capture.read()
    if ret is not True:
        break
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 在前面已经转换了

    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.08,  # 越小越慢，越可能检测到
        minNeighbors=2,  # 越小越慢，越可能检测到
        minSize=(80, 80),
        maxSize=(149, 180),
        # minSize=(30, 30)
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    print(faces)
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # if face x属于一个范围才输入到数据中，这样提高准确率
        # print('x1:', x, y, 'x2:', x+w, y+h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        # print("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
