import cv2
import sys

#cascPath = sys.argv[0]
bodyCascade = cv2.CascadeClassifier('haarcascade_mcs_upperbody.xml')


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    body = bodyCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100,200),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )


    # Draw a rectangle around the faces
    for (x, y, w, h) in body:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)    
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
