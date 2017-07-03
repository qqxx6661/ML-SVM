import cv2
import numpy as np
from collections import deque
import time

camera = cv2.VideoCapture("newtest2.mp4")
firstframe = None
buffer = 50
pts = deque([0]*buffer, maxlen=buffer)

counter = 0
(dX, dY) = (0, 0)
direction = ""


while True:
    ret, frame = camera.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    if firstframe is None:
        firstframe = gray
        continue

    frameDelta = cv2.absdiff(firstframe, gray)
    thresh = cv2.threshold(frameDelta, 0, 255, cv2.THRESH_OTSU)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contour = cv2.Canny(gray,100, 200)
    (_,cnts,_) = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        if cv2.contourArea(c) < 5000:
            continue

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        center = tuple(np.int0(0.25*sum(box)))
        pts.appendleft(center)

        img = cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

    for i in np.arange(1, len(pts)):

        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] == 0 or pts[i] == 0:
            continue

        # check to see if enough points have been accumulated in
        # the buffer
        if counter >= buffer and i == 1 and pts[-buffer] is not None:
            # compute the difference between the x and y
            # coordinates and re-initialize the direction
            # text variables
            dX = pts[-buffer][0] - pts[i][0]
            dY = pts[-buffer][1] - pts[i][1]
            (dirX, dirY) = ("", "")

            # ensure there is significant movement in the
            # x-direction
            if np.abs(dX) > 3:
                dirX = "right" if np.sign(dX) == 1 else "Left"

            # ensure there is significant movement in the
            # y-directionq
            if np.abs(dY) > 3:
                dirY = "Down" if np.sign(dY) == 1 else "Up"

            # handle when both directions are non-empty
            if dirX != "" and dirY != "":
                direction = "{}-{}".format(dirY, dirX)

            # otherwise, only one direction is non-empty
            else:
                direction = dirX if dirX != "" else dirY
        thickness = int(np.sqrt(20 / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 255), 3)
    cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)

    cv2.imshow('contour', contour)
    cv2.imshow("frame", frame)
    cv2.imshow("frame2", thresh)
    key = cv2.waitKey(1) & 0xFF

    counter += 1
    time.sleep(0.03)

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()