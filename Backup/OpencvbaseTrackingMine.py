import cv2
import sys

if __name__ == '__main__':


    # Set up tracker.
    # Instead of MIL, you can also use
    # BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN

    tracker = cv2.Tracker_create("KCF")

    # Read video
    video = cv2.VideoCapture(0)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    '''
    # Read first frame.
    ok, frame = video.read()
    # ok = 1
    # frame = cv2.imread('test.png')
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    # bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    '''
    flag = 0
    point_x, point_y = 0, 0
    while True:
        if flag == 0:
            ok, frame = video.read()
            if not ok:
                break
            cv2.imshow("Tracking", frame)

            if cv2.waitKey(20) & 0xFF == ord("s"):
                bbox = cv2.selectROI(frame, False)
                ok = tracker.init(frame, bbox)
                flag = 1
            continue
        if flag == 1:
            # Read a new frame
            ok, frame = video.read()
            if not ok:
                break

            # Update tracker
            ok, bbox = tracker.update(frame)

            # Draw bounding box
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 0, 255))  # tuple
                if point_x == 0 and point_y == 0:
                    pass
                else:
                    v_updown = point_y - p1[1]
                    v_leftright = p1[0] - point_x
                    print(v_leftright, v_updown)
                point_x = p1[0]
                point_y = p1[1]



            # Display result
            cv2.imshow("Tracking", frame)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
