#!/usr/bin/env python3
# coding=utf-8
import numpy as np
import cv2
import time
from PIL import Image


def entropy(band):
    hist, _ = np.histogram(band, bins=range(0, 256))
    hist = hist[hist > 0]
    return -np.log2(hist / hist.sum()).sum()


def show_entropy(band_name, band):
    bits = entropy(band)
    per_pixel = bits / band.size
    print("{:3s} entropy = {:.2f} bits, {:.6f} per pixel"
          .format(band_name, bits, per_pixel))


def process(img_file):
    im = Image.open(img_file)
    print(img_file, im.format, im.size, im.mode)
    print()

    rgb = im.convert("RGB")
    hsv = im.convert("HSV")
    grey = im.convert("L")
    r, g, b = [np.asarray(component) for component in rgb.split()]
    h, s, v = [np.asarray(component) for component in hsv.split()]
    l = np.asarray(grey)

    show_entropy("R", r)
    show_entropy("G", g)
    show_entropy("B", b)
    print()

    show_entropy("H", h)
    show_entropy("S", s)
    show_entropy("V", v)
    print()

    show_entropy("L", l)
    print()


def process_rgb_average(img_file):
    im = Image.open(img_file)
    print("Processing image:",img_file)
    rgb = im.convert("RGB")
    r, g, b = [np.asarray(component) for component in rgb.split()]
    RGB_average = (entropy(r)+entropy(g)+entropy(b))/3
    print("Entropy:", RGB_average)


def process_rgb_delta(img_file, entropy_last_inner):
    im = Image.open(img_file)
    # print("Processing image:",img_file)
    rgb = im.convert("RGB")
    r, g, b = [np.asarray(component) for component in rgb.split()]
    rgb_average = (entropy(r)+entropy(g)+entropy(b))/3
    jitter = abs(rgb_average - entropy_last_inner)
    print("Entropy Jitter:", jitter)
    return rgb_average

if __name__ == "__main__":

    start = time.time()
    # 读入视频文件
    vc = cv2.VideoCapture('video/final.mp4')
    # 判断是否正常打开
    if vc.isOpened():
        totalFrameNumber = vc.get(cv2.CAP_PROP_FRAME_COUNT)
        print("Total frame number:", totalFrameNumber)
    else:
        rval = False
        print(rval)
    # 开始结束帧
    frametostart = 0
    frametostop = totalFrameNumber - 1
    vc.set(cv2.CAP_PROP_POS_FRAMES, frametostart)
    frame_now = frametostart
    unstop = 1
    entropy_last = 0
    while unstop:
        rval, frame = vc.read()
        print(type(rval), type(frame))
        cv2.imwrite('image/' + str(frame_now) + '.jpg', frame)
        # print("Saved:", 'image/'+str(frame_now) + '.jpg')
        entropy_last = process_rgb_delta('image/'+str(frame_now) + '.jpg', entropy_last)
        frame_now = frame_now + 1
        if frame_now > frametostop:
            unstop = 0

    end = time.time()
    print(end - start)
    vc.release()
    cv2.destroyAllWindows()
