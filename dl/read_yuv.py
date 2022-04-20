#!/usr/bin/env python3

import cv2
import numpy as np
import sys

h, w = 576, 704
if len(sys.argv) == 4:
    w = int(sys.argv[2])
    h = int(sys.argv[3])
print(f'{w}x{h}')

with open(sys.argv[1], 'rb') as f:
    content = f.read()
assert len(content) == int(w*h*3/2)

yuv = np.frombuffer(content, dtype=np.uint8)
yuv = yuv.reshape((int(h*1.5), w))

bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)

cv2.imshow('frame', bgr)
cv2.waitKey(0)
