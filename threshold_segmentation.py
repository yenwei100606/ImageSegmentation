import numpy as np
import matplotlib.pyplot as plt
import cv2

# Step1. 加載圖像
image = cv2.imread(r'C:\Users\Yen_Wei\vegetable4.jpg',cv2.IMREAD_GRAYSCALE)
src = cv2.resize(image,(512,512),interpolation=cv2.INTER_CUBIC)

triThe = 127
maxval = 255
triThe, dst_tri = cv2.threshold(src, triThe, maxval, cv2.THRESH_TRIANGLE + cv2.THRESH_BINARY)
triThe1, dst_tri1 = cv2.threshold(src, triThe, maxval, cv2.THRESH_TRIANGLE + cv2.THRESH_BINARY_INV)
print ("triThe:",triThe)
print ("triThe1",triThe1)
cv2.imshow("image", src)
cv2.imshow('thresh_out', dst_tri)
cv2.imshow('thresh_out1', dst_tri1)
cv2.waitKey(0)
cv2.destroyAllWindows()