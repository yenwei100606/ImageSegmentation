import cv2
import numpy as np

src = cv2.imread(r"C:\Users\Yen_Wei\vegetable2.jpg")
image = cv2.resize(src,(512,512),interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
canny = cv2.Canny(blurred, 30, 150)
result = np.hstack([gray, blurred, canny])
cv2.imshow("Result:", result)
cv2.waitKey(0)