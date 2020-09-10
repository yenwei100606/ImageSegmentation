import cv2
import numpy as np

# image = cv2.imread(r'C:\Users\Yen_Wei\vegetable2.jpg')
image = cv2.imread(r'C:\Users\Yen_Wei\vegetable2.jpg')
img = cv2.resize(image,(512,512),interpolation=cv2.INTER_CUBIC)

#img.shape[:2]=img.shape[0:2] 代表取彩色影象的長和寬  img.shape[:3] 代表取長+寬+通道
mask = np.zeros(img.shape[:2], np.uint8)

# 背景模型，如果為None，函式內部會自動建立一個bgdModel；bgdModel必須是單通道浮點型影象，且行數只能為1，列數只能為13x5；
bgdModel = np.zeros((1, 65), np.float64)

# fgdModel——前景模型，如果為None，函式內部會自動建立一個fgdModel；fgdModel必須是單通道浮點型影象，且行數只能為1，列數只能為13x5；
fgdModel = np.zeros((1, 65), np.float64)

cv2.grabCut(img,mask,(1,1,511,511), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0),0,1).astype('uint8')
img_show = img * mask2[:,:,np.newaxis]

#threshold segmentation
triThe = 160
maxval = 255
src = cv2.cvtColor(img_show,cv2.COLOR_BGR2GRAY)
triThe, dst_tri = cv2.threshold(src, triThe, maxval, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
# triThe, dst_tri = cv2.threshold(src, triThe, maxval, cv2.THRESH_TRIANGLE + cv2.THRESH_BINARY)
# triThe, dst_tri = cv2.threshold(src, triThe, maxval, cv2.THRESH_BINARY)
print ("triThe:",triThe)

# 利用threshold分割出輪廓
dst = cv2.adaptiveThreshold(dst_tri,210,cv2.BORDER_REPLICATE,cv2.THRESH_BINARY_INV,3,10)
contours,hierarchy = cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(dst,contours,-1,(255,0,255),1)

cv2.imshow("image",img)
cv2.imshow("grabCut",img_show)
cv2.imshow('thresh_out', dst_tri)
cv2.imshow("dst",dst)

# # cv2.imshow("test",img_test)
# area = 0
# for i in range(len(contours)):
#     print("area",i,"=",contours[i],"=",cv2.contourArea(contours[i]))
#     area += cv2.contourArea(contours[i])
# print("area=",area)
ares_avrg=0
count = 0
for cont in contours:

    ares = cv2.contourArea(cont)#計算包圍性狀的面積

    
    ares_avrg+=ares

    print("{}-blob:{}".format(count,ares),end="  ") #打印出每個米粒的面積

    rect = cv2.boundingRect(cont) #提取矩形座標

    print("x:{} y:{}".format(rect[0],rect[1]))#打印座標

    cv2.rectangle(img,rect,(0,0,0xff),1)#繪製矩形

    y=10 if rect[1]<10 else rect[1] #防止編號到圖片之外

    cv2.putText(img,str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1) #在米粒左上角寫上編


cv2.waitKey()
cv2.destroyAllWindows()
