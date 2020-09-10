import cv2
import numpy as np

'''
       created on  08:10:27 2018-11-15
       @author:ren_dong

                Grabcut 影象分割  
                
                加入滑鼠回撥函式,可以進行互動式操作,滑鼠選擇矩形區域作為ROI
                
               cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount[, mode]) → None
               on_mouse()


'''


# 滑鼠事件的回撥函式
def on_mouse(event, x, y, flag, param):
    global rect
    global leftButtonDown
    global leftButtonUp

    # 滑鼠左鍵按下
    if event == cv2.EVENT_LBUTTONDOWN:
        rect[0] = x
        rect[2] = x
        rect[1] = y
        rect[3] = y
        leftButtonDown = True
        leftButtonUp = False

    # 移動滑鼠事件
    if event == cv2.EVENT_MOUSEMOVE:
        if leftButtonDown and not leftButtonUp:
            rect[2] = x
            rect[3] = y

    # 滑鼠左鍵鬆開
    if event == cv2.EVENT_LBUTTONUP:
        if leftButtonDown and not leftButtonUp:
            x_min = min(rect[0], rect[2])
            y_min = min(rect[1], rect[3])

            x_max = max(rect[0], rect[2])
            y_max = max(rect[1], rect[3])

            rect[0] = x_min
            rect[1] = y_min
            rect[2] = x_max
            rect[3] = y_max
            leftButtonDown = False
            leftButtonUp = True


# 讀入圖片
img = cv2.imread(r'C:\Users\YenWei\picture\vegetable3.jpg')
img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
'''
    
     掩碼影象，如果使用掩碼進行初始化，那麼mask儲存初始化掩碼資訊；
     在執行分割的時候，也可以將使用者互動所設定的前景與背景儲存到mask中，
     然後再傳入grabCut函式；
     在處理結束之後，mask中會儲存結果

'''
#img.shape[:2]=img.shape[0:2] 代表取彩色影象的長和寬  img.shape[:3] 代表取長+寬+通道
mask = np.zeros(img.shape[:2], np.uint8)

# 背景模型，如果為None，函式內部會自動建立一個bgdModel；bgdModel必須是單通道浮點型影象，且行數只能為1，列數只能為13x5；
bgdModel = np.zeros((1, 65), np.float64)
# fgdModel——前景模型，如果為None，函式內部會自動建立一個fgdModel；fgdModel必須是單通道浮點型影象，且行數只能為1，列數只能為13x5；
fgdModel = np.zeros((1, 65), np.float64)

# 用於限定需要進行分割的影象範圍，只有該矩形視窗內的影象部分才被處理；
# rect 初始化
rect = [0, 0, 0, 0]

# 滑鼠左鍵按下
leftButtonDown = False
# 滑鼠左鍵鬆開
leftButtonUp = True

# 指定視窗名來建立視窗
cv2.namedWindow('img')
# 設定滑鼠事件回撥函式 來獲取滑鼠輸入
cv2.setMouseCallback('img', on_mouse)

# 顯示圖片
cv2.imshow('img', img)


##設定迴圈,進行互動式操作
while cv2.waitKey(2) == -1:
    # 左鍵按下，畫矩陣
    if leftButtonDown and not leftButtonUp:

        img_copy = img.copy()
        # 在img影象上，繪製矩形  線條顏色為green 線寬為2
        cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        # 顯示圖片
        cv2.imshow('img', img_copy)

    # 左鍵鬆開，矩形畫好
    elif not leftButtonDown and leftButtonUp and rect[2] - rect[0] != 0 and rect[3] - rect[1] != 0:
        # 轉換為寬度高度
        rect[2] = rect[2] - rect[0]
        rect[3] = rect[3] - rect[1]
        # rect_copy = tuple(rect.copy())
        rect_copy = tuple(rect)
        rect = [0, 0, 0, 0]
        # 物體分割
        cv2.grabCut(img, mask, rect_copy, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        #cv2.grabCut(img, mask, (1,1,511,511), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img_show = img * mask2[:, :, np.newaxis]
        # 顯示圖片分割後結果
        cv2.imshow('grabcut', img_show)
        # 顯示原圖
        cv2.imshow('img', img)
 
        

cv2.waitKey()
cv2.destroyAllWindows()