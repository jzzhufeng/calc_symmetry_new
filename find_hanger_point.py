import cv2
import numpy as np


hanger_color = 'green'
point_color = 'gray'
color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
              'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
              #'gray': {'Lower': np.array([0, 0, 46]), 'Upper': np.array([180, 43, 220])},
              'gray': {'Lower': np.array([0, 0, 40]), 'Upper': np.array([180, 50, 230])},
              'purple': {'Lower': np.array([125, 43, 46]), 'Upper': np.array([155, 255, 255])}
              }

def hsvHandle(img, color):
    gs_frame = cv2.GaussianBlur(img.copy(), (5, 5), 0)# 高斯模糊
    hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)# 转化成HSV图像
    erode_hsv = cv2.erode(hsv, None, iterations=2) # 腐蚀 粗的变细
    inRange_hsv = cv2.inRange(erode_hsv, color_dist[color]['Lower'], color_dist[color]['Upper'])
    cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)#返回最小外接矩形的中心点坐标，宽高和倾角
    return rect

def check_hanger(imgSrc, testDir):
    img = imgSrc.copy()
    rect = hsvHandle(img, hanger_color)
    box = cv2.boxPoints(rect)

    cv2.drawContours(img, [np.int0(box)], -1, (0, 255, 255), 2)
    cv2.imwrite(testDir + 'find_hanger_box.jpg', img)

    box_sort_y = np.sort(box, axis = 0)
    box_sort_x = np.sort(box, axis = 1)
    
    min_y = box_sort_y[0][1].astype(np.uint32)
    max_y = box_sort_y[2][1].astype(np.uint32)
    min_x = box_sort_x[0][1].astype(np.uint32)
    max_x = box_sort_x[2][1].astype(np.uint32)
    
    hanger = imgSrc[min_y: max_y, min_x: max_x]
    hanger = hanger[0: round(2 * hanger.shape[0] / 3), :]
    cv2.imwrite(testDir + 'hanger_temp.jpg', hanger)
   
    rect = hsvHandle(hanger, hanger_color)
    box = cv2.boxPoints(rect)
    certer_x = round(rect[0][0]) - 10
    certer_y = round(rect[0][1])
    cv2.imwrite(testDir + 'hanger.jpg', hanger)

    hanger = hanger[:, certer_x - 35: certer_x + 55]
    cv2.imwrite(testDir + 'hanger_center.jpg', hanger)
    hsv = cv2.cvtColor(hanger, cv2.COLOR_BGR2HSV)# 转化成HSV图像
    hsv = hsv[certer_y:certer_y + 1 , :, :]
    cv2.imwrite(testDir + 'hsv_t.jpg', hsv)
    
    start_x = -1
    end_x = -1
    find_start_over = False
    for i in range(hsv.shape[1]):
        if i < 35:
            continue
        if not find_start_over:
            if (hsv[0][i][0] >= 35 and  hsv[0][i][1] >= 43 and  hsv[0][i][2] >= 35 and hsv[0][i][0] <= 90):
                start_x = i
                find_start_over = True
        else:
            if not (hsv[0][i][0] >= 35 and  hsv[0][i][1] >= 43 and  hsv[0][i][2] >= 35 and hsv[0][i][0] <= 90):
                end_x = i
    
    hanger = hanger[:, start_x: end_x]
    for i in range(hanger.shape[1]):
        if i < 10 or i > hanger.shape[1] - 10:
            hanger[:, i, :] = 0
 
            
    point_up = hanger[0:80, :]
    point_down = hanger[80:hanger.shape[0], :]
    
    rect_up = hsvHandle(point_up, point_color)
    point_up_axix_x = round(rect_up[0][0])
    point_up_axix_y = round(rect_up[0][1])

    rect_down = hsvHandle(point_down, point_color)
    point_down_axix_x = round(rect_down[0][0])
    point_down_axix_y = round(rect_down[0][1])

    x1 = min_x + certer_x - 35 + start_x + point_up_axix_x 
    y1 = min_y + point_up_axix_y
    x2 = min_x + certer_x - 35 + start_x + point_down_axix_x 
    y2 = min_y + 80 + point_down_axix_y
    cv2.circle(img, (x1, y1), 2, (0, 0, 255), -1)
    cv2.circle(img, (x2, y2), 2, (0, 0, 255), -1)
    cv2.imwrite(testDir + 'img_with_point.jpg', img)
    return x1, y1, x2, y2



