import cv2
import numpy as np
import sys
import math
from find_hanger_point import check_hanger
import os
import argparse

BLACK = [0 ,0 ,0]

def line(h, w, h1, w1, h2, w2):
    k = (w2-w1)/(h2-h1)  
    b = w1 - k*h1        
    return (h * k + b) <= w  
    
def cut(img, point_A, point_B):
    h1, w1 = point_A[0], point_A[1]
    h2, w2 = point_B[0], point_B[1]
    height, width, _ = img.shape
    
    up = np.zeros((height, width, 3), dtype='uint8')
    down = np.zeros((height, width,3), dtype='uint8')  
    
    for i in range(height):
        for j in range(width):
            if line(i,j, h1, w1, h2, w2):
                up[i,j] = img[i,j]
            else:
                down[i,j] = img[i,j]
    return np.array(up), np.array(down)


def startCalc(path, testDir):
    img = cv2.imread(path)
    img = cv2.copyMakeBorder(img, 50, 50, 0, 0, cv2.BORDER_CONSTANT, value=BLACK) #用黑色拓展上下边界，防止翻转的时候丢失像素
    totalHeight, totalwidth, _ = img.shape
    
    imgNew = img.copy()
    outputDir = testDir + '1/'
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    x1, y1, x2, y2 = check_hanger(img, outputDir)
    print(f'${x1} ${y1} ${x2} ${y2}')

    k = (x2 - x1) / (y2 - y1)
    h = math.degrees(math.atan(k))
    print(h)
    roteate_matrix = cv2.getRotationMatrix2D(center = (totalwidth/2, totalHeight/2), angle=-h,scale=1)
    rotated_image = cv2.warpAffine(src=imgNew, M= roteate_matrix, dsize=(totalwidth, totalHeight))
    cv2.imwrite(testDir + 'hanger_rotate.jpg', rotated_image)

    imgNew = rotated_image.copy()
    outputDir = testDir + '2/'
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    x1, y1, x2, y2 = check_hanger(rotated_image, outputDir)
    print(f'${x1} ${y1} ${x2} ${y2}')
    
    cut_x = x1
    cut_y = y1
    point_A = (cut_y, cut_x)
    point_B = (cut_y + 100, cut_x) #构造出一条直线  

    right_raw, left_raw = cut(imgNew, point_A, point_B)

    left_raw = np.fliplr(left_raw)

    cv2.imwrite(testDir + 'right_raw.jpg', right_raw)
    cv2.imwrite(testDir + 'left_raw.jpg', left_raw)

    right = right_raw[0:right_raw.shape[0], cut_x:right_raw.shape[1]]
    cv2.imwrite(testDir + 'right.jpg', right)
    left = left_raw[0:left_raw.shape[0], totalwidth - cut_x:left_raw.shape[1]]
    cv2.imwrite(testDir + 'left.jpg', left)

    right_new = right
    left_new = left
    zeroMat = np.zeros((right.shape[0], np.abs(left.shape[1] - right.shape[1]), 3))
    print(zeroMat.shape)
    if (right.shape[1] < left.shape[1]):
        right_new = np.concatenate((right_new, zeroMat), axis = 1)
    elif (right.shape[1] > left.shape[1]):
        left_new = np.concatenate((left_new, zeroMat), axis = 1)

    dst = cv2.addWeighted(left_new.astype(np.uint8), 0.8, right_new.astype(np.uint8), 0.4, 0)
    cv2.imwrite(testDir + 'dst.jpg', dst)

    total = np.logical_or(right_new, left_new)
    piece = np.logical_and(right_new, left_new)
    rate = np.count_nonzero(piece) / np.count_nonzero(total)
    print(rate)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="img/", help="local image position")
    args = parser.parse_args(sys.argv[1:])
    dir = args.dir
    imgList = os.listdir(dir)
    for i in range(len(imgList)):
        path = os.path.join(dir, imgList[i])
        imgName = imgList[i].split(".", 1)[0]
        print(imgName)
        outputDir = os.path.join('output', dir, imgName)
        outputDir += '/'
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        startCalc(path, outputDir)
