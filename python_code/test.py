import os

import cv2
import numpy as np
def GB_total(grid_RGB,cX,cY,interval):
    GB = 0
    t = 0
    for row in grid_RGB[cY:cY+1]:
        for column in row[cX-interval:cX+interval]:
            GB += int(column[1])+int(column[2])
    return GB
cwd = os.getcwd() + "/Images/"
files = sorted(os.listdir(cwd))
for file in files:
    print(file)
    img = cv2.imread(cwd+file)
    grid_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)

    lower1 = np.array([0,0,240])
    upper1 = np.array([180,150,255])
    mask1 = cv2.inRange(grid_HSV, lower1, upper1)
    res1 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask1)

    lower2 = np.array([0,0,240])
    upper2 = np.array([180,150,255])
    mask2 = cv2.inRange(grid_HSV, lower2, upper2)
    res2 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask2)

    mask3 = mask1 + mask2

    mask3 = cv2.GaussianBlur(mask3, (5, 5), 0)
    ret, binaryMask = cv2.threshold(mask3, 100, 255, cv2.THRESH_BINARY);
    contours, hierarchy = cv2.findContours(binaryMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
    cv2.imshow("image", img)
    # maxX = 0
    # maxY = 0
    t = 1
    num = 0
    interval = 3
    max = 200  #filter for max GB total
    for i in range(len(contours)):
        M = cv2.moments(contours[i])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX,cY = 0,0
        print("center point",t,":",cX,cY)
        t += 1
        print("GB total:",GB_total(grid_RGB,cX,cY,interval))
        if GB_total(grid_RGB,cX,cY,interval)/(interval*2*2) < max:
            for j in contours[i]:
                print("x:",j[0][0],"y:",j[0][1])
            cv2.drawContours(img, [contours[i]], -1, (0,255,0), thickness = -1)
        print()
    #print("Max:",maxX,maxY)



    cv2.imshow("mask3", mask3)


    cv2.imshow("img2", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
