import os

import cv2


SIZE = 160
color_img = []
path = './dataset2/color'
grayPath = './dataset2/gray'
newPath = './newDataset/color'
newGrayPath = './newDataset/gray'

for i in range(len(os.listdir(path))):
    imgList = os.listdir(path)
    # os.rename(os.path.join(path, imgList[i]), os.path.join(path, str(i)+'.jpg'))
    
    img = cv2.imread(os.path.join(path, imgList[i]))
        # open cv reads images in BGR format so we have to convert it to RGB
        #resizing image
    img = cv2.resize(img, (SIZE+80, SIZE))[0:160, 40:200]
    cv2.imwrite(os.path.join(newPath, imgList[i]), img)
    cv2.imwrite(os.path.join(newGrayPath, imgList[i]), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    
    print('create ' + str(i)+'.jpg')