import os
import cv2



SIZE = 16
color_img = []
path = './dataset2/color'
grayPath = './dataset2/gray'


for i in range(len(os.listdir(path))):
    imgList = os.listdir(path)
    os.rename(os.path.join(path, imgList[i]), os.path.join(path, str(i)+'.jpg'))
    
    img = cv2.imread(os.path.join(path, str(i)+'.jpg'))
        # open cv reads images in BGR format so we have to convert it to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #resizing image
    cv2.imwrite(os.path.join(grayPath, str(i)+'.jpg'), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    print('create ' + str(i)+'.jpg')