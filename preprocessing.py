from curses.ascii import SI
import os
from tqdm import tqdm
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import re
import cv2


it = 10                     # 생성할 이미지 개수
size = 160
width_shift = size*0.1      # 수평으로 이동(pixel scale)
height_shift = size*0.1     # 수직으로 이동(pixel scale)
rotate = 30         # rotation(degree scale)

# to get the files in proper order
def sorted_alphanumeric(data):  
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(data,key = alphanum_key)

path = '/Users/chajiwoo/Desktop/newDataset/color'
files = os.listdir(path)
files = sorted_alphanumeric(files)
files = files[:1]
#print(files)
# print(load_img)
# print(tqdm(files))

# define generator
datagen = ImageDataGenerator(
    width_shift_range=[(-1)*width_shift, width_shift],      # 수평 이동
    height_shift_range=[(-1)*height_shift, height_shift],   # 상하 이동
    horizontal_flip=True,   # 좌우 반전
    vertical_flip=False,    # 상하 반전(false)
    rotation_range=rotate,  # rotate까지 랜덤 각도 회전
    zoom_range=0.4)   # zoom?

for i in range(len(tqdm(files))):
    
    img = cv2.imread(path + '/'+str(i)+'.jpg',1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = img_to_array(img)        # 이미지를 array 형태로 바꿈
    img = np.expand_dims(data, 0)  # 첫 번째 차원을 추가하여 확장
    
    #img = np.expand_dims(ndimage.imread(path + '/'+str(i)+'.jpg'),0)
    #img = np.expand_dims(ndimage.imread(path + '/'+str(i)+'.jpg',1), 0)
    datagen.fit(img)
    # prepare iterator
    for x, val in zip(datagen.flow(img,                    #image we chose
        save_to_dir=path,     #this is where we figure out where to save
         save_prefix='0',        # it will save the images as 'aug_0912' some number for every new augmented image
        save_format='jpg'),range(it)) :     # here we define a range because we want 10 augmented images otherwise it will keep looping forever I think
        pass

    print('generate data about '+str(i)+'.jpg')


    # it = array_to_img(datagen.flow(samples)) 
    

    # 시각화
    # figure 생성
    #fig = plt.figure(figsize = (30, 30))
    #print(type(it))
    # 9개 이미지 생성 
    '''
    for j in range(9):
        # plt.subplot(3, 3, i+1)

        # generate batch of images
        #batch = it.next()

        # convert to unsigned integers for viewing
        #image = batch[0].astype('uint8')

        # add image in file
        image = it.next()
        image = image.astype('int8')
        cv2.imwrite(path + '/'+str(i)+'_'+str(j)+'.jpg', it.next().astype('uint8'))
        
        # debug
        print('create'+str(i)+'_'+str(j)+'.jpg')
        plt.title('create'+str(i)+'_'+str(j)+'.jpg')
        plt.figure(figsize=(30,30))
        plt.plot(image)
        plt.imshow()
    
        # show the figure 
        #plt.title("result")
    #plt.show()
    '''
