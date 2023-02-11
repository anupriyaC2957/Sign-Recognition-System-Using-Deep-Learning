import numpy as np
import os
import cv2
import scipy.ndimage as sci



def cropIt(gray,top=10,left=290,right=290,down=10):
    w, h = gray.shape
    croped_image=gray[top:(w-down), right:(h-left)]
    return croped_image

def resizeIt(img,size=100,median=2):
    img=np.float32(img)
    r,c=img.shape

    resized_img=cv2.resize(img,(size,size))
    cv2.imshow("RESIZED IMAGE",resized_img)
    filtered_img=sci.median_filter(resized_img,median)
    cv2.imshow("FILTERED IMAGE",filtered_img)
    return np.uint8(filtered_img)

def preprocessing(img0,IMG_SIZE=100):
    img_resized=resizeIt(img0,IMG_SIZE,1) 
    cv2.imshow("GREYSCALE IMAGE",img_resized)
    img_blur = cv2.GaussianBlur(img_resized,(5,5),0)
    cv2.imshow("IMAGE BLUR",img_blur)
    edges = cv2.Canny(img_resized,170, 300)
    cv2.imshow(" EDGES ",edges)
    imgTh=cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,3)
    cv2.imshow("IMAGE THRESHOLD",imgTh)
    ret,img_th = cv2.threshold(imgTh,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    
    return img_th

DATADIR = "data\\"

ALPHABET = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','1','2','3','4','5','6','7','8','9'] #array containing letters to categorize and create path to video

training_data=[]


IMG_SIZE=200

for category in ALPHABET:
    Path = os.path.join(DATADIR,category)  
    print(Path)
    
    for img_path in os.listdir(Path): 

        img0 = cv2.imread(os.path.join(Path,img_path) ,cv2.IMREAD_GRAYSCALE) 
        img_processed=preprocessing(img0,IMG_SIZE)
        cv2.imshow("FINAL PROCESSED IMAGE ",img_processed)
        cv2.waitKey(1)

        class_num =ALPHABET.index(category)
        training_data.append([img_processed, class_num]) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()


import random

random.shuffle(training_data)

x = []
y = []

for features,label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y = np.array(y)


import pickle

pickle_out = open("x.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

print('done with the data processing')



















