import cv2
import tensorflow as tf
import numpy as np
import scipy.ndimage as sci
import time
import os


def resizeIt(img,size=100,median=2):
    img=np.float32(img)
    r,c=img.shape
    resized_img=cv2.resize(img,(size,size))
    filtered_img=sci.median_filter(resized_img,median)
    rs= np.uint8(filtered_img)
    #return np.uint8(filtered_img)
    return rs

def preprocessing(img0,IMG_SIZE=100):
    img_resized=resizeIt(img0,IMG_SIZE,1)
    img_blur = cv2.GaussianBlur(img_resized,(5,5),0)
    imgTh=cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,3)
    ret,img_th = cv2.threshold(imgTh,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    return img_th

ALPHABET =  ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','1','2','3','4','5','6','7','8','9'] 

prev=""
model = tf.keras.models.load_model("model_name.model")

prev_time = time.time()
path = "data\\S\\1.jpg"
   
src = cv2.imread(path) 
for i in src:  
  print("MATRIX OUTPUT")
  print()
  print(i)
  print()
  print("OUTPUT")  
  print()
  
IMG_SIZE = 200  
img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

 
img_test = preprocessing(img_gray,IMG_SIZE)

    
 
prediction = model.predict([img_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)])
print(prediction)

print(prediction[0])
    
text = ALPHABET[int(np.argmax(prediction[0]))]
cv2.imshow('frame', img_test)
      
 
#_ = os.system('cls')
print(prediction)

print(prediction[0])
  
print('Alphabet: '+text+' Time Required: '+str(time.time()-prev_time))
prev_time = time.time()

