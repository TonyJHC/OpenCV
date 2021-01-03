#지정한 크기로 새 영상 생성하기 
#numpy.empty(shape, dtype =float,...) -->> 임의의 값으로 초기화된 배열을 생성
#numpy.zeros(shape, dtype =float,...) -->> 0으로 초기화된 배열을 생성
#numpy.ones(shape, dtype =None,...) -->> 1로 초기화된 배열을 생성
#numpy.full(shape, fill_value ,dtype =None,...) -->> fill_value로 초기화된 배열을 생성 

import numpy as np 
import cv2

#새 영상 생성하기 
img1 = np.empty((240,320), dtype = np.uint8) #grayscale image
img2 = np.zeros((240,320,3),dtype=np.uint8)
img3 = np.ones((240,320,3),dtype=np.uint8) *255 # 전부다 255가 되기 때문에 흰색
img4 = np.full((240,320),128,dtype=np.uint8) #128 : 회색

cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.imshow('img3',img3)
cv2.imshow('img4',img4)

key = cv2.waitKey()


cv2.destroyAllWindows()
