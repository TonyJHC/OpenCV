#지정한 크기로 새 영상 생성하기 
#numpy.empty(shape, dtype =float,...) -->> 임의의 값으로 초기화된 배열을 생성
#numpy.zeros(shape, dtype =float,...) -->> 0으로 초기화된 배열을 생성
#numpy.ones(shape, dtype =None,...) -->> 1로 초기화된 배열을 생성
#numpy.full(shape, fill_value ,dtype =None,...) -->> fill_value로 초기화된 배열을 생성 

import numpy as np 
import cv2
import sys
#새 영상 생성하기 
img1 =cv2.imread('HappyFish.jpg')

img2 =img1  #img1 과  img2가 데이터를 공유(참조와 같은 개념)  call of referenc로 이해
img3 = img1.copy() # call of value로 이해 (복사본을 완전히 새로 만듬)

#부분 영상 생성하기
img4=  img1[40:120, 30:150].copy() #메모리를 새롭게 할당하여 부분영상 저장
img5 = img1[120:200, 150:270].copy()#메모리를 새롭게 할당하여 부분영상 저장


img1[:, :] = (0, 255, 255) # img1의 데이터에 노란색 픽셀값 저장
cv2.circle(img2, (50,50), 20, (0,0,255),2) # img2에 원그리기 img1과 메모리 공유하니까 img1에도 그려짐
# 좌표(50,50) ,반지름(20), 픽셀값(0,0,255), 두께(2)

cv2.imshow('img1',img1) #노란색 
cv2.imshow('img2',img2) #노란색 
cv2.imshow('img3',img3) #그대로 happyfish출력
cv2.imshow('img4',img4) #부분 영상 출력
cv2.imshow('img5',img5) #부분 영상 출력

key = cv2.waitKey()

if key ==27:
    sys.exit()

cv2.destroyAllWindows()
