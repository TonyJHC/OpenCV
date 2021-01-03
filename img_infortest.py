#openCV는 영상 데이터를 numpy.ndarray타입으로 표현 
#import cv2
#img1 = cv2.imread('cat.bmp',cv2.IMREAD_GRAYSCALE)
#img2 = cv2.imread('cat.bmp',cv2.IMREAD_COLOR)
#img1과 img2는 numpy.ndarray타입으로 저장
#<numpy.ndarray>
# -ndim : 차원수. len(img.shape)과 같음 , 즉 그레이인지 컬러인지 2,3인지만 판단해주면됨
# -shape : 각 차원의 크기 (h,w)그레이스케일,
#          (h,w,3) 컬러 영상 h:세로크기 w:가로크기
#size: 전체 원소 개수 
#dtype : 원소의 데이터 타입.  영상 데이터는 uint8
# 영상의 속성과 픽셀 값 참조
# -그레이스케일 영상 : cv2.CV_8UC1(OpenCv 영상 데이터 자료형, 1은 1채널이라는뜻)
#  : numpy.uint8, shape = (h,w)
# -컬러 영상 : cv2.CV_8UC3 : 
# numpy.uint8, shape =(h,w,3)

import sys
import cv2

#영상 불러오기 
img1 = cv2.imread('cat.bmp',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('cat.bmp',cv2.IMREAD_COLOR)
img3 = cv2.imread('cat.bmp',cv2.IMREAD_COLOR)
img4 = cv2.imread('cat.bmp',cv2.IMREAD_COLOR)

if img1 is None or img2 is None:
    print('Image load failed!')
    sys.exit()

print(type(img1)) #결과 : <clss 'numpy.ndarray'>
print(img1.shape) #결과 : (480,640) : 가로,세로 
print(img2.shape) #결과 : (480,640,3)
print(img1.dtype) #결과 : uint8
print(img2.dtype) #결과 : uint8

#가로 세로 크기 변수로 받기 (그레이 스케일)
h, w =img1.shape
print('grayscale: w * h = {} * {}'.format(w,h))

#가로 세로 크기 변수로 받기 (컬러 스케일)
h, w =img1.shape[:2] #가로,세로 길이인 앞에 인자 2개만 받겠다는 뜻
print('colorscale: w * h = {} * {}'.format(w,h))

#그레이스케일 판별하기 
if img1.ndim ==2: #차원수 판별 2차원이면 그레이스케일 
#if len(img1.shape) ==2:으로도 표현가능 
    print('img1  is a grayscale image')

#영상의 픽셀 값 참조 예제 
# 좌표평면상 20,10에 있는 픽셀값을 알고 싶다 
x= 20 
y= 10
p1 =img1[y,x] #행,열 순 이므로 y, x 순서
print(p1) # 결과 : 238

img1[y,x] = 0 #반대로 영상에 0인 검은점을 넣기

p2 =img2[y,x]
print(p2) #결과: [237 242 232]
img2[y,x] = (0,0,255) #반대로 영상에 0인 빨간점을 넣기(B,G,R)순서

cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
#반대로 영상의 픽셀값 수정하기(루프도 가급적 사용하지 않기 엄청나게 느려짐)
#for y in range(h): #행 전체(y축 ) 
  #  for x in range(w): #열 전체 (x 축)
   #     img3[y, x] = 0  #전체 검은색 픽셀값
    #    img4[y, x] = (0,255,255) #전체 노란색 픽셀값 

#반대로 영상의 픽셀값 수정하기
img3[:,:] = 0  # y좌표 전체 , x좌표 전체 검은색 
img4[:,:] = (0,255,255) # y좌표 전체 , x좌표 전체 노란색 

cv2.imshow('img3',img3)
cv2.imshow('img4',img4)
key = cv2.waitKey()
print(key)

while True:
    if key == 27:
        break



cv2.destroyAllWindows()
