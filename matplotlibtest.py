#matplotlib를 사용하여 영상 출력하기
#컬러 영상의 색상 정보가 RGB순서
#cv2.imread()함수로 불러온 영상의 색상 정보는 BGR 순서이므로 
#이를 RGB순소로 변경해야됨 --> cv2.cvtColor()함수 이용 
#그레이스케일 영상 출력 :
#plt.imshow() 함수에서 컬러맵을 cmap='gray'으로 지정
import matplotlib.pyplot as plt
import cv2

#컬러 영상 출력 
imgBGR =cv2.imread('cat.bmp') #cv2.imread() 불러온 영상 색상 정보는 BGR순서
#BGR로 읽은 것을 RGB로 바꾸기
imgRGB =cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB)

plt.axis('off') #x,y축 없애기
plt.imshow(imgRGB) 
plt.show()

#그레이스케일 영상 출력 : 2차원 행렬형태로 저장(byte값만 저장)
imgGray = cv2.imread('cat.bmp',cv2.IMREAD_GRAYSCALE)

plt.axis('off')
plt.imshow(imgGray,cmap='gray') #cmap을 gray로 설정해야됨 
plt.show()

#두 개의 영상을 함께 출력 
plt.subplot(121)#1행으로 표현 2열로 나눔 1열에 그림을 그려라 
plt.axis('off'), plt.imshow(imgRGB)
plt.subplot(122)#1행으로 표현 2열로 나눔 2열에 그림을 그려라
plt.axis('off'), plt.imshow(imgGray,cmap='gray')
plt.show()

#이렇듯 한 화면에 여러 영상을 출력하고 싶을때
#matplotlib를 사용하면 유용하다.
