#ROI란(Region of Interest , 관심 영역 )
#영상에서 특정 연산을 수행하고자 하는 임의의 부분 영역

#마스크 연산 : 원하는 객체 픽셀을 뽑아내기위해 만든 흑백 영상
#객체와 객체가 아닌 background를 분리 
#ex) 사과 추출 : 
# 사과인 부분은 흰색 (이 부분이 ROI)
# 사과가 아닌 background를 검은색으로 칠함 
# OpenCV는 일부 함수에 대해 ROI연산을 지원,
# 이때 마스크 영상을 인자로 함께 전달해야됨 
#ex) cv2.copyTo(), cv2.calcHist(), cv2.bitwise_or(), cv2.matchTemplate(),etc.
#마스크 영상은 cv2.CV_8UC1 타입(그레이스케일 영상)
#마스크 영상의 픽셀 값이 0(검은색)이 아닌 위치에서만 연산이 수행
  # -- > 보통 마스크 영상으로는 0또는 255로 구성된 이진 영상(binary image)을 사용
# 즉, ROI는 영상에서 픽셀값이 0이 아닌 부분


#마스크 연산을 지원하는 픽셀 값 복사 함수
#cv2.copyTo(src, mask, dst= None) 
# -> src : 입력 영상 
# -> mask : 마스크 영상.  cv2.CV_8U.(numpy,uint8) 
#    0이 아닌 픽셀에 대해서만 복사연산을 수행. 
# -> dst: 출력 영상. 만약 src와 크기 및 타입이 같은 dst를 
#     입력으로 지정하면 dst를 새로 생성하지 않고 연산을 수행.
#    dst  dst에 None을 하면 출력만 
#    매개변수 dst에 값을 넣어주면 입력과 출력 동시에
# 즉 , src영상에서 roi로 설정되어 있는 mask(mask영상에서 0이 아닌부분)의 픽셀 
#  위치에 대해서만 src 픽셀값을 dst로 복사  
#예제로 이해 
import sys
import cv2

#마스크 영상을 이용한 영상 합성
src = cv2.imread('airplane.bmp', cv2.IMREAD_COLOR)
mask = cv2.imread('mask_plane.bmp', cv2.IMREAD_GRAYSCALE)
dst = cv2.imread('field.bmp', cv2.IMREAD_COLOR)

cv2.imshow('src',src)
cv2.imshow('dst',dst)
cv2.imshow('mask',mask)


#src영상에서 비행기 부분(mask)만 dst영상에 합성하고 싶음
cv2.copyTo(src,mask,dst) # src영상에서 mask부분을 추출해서 dst에 합성
# dst[mask > 0 ] = src[mask > 0] # 같은 표현
#주의)) src,mask,,dst는 사이즈가 같아야 하고 src와 dst는 타입이 같아야함(src가 grayscale이면 dst도 grayscale이어야함)
#        mask는 언제나 grayscale이어야함
cv2.imshow('dst_result',dst)

cv2.waitKey()
cv2.destroyAllWindows()

