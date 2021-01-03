#OpenCV - Tony
#contact : jongvin01@naver.com
import sys
import cv2

print('hello, opencv',cv2.__version__)

img = cv2.imread('cat.bmp')
img2 = cv2.imread('cat_gray.png')
img3 = cv2.imread('dayun.bmp')
if img is None:  #예외처리 , cat.bmp를 제대로 못 불러왔을때
    print('Image load failed')
    sys.exit()

cv2.imwrite('cat_gray.png',img) #영상을 cat_gray.png 파일로 저장

cv2.namedWindow('image',cv2.WINDOW_NORMAL) # 창을 하나 생성해줌'image'라는 이름의, WINDow~ 는 화면의 크기를 마우스로 조절할 수 있음
cv2.namedWindow('image2') # 창을 하나 생성해줌'image'라는 이름의
cv2.namedWindow('image3')
cv2.imshow('image',img) # image창에 img에 저장된 것을 보여줘라
#만약 image라는 창이 없다면 window오토사이즈로 default 생성
#img가 들어간 곳은 출력할 영상 데이터 (numpy.ndarray)
# numpy의 ndarray 클래스는 다차원 행렬 자료구조 클래스
# 선형대수 연산이 일반 파이썬의 List보다 장점이 많음
# c언어의 배열의 특성처럼 연속적인 메모리에 배치됨
#인접한 메모리 배치로 인해 선형대수 연산의 속도를 향상시킬 수 있음
#img가 들어간 칸에는 uint8형으로 넣는다고 가정하셈
#다른 자료형이면 수정사항이 있음(나중에 배울거임)
cv2.imshow('image2',img2) # image창에 img에 저장된 것을 보여줘라
cv2.imshow('image3',img3)
key = cv2.waitKey() # 키보드입력을 기다리기 또한 실제로 화면에 보여지게하는것
#waitKey가 없으면 실제로 화면에 원하는대로 띄워지지 않음
#cv2.waitKey(3000) 3초 있다가 화면 없어짐
# ret값은 키보드로 받은 문자의 아스키 코드 ex) esc == 27
print(key)

while True:
  if key == 27:
      break

cv2.destroyAllWindows() #화면의 모든창을 닫아라
