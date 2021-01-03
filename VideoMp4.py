"""
Opencv에서는 카메라와 동영상으로부터 프레임(frame)을
받아오는 작업을 cv2.VideoCapture 클래스 하나로 처리함
cv2.VideoCapture(filename,apiPreference ==None) ->retval 
index : camera id + domain_offset_id
        시스템 기본 카메라를 기본방법으로 열려면 index에 0을 전달
apiPreference: 선호하는 카메라 처리 방법을 지정
retval: cv2.videoCapture 객체 

<비디오 캡쳐가 준비되었는지 확인>
cv2.VideoCapture.isOpend() ->retval
retval: 성공하면 True, 실패하면 False

<프레임 받아오기>
cv2.VideoCapture.read(image=None) ->retval,image
retval: 성공하면 True, 실패하면 False.
image: 현재 프레임(numpy.ndarray)

"""
import sys
import cv2

#cap = cv2.VideoCapture()#객체 생성
#cap.open(0) #0 : 기본 카메라 열기 
cap = cv2.VideoCapture('video1.mp4') #프레임에 출력하고 싶은 동영상 이름

#프레임 크기 출력 :get
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #default값이 float이기 때문에 int형 변환(보여주기 편하게)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(w,h)

#프레임의 크기를 지정 : set
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,320 ) #default값이 float이기 때문에 int형 변환(보여주기 편하게)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)


if not cap.isOpened():
    print('video open failed!')
    sys.exit()

while True: #루프문 
    #현재프레임을 하나 씩 읽어오기 
    ret, frame =cap.read() # ret =Ture/False , frame = 현재프레임(numpy.ndarray)
    
    if not ret: #사용자의 카메라와 달리 video를 켜면 자동으로 해당 코드에 걸려서 종료
        break    

    #윤관선 나타내기
    edge = cv2.Canny(frame, 50 , 150) 

    cv2.imshow('frame',frame)
    cv2.imshow('edge',edge)
    if cv2.waitKey(20) ==27: # 20미리세컨드 기다린다 #ESC 누를시 루프문 종료
        break

cap.release() # 객체 할당해제
cv2.destroyAllWindows()
