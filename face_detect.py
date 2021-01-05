import sys
import numpy as np
import cv2
"""
OpenCV DNN 얼굴 검출(face_detect.py)
-SSD(Single Shot MultiBox Detector)알고리즘 기반 얼굴 검출 네트워크
-SSD는 2016년 기준 다른 객체 검출 알고리즘(ex. YOLOv1과 비교하여 성능과 속도 두 가지를 모두 만족시킴)
-DNN 모듈을 사용한 얼굴 검출 
-딥러닝 기반의 검출(기존의 Haar-Cascade방법, DL방법)
-300 x 300 고정 


https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
1)다운로드할 파일들
opencv_face_detector.pbtxt
deploy.prototxt.txt
weights.meta4.txt
download_models.py

**https://github.com/opencv/opencv/blob/master/samples/dnn/models.yml
>>models.yml 모듈이 없다고 에러뜨면 해당 링크로 가서 다운받기

2)해당 파일들이 있는 경로상에 cmd창 켜기 
python download_weights.py 실행

"""

#학습된 파일들
model = 'opencv_face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
config = 'opencv_face_detector/deploy.prototxt' #학습된 설정파일
#model = 'opencv_face_detector/opencv_face_detector_uint8.pb'
#config = 'opencv_face_detector/opencv_face_detector.pbtxt'

#카메라 오픈
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Camera open failed!')
    sys.exit()
#모델파일 불러오기
net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    sys.exit()
#프레임을 하나씩 받아오기(루프문을 통해 전체 다 받아옴)
while True:
    ret, frame = cap.read()

    if not ret:
        break
    #현재 프레임인 frame(640,480)을 (300,300)으로 재설정, 컬러영상
    #재설정하는 이유는 딥러닝 기반의 방법은 입력영상의 크기가 300 x 300고정, 그 다음 얼굴을 찾는다
    #blob은 4차원행렬
    blob = cv2.dnn.blobFromImage(frame, 1, (400, 400), (104, 177, 123))
    net.setInput(blob)
    out = net.forward() #4차원(ndarray)중에 뒤에 있는 2개의 차원만 받아오기(0번째 1번째 값은 0아니면 1이라고 생각 무시고고) (1,1,200,10)

    detect = out[0, 0, :, :]#4차원(ndarray)중에 뒤에 있는 2개의 차원만 받아오기
    (h, w) = frame.shape[:2]#입력영상크기:4차원(ndarray)중에 뒤에 있는 2개의 차원만 받아오기

    for i in range(detect.shape[0]):
        confidence = detect[i, 2] #0,1번째는 0아니면 1 3번째 인덱스에 확률값있음(실수값)
        if confidence < 0.5: #확률값이 0.5보다 클때만(전체행렬을 검사하지 않도록)
            break
       #얼굴 인식을 하고 있다는 표식을 해주기 위해
       #실제 좌표 검출-->>입력영상의 크기를 곱해줘야함 
        x1 = int(detect[i, 3] * w)
        y1 = int(detect[i, 4] * h)
        x2 = int(detect[i, 5] * w)
        y2 = int(detect[i, 6] * h)
       
       #사각박스를 그려줌 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

        label = f'Face: {confidence:4.2f}' #소수점 둘째자리까지 출력해줌
        cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
