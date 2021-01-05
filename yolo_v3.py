import sys
import numpy as np
import cv2
#coco dataset
#YOLOv3-320 weight 적용
#3개의 ndarray(3개 layer)를 조정할 필요가 있는것이 핵심
#50ms 밑으로 나오려면 gpu사용(사양 좋은것으로)
#cpu로 돌리면 500ms +-
#Yolov3를 OpenCV의 DNN모듈로 돌리면 10배정도 빠름


# 모델 & 설정 파일
model = 'yolo_v3/yolov3.weights' #모델파일
config = 'yolo_v3/yolov3.cfg' #설정파일
class_labels = 'yolo_v3/coco.names' #클래스 이름
confThreshold = 0.5
nmsThreshold = 0.4

# 테스트 이미지 파일
img_files = ['dog.jpg', 'person.jpg', 'sheep.jpg', 'kite.jpg'] #jpg파일

# 네트워크 생성
net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    sys.exit()

# 클래스 이름 불러오기(.txt) 80개
classes = []
with open(class_labels, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

colors = np.random.uniform(0, 255, size=(len(classes), 3)) #각각의 이름마다 바운딩박스 색깔 다르게


# 출력 레이어 이름 받아오기
# 출력이 생긴 layer가 무슨 layer인지 알기 위함
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] #layer가 몇인지 알려주는 코드
# output_layers = ['yolo_82', 'yolo_94', 'yolo_106'] # 결과임

# 실행

for f in img_files:
    img = cv2.imread(f) #80개의 영상 탐색

    if img is None:
        continue

    # 블롭 생성 & 추론  , **참고: 속도 빠르게 하려면 320,320으로 , 정확도를 높이려면 608 608
    blob = cv2.dnn.blobFromImage(img, 1/255., (320 , 320), swapRB=True)#RGB순서이기 때문에 swapRB=True
    net.setInput(blob)
    outs = net.forward(output_layers) #output_layers에 저장된 문자열 즉 해당 layer의 위치의 output을 list로 묶음(outs 는 ndarray 3개)
    
    # outs는 3개의 ndarray 리스트.
    # outs[0].shape=(507, 85), 13*13*3=507
    # outs[1].shape=(2028, 85), 26*26*3=2028
    # outs[2].shape=(8112, 85), 52*52*3=8112
    # 3개를 곂쳐서 가장 적합한 값을 추출
    h, w = img.shape[:2]

    class_ids = []
    confidences = []
    boxes = []

    for out in outs: # outs에 저장된 모든 ndarray를 전부다 체크
        for detection in out: # 각각의 ndarray의 행이 detection
            # detection: 4(bounding box) + 1(objectness_score) + 80(class confidence : 확률값)
            scores = detection[5:] #5번째 이상의 것들 즉 class confidence 80개
            class_id = np.argmax(scores) # confidence중 최대값이 있는 위치의 id값을 뽑아와서
            confidence = scores[class_id] #해당 위치에 있는 confidence값을 체크
            if confidence > confThreshold: #confidence값이 0.5이상이면 바운딩 박스를 취합
                # 바운딩 박스 중심 좌표 & 박스 크기
                cx = int(detection[0] * w) #detection[0] :바운딩 박스 가운데 값
                cy = int(detection[1] * h) #detection[1] :바운딩 박스 가운데 값
                bw = int(detection[2] * w) #detection[2] :바운딩 박스 가로
                bh = int(detection[3] * h) #detection[3] :바운딩 박스 세로

                # 바운딩 박스 좌상단 좌표
                sx = int(cx - bw / 2)
                sy = int(cy - bh / 2)
                
                #일단 저장해서
                boxes.append([sx, sy, bw, bh]) 
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

    # 비최대 억제(저장한것들중 곂치는 것 중에서 가장 좋은 박스 하나만 골라낸다.)
    #boxes에 대해서 confThreshold(0.5)보다 큰 confidence를 가진것 중 nmsThreshold(0.4)이상이 되는 박스를 골라라
    #nmsThreshold ==0.4는 박스가 40%정도 곂쳐있다는 뜻
    #다시말해 박스가 40%정도 곂쳐져 있고 그중 confidence가 0.5이상인 것 하나만 골라서 indices에 저장
    #boxes에 저장된 것 중에 몇번째 박스를 쓰겠는지 indices에 저장
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold) 

    for i in indices: #
        i = i[0]
        sx, sy, bw, bh = boxes[i]
        label = f'{classes[class_ids[i]]}: {confidences[i]:.2}' #classes[class_ids[i]] : 클래스 id에 해당하는 문자열 받아오기, cond~도 똑같음 
        color = colors[class_ids[i]]
        cv2.rectangle(img, (sx, sy, bw, bh), color, 2)
        cv2.putText(img, label, (sx, sy - 10),  #label을 화면에 보여줌
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    #실행하는데 얼마나 시간이 걸렸는지
    t, _ = net.getPerfProfile() 
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.waitKey()

cv2.destroyAllWindows()
"""
YOLOv3 객체 검출
Yolo링크 : https://pjreddie.com/darknet/yolo/
coco data 링크 : https://cocodataset.org/#home

1.학습된 데이터 파일(coco data~ 확장자 weights)
:수만장의 사진 학습후 80개의 클래스로 결과물
-coco data 링크로 들어가서 모델파일 다운로드
  conf와 weight둘다 다운로드 해야됨 
  **참고: Yolov3-tiny는 정확도는 떨어지지만 좀 더 빠르게 검출 할 수 있다.
 링크: https://cocodataset.org/#home
-클래스 이름 파일(80개 클래스 이름)
 링크:https://github.com/pjreddie/darknet/blob/master/data/coco.names

2.네트워크 구조 
-입력: 
416 x 416 컬러 영상 ( 픽셀 값의 범위는 0~1)
RGB순서 (swap이라는 플래그 True로 줘야된다 ?)
-출럭:
하나의 출력이 아닌 3군데가 있음 
ex) 82번째 출력(layer) 94, 106 layer를 조합해서
ex2) 82번째 layer: 13x13x3x(4+1+80) -->> 507x85 행렬
-13x13 :가로 13 세로 13 픽셀
- 3 : 픽셀 각각의 셀에대해서 3개의 박스의 범위
- 4: 4개의 박스에대한 좌표값
- 1: 박스에대한 신뢰값(confidence value)
- 80: 80개의 클래스에대한 확률이 어떻게 되냐
**3개의 layer를 조합해서 최종적인 객체 위치와 크기를 표현

3.Yolov3 입력(Yolov3 320,416,608) (픽셀)
: 320은 빠르지만 정확도 낮음  608은 느리지만 정확도 높음
-그 중간인 (416 416) 사용
.
.
4.Yolov3 출력(3개의 출력 레이어)
:3개의 layer에서 나온값 다 받기
출력 layer 3개의 결과값를 list형태로 묶어서 준다(ndarray형)
ex)
outs[0].shape=(507,85), 507=13*13*3 
outs[1].shape=(2028,85), 2028=26*26*3
outs[2].shape=(8112,85), 8112=52*52*3
**85가 뜻하는 것
-처음4개: 바운더리 박스(객체 인식하고 있다는 표시)에 대한 정보 
  tx,ty : 바운더리 박스의 가운데 값
  tw,th : 바운더리 박스의 가로크기,세로크기
  -->>좌표변환을 해야됨
-5번째(p0): Objectness Score (신뢰값? 이라고 생각 :confidence)
"""

