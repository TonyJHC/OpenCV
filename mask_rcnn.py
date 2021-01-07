import sys
import numpy as np
import cv2
"""
Mask-RCN사용
구글넷 이라는 딥러닝 알고리즘을 사용
(구글넷:영상에 있는 대표적인 객체가 무엇이 있는지 수행)
1.영역 분할이란?
<정의>
객체의 바운딩 박스뿐만 아니라 픽셀 단위 클래스 분류까지 -> 객체의 윤곽 구분
<single object>
Classifiacation 분류 
Classifiacation + Localization :분류 및 위치까지
<Multiple objects>
Object Detection : 바운딩 박스까지 찾는것 
Instance Segmentation : 바운딩 박스 및 객체 외곽선까지 
2.Mask-RCNN이란?
:대표적인 객체 영역 분할 딥러닝 알고리즘(2017년~)
-Faster R-CNN+FCN(semantic segmentation)
-Faster R-CNN을 이용해 객체 검출(Yolov3이전의 기술) :object detection
-FCN(semantic segmentation)
Faster R-CNN으로 객체단위로 바운딩 박스로 잘라내고(객체검출)
바운딩 박스에대해서 FCN을 동작시킴(객체 단위 클래스 분류)
그 결과 객체 단위 마스크 맵(히트맵)이 만들어짐
<Mask-RCNN input>
-size : 임의의 크기(auto resize : 대략 600정도로)
-scale: 1(1~255)
-Mean:[0,0,0]
-RGB:true
<Mask-RCNN output>
1)2개의 출력 레이어 사용 
 -'detection_out_final':바운딩 박스의 정보를 출력으로 주는 layer이름
  ->(ndarray형) boxex: boxex.shape = (1,1,100,7) #0,1번째 인덱스는 더비(쓰레기) 100은 최대 100개의 객체를 검출한다는뜻,7은 클래스 정보 또는 바운딩 박스 정보
 -'detection_masks':각각의 바운딩 박스의 마스크 정보(객체의 윤곽을 나타내는)를 출력으로 주는 layer이름
  ->(ndarray형) masks: masks.shape=(100,90,15,15)100은 최대 100개 ~ 15 x 15 마스크가 90개 클래스에대한 마스크가 나온다
       #90개의 마스크맵을 다쓸 필요는 없고 가장 주된 객체 번호(클래스)에 해당하는 마스크만 쓰면됨 
 **이러한 두가지 출력을 후처리과정을 거쳐 결과물을 만들어냄
"""

def drawBox(img, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(img, (left, top), (right, bottom), colors[classId], 2)

    label = f'{classes[classId]}: {conf:.2f}'

    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(img, (left - 1, top - labelSize[1] - baseLine),
                  (left + labelSize[0], top), colors[classId], -1)
    cv2.putText(img, label, (left, top - baseLine), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 1, cv2.LINE_AA)


# 모델 & 설정 파일
model = 'mask_rcnn/frozen_inference_graph.pb'
config = 'mask_rcnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
class_labels = 'mask_rcnn/coco_90.names'
confThreshold = 0.6
maskThreshold = 0.3

# 테스트 이미지 파일
img_files = ['dog.jpg', 'traffic.jpg', 'sheep.jpg']

# 네트워크 생성
net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    sys.exit()

# 클래스 이름 불러오기

classes = []
with open(class_labels, 'rt') as f: #90개의 클래스 이름을 한줄씩 자르기
    classes = f.read().rstrip('\n').split('\n') #문자열 형태

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 전체 레이어 이름 받아오기
'''
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
for name in layer_names:
    print(name)
'''

# 실행

for f in img_files:  #영상파일 불러오기
    img = cv2.imread(f)

    if img is None:
        continue

    # 블롭 생성 & 추론(블롭(binary large object):이미지,비디오,사운드 등과 같은 멀티 미디어 객체들을
    # 저장하는 자료형?) 최대 4GB 이진 데이터 저장 가능->이것의 포인터를 DB에 저장
    blob = cv2.dnn.blobFromImage(img, swapRB=True) 
    net.setInput(blob)
    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])
     #detection_out_final이 바운딩 박스 정보
     #'detection_masks'이 각각의 바운딩 박스에서 마스크 정보
    # boxes.shape=(1, 1, 100, 7) #0,1번째 인덱스는 더비(쓰레기) 
    #                            100은 최대 100개의 객체를 검출한다는뜻,
    #                            7은 클래스 정보 또는 바운딩 박스 정보
    # masks.shape=(100, 90, 15, 15) #100은 최대 100개 ~ 
    #                                15 x 15사이즈 마스크가 90개 클래스에대한 
    #                                마스크가 나온다
                                     
    #90개의 마스크맵을 다쓸 필요는 없고 가장 주된 객체 번호(클래스)에 해당하는 마스크만 쓰면됨 
    #**이러한 두가지 출력을 후처리과정을 거쳐 결과물을 만들어냄
    #객체 바운딩박스를 계산하고 해당 바운딩 박스 안에서 마스크정보를 이용해서 
    #객체윤곽을 뽑아내야됨

    h, w = img.shape[:2]
    numClasses = masks.shape[1]  # 90(코코 데이터셋의 클래스 갯수)
    numDetections = boxes.shape[2]  # 100(detection갯수 default값) 만약 100보다 적으면 confidence값이 적게 나옴

    boxesToDraw = []
    for i in range(numDetections): # 0~99까지  각각의 바운딩박스정보 받아오기
        box = boxes[0, 0, i]  # box.shape=(7,) --> 0,classID(클래스번호),confidence값,x1,y1,x2,y2 :총 7개
        #x1,y1은 바운딩박스의 좌측 상단 , x2,y2는 우측 하단점의 좌표
        #단 실수로 나오기때문에 가로크기,세로크기를 곱해줘야함
        mask = masks[i]  # mask.shape=(90, 15, 15) 15x15마스크맵 90개
        score = box[2]
        if score > confThreshold:  
            classId = int(box[1]) #90개의 마스크맵중 confidence값이 confthreshold보다 높은거만 사용 
            #print(classId, classes[classId], score)
            
            #바운딩박스 실제좌표 계산
            x1 = int(w * box[3])
            y1 = int(h * box[4])
            x2 = int(w * box[5])
            y2 = int(h * box[6])

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            boxesToDraw.append([img, classId, score, x1, y1, x2, y2])
            classMask = mask[classId] #15 x15 float형의 matrix, 해당 martix는 객체가 있는부분은 값이 크게,배경은 작게 
            #즉 배경은 검은색, 객체는 흰색에 가까운 값

            # 객체별 15x15 마스크를 바운딩 박스 크기로 resize한 후, 불투명 컬러로 표시
            classMask = cv2.resize(classMask, (x2 - x1 + 1, y2 - y1 + 1))
            mask = (classMask > maskThreshold)
            
            #불투명하게 ?? 클래스의 윤곽 정보
            roi = img[y1:y2+1, x1:x2+1][mask]
            img[y1:y2+1, x1:x2+1][mask] = (0.7 * colors[classId] + 0.3 * roi).astype(np.uint8)

    # 객체별 바운딩 박스 그리기 & 클래스 이름 표시
    for box in boxesToDraw:
        drawBox(*box)

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.waitKey()

cv2.destroyAllWindows()

