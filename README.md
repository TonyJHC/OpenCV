# OpenCV
 I will be a competent engineer :)
"""딥러닝 이란 
2000년대부터 사용되고 있는 심층 신경망(deep neural network)의 또 다른 이름
입력영상을 잔득주면 자동으로 분류해준다(머신러닝과 다른점)
Object Detection, Object Segmentation(물체의)

퍼셉트론(Perceptron)
다수의 입력으로부터 가중합을 계산하고, 이를 이용하여 하나의 출력을
만들어내는 구조(1950년대)
 
단층 퍼셉트론 vs 다층 퍼셉트론 vs 심층 신경망(Deep Neural Network)
- 다층: 은닉층이 한 개 이상
- 심층: 은닉층이 두 개 이상

신경망 학습이란? 
-좋은 가중치(weight)를 결정하는 방법 
 가중치는 딥러닝을 시켜 나온 결과물(퍼셉트론을 거쳐나온 결과값)

신경망 학습 방법
-학습의 기준이 되는 비용(cost)또는 손실(loss)함수를 정의한 후 ,
 비용을 최소화하는 방향으로 학습을 진행 
- 경사 하강법(gradient descent)또는 오류역전파(error backpropagation)알고리즘 사용 
  경사 하강법 : 그래디언트 반대 방향으로 이동하면서 최솟값 위치를 찾는 방법
  오류 역전파 : 미분의 체인룰(chain-rule)을 이용하여 전체 가중치를 업데이트하는 방법
  **tensorflow혹은 pytorch에서 자동으로 결정해준다

심층 신경망의 문제점과 해결 방안
1)학습이 제대로 안 됨 
 -Vanishing Gradient  -> ReLU(Rectified Linear Units) 
 -Overfitting -> Regularization (Dropout,Batch Normalization)
 -Weight initialization -> Random initialization, Xavie method,etc.
2)학습이 너무 느림
 -하드웨어 성능 ->cpu, gpu발전
 -Gradient Decent -> SGD, Mini-Batch , Adam method
3)데이터 셋 부족 -> 카메라 보급,인터넷 발전
                -> Larage dataset : ex)ImageNet, COCO,etc
"""
