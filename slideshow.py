#특정 폴더에 있는 모든 이미지 파일을 이용하여 슬라이드쇼를 수행
#기능:
#특정 폴더에 있는 이미지 파일 목록 읽기 
#이미지를 전체 화면으로 출력하기 
#일정 시간동안 이미지를 화면에 출력하고, 다음 이미지로 교체하기(무한루프)

import sys
import glob #특정형태의 파일 이름을 가진놈들 가져올 수 있음 
import cv2

img_files =glob.glob('.\\images\\*.jpg')# .\\ (현재폴더 밑에) images\\ (images라는 폴더 밑에) *.jpg 이라는 이름을 가진 놈 전부다 불러오기 
#파일 리스트가 문자열 리스트처럼 저장됨 

if not img_files: # 읽어오기 실패 코드
    print("There are no jpg files in 'images' folder")
    sys.exit()


#전체 화면 출력하기 'image'창 생성
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, 
                      cv2.WINDOW_FULLSCREEN)

#출력하는 무한 루프
cnt=len(img_files)
idx =0 #현재 몇번째 사진을 보고 있는지 저장하기 위해

while True:
    img=cv2.imread(img_files[idx])
    
    if img is None: #이미지 읽어오기 실패시
        print('Image load failed!')
        break

    cv2.imshow('image',img) #읽어오기 성공하면 출력하기
    if cv2.waitKey(1000) == 27 : #1초뒤에  아스키 코드값이 27인 ESC가 눌려져있다면 종료, 아니면 -1을 return해서 무한 루프
        break
    idx +=1
    if idx>= cnt:
        idx = 0

cv2.destroyAllWindows()



