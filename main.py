# Haar Cascade FrontalFace Algorithm
# Loading Haar Cas -> initialize camera->reading camera Frame
# ->Convering Color image to Grayscale-> obtaining Face co-ordinates-> drawing the rectangle->
#->Display output
import cv2
# import os
# df='dataset'
# name='chap'
# path=os.path.join(df,name)
# if not os.path.isdir(path):
#     os.mkdir(path)
(width,height)=(130,100)
alg='haarcascade_frontalface_default.xml'
haar_cas=cv2.CascadeClassifier(alg)
cam=cv2.VideoCapture(0)
count=1
while True:
    print(count)
    _,img=cam.read()
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=haar_cas.detectMultiScale(grayImg,1.3,4)
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        faceOnly=grayImg[y:y+h,x:x+w]
        resize=cv2.resize(faceOnly,(width,height))
        #cv2.imwrite('%s/%s.jpeg'%(path,count),faceOnly)
        count +=1
    cv2.imshow('faceDetection',img)
    key=cv2.waitKey(10)
    if key == 27:
        break
print('Face Captured succesfully')
cam.release()
cv2.destroyAllWindows()