import sys
import numpy as np
import cv2
from tqdm import tqdm
import os

model = '/Users/krc/Downloads/model/opencv_face_detector_uint8.pb'
config = '/Users/krc/Downloads/model/opencv_face_detector.pbtxt.txt'

def detectAndDisplay(frame):
        # Network 객체 생셩
        net = cv2.dnn.readNet(model, config)

        # frame -> (300,300) resize, (104, 177, 123) mean 
        blob = cv2.dnn.blobFromImage(frame, 1, (300,300), (104, 177, 123))
        net.setInput(blob)
        detections = net.forward() # 출력 (1,1,200,7)

        for i in range(0, detections.shape[2]): 
              
                confidence = detections[0, 0, i, 2]
                min_confidence=0.3
                
                if confidence > min_confidence:
                        
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    crop = frame[startY:endY, startX:endX]
                    
                    return crop
                else:
                    return None
                    
# img = cv2.imread('./lfw_funneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg')
# (height, width) = img.shape[:2]
# crop = detectAndDisplay(img)
# cv2.imwrite(f'cropped_{i}')
# cv2.imshow("cropped Image", crop)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cropping & save

path = '../megaage_asian/'

for fold in ['train','test']:
    for filename in tqdm(os.listdir(path+fold)):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(path+fold,filename))
    
            (height, width) = img.shape[:2]
            crop = detectAndDisplay(img)
            if crop is not None:
                cv2.imwrite(f'./cropped/{fold}/{filename}', crop)