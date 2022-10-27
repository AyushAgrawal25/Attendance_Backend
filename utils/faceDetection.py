from mtcnn.mtcnn import MTCNN
import os
import cv2

def crop_faces(imgPath, croppedFacesFolderPath):
    imgName=os.path.basename(imgPath)

    img=cv2.imread(imgPath)
    detector=MTCNN()
    faces=detector.detect_faces(img)
    
    count=1
    for face in faces:
        x,y,w,h=face['box']
        crop_img=img[y:y+h,x:x+w]

        # Save cropped faces
        croppedFaceName=imgName+'_face.'+str(count)+'.jpg'
        cv2.imwrite(os.path.join(croppedFacesFolderPath, croppedFaceName), crop_img)  
        count+=1

    return count-1
