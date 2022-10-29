from mtcnn.mtcnn import MTCNN
import os
import cv2

# Use retina CNN for cropping faces
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

def crop_facesFromVideo(videoPath, croppedFacesFolderPath):
    videoName=os.path.basename(videoPath)

    cap=cv2.VideoCapture(videoPath)
    detector=MTCNN()
    count=1
    while True:
        ret, frame=cap.read()
        if not ret:
            break

        faces=detector.detect_faces(frame)
        for face in faces:
            x,y,w,h=face['box']
            crop_img=frame[y:y+h,x:x+w]

            # Save cropped faces
            croppedFaceName=videoName+'_face.'+str(count)+'.jpg'
            cv2.imwrite(os.path.join(croppedFacesFolderPath, croppedFaceName), crop_img)  
            count+=1

    cap.release()
    return count-1
    