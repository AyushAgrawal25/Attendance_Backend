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

def crop_facesFromVideo(videoPath, croppedFacesFolderPath, rqFps=5):
    videoName=os.path.basename(videoPath)

    cap=cv2.VideoCapture(videoPath)
    
    # Get the number of frames
    frameCount=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    length=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS))
    
    currFps=cap.get(cv2.CAP_PROP_FPS)

    frameInterval=int(currFps/rqFps)

    print('Frame Count: ', frameCount)
    print('Frame Interval: ', frameInterval)

    detector=MTCNN()
    count=1
    frameCount=0
    while True:
        ret, frame=cap.read(cv2.IMREAD_GRAYSCALE)

        # TEMP:
        # Rotate the frame by 180 degrees
        frame=cv2.rotate(frame, cv2.ROTATE_180)

        if not ret:
            break

        frameCount+=1
        if (frameCount % frameInterval)>0:
            continue

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
    