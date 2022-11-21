from cProfile import label
from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pymysql
import secrets
import os
import shutil
from utils.faceClassifier import extract_embeddings, getClassifier, load_shuffleFaceNet
from utils.faceDetection import crop_faces, crop_facesFromVideo
import pandas as pd
from utils.faceRecognition import predict

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://remote:password@localhost:3306/attendance_backend'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Student(db.Model):
    __tablename__ = 'student'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    rollNo= db.Column(db.Integer, nullable=False, unique=True)
    name = db.Column(db.String(20), nullable=False)
    branch=db.Column(db.String(20), nullable=False)
    semester=db.Column(db.Integer, nullable=False)
    anchor_link=db.Column(db.String(100), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    status=db.Column(db.Boolean, default=True)

    def __repr__(self):
        return f"{self.rollNo} - {self.name}"

@app.route('/')
def index():
    return "Hello World"

@app.route('/addStudent', methods=['POST'])
def addStudent():
    name=request.form['name']
    rollNo=request.form['rollNo']
    branch=request.form['branch']
    semester=request.form['semester']
    anchor_link="/static/uploads/"+request.form['rollNo']+"/";

    try:
        os.mkdir(os.path.join(app.static_folder, 'uploads', rollNo))
        student=Student(name=name, rollNo=rollNo, branch=branch, semester=semester, anchor_link=anchor_link)
        db.session.add(student)
        db.session.commit()

        return "Student Added Successfully"    
    except:
        return "There was an issue adding your student"

@app.route('/uploadAnchor', methods=['POST'])
def uploadAnchor():
    rollNo=request.form['rollNo']
    requestId=secrets.token_hex(16)
    os.mkdir(os.path.join(app.static_folder, 'temps', requestId))

    # Check whether image is present in request.files
    if request.files.get('image'):
        file=request.files['image']    
        file.save(os.path.join(app.static_folder, 'temps', requestId, file.filename))

        faceCount=crop_faces(os.path.join(app.static_folder, 'temps', requestId, file.filename), os.path.join(app.static_folder, 'uploads', rollNo))
        print(faceCount)


    if request.files.get('video'):
        file=request.files['video']
        file.save(os.path.join(app.static_folder, 'temps', requestId, file.filename))

        faceCount=crop_facesFromVideo(os.path.join(app.static_folder, 'temps', requestId, file.filename), os.path.join(app.static_folder, 'uploads', rollNo))
        print(faceCount)

    # Delete the temp folder
    shutil.rmtree(os.path.join(app.static_folder, 'temps', requestId))
    
    return "File Uploaded Successfully"

@app.route('/attendance', methods=['POST'])
def attendance():
    file=request.files['image']
    requestId=secrets.token_hex(16)
    
    folderPath=os.path.join(app.static_folder, 'temps', requestId)
    os.mkdir(folderPath)

    orgFolder=os.path.join(folderPath, 'org')
    os.mkdir(orgFolder)

    filePath=os.path.join(orgFolder, file.filename)
    file.save(filePath)

    # Cropping Faces
    facesFolder=os.path.join(folderPath, 'faces')
    os.mkdir(facesFolder)
    facesCount=crop_faces(filePath, facesFolder)
    print('Faces Count: ', facesCount)

    # Face Recognition
    students=Student.query.all()
    rollNos=[student.rollNo for student in students]
    facesFiles=os.listdir(facesFolder)
    
    df=pd.DataFrame(columns=rollNos, index=facesFiles)
    df=df.fillna(-1.0)

    for rollNo in rollNos:
        for faceFile in facesFiles:
            anchorImages=os.listdir(os.path.join(app.static_folder, 'uploads', str(rollNo)))
            if len(anchorImages) == 0:
                continue
            
            confidence=0.0
            for anchorImage in anchorImages:
                currConfidence=predict(os.path.join(app.static_folder, 'uploads', str(rollNo), anchorImage), os.path.join(facesFolder, faceFile))
                # confidence+=currConfidence
                confidence=max(confidence, currConfidence)
            
            # confidence/=len(anchorImages)            
            df.loc[faceFile, rollNo]=confidence

    attendanceResult=[]
    for rollNo in rollNos:
        if df[rollNo].max() >= 0.5:
            attendanceResult.append({
                'rollNo': rollNo,
                'status': 'Present',
                'confidence': df[rollNo].max()
            })
        else:
            attendanceResult.append({
                'rollNo': rollNo,
                'status': 'Absent',
                'confidence': df[rollNo].max()
            })

    # Delete the requestID folder 
    shutil.rmtree(folderPath)

    # Return a dummy json
    return attendanceResult

@app.route('/videoAttendance', methods=['POST'])
def videoAttendance():
    file=request.files['video']
    requestId=secrets.token_hex(16)

    print(requestId)
    
    folderPath=os.path.join(app.static_folder, 'temps', requestId)
    os.mkdir(folderPath)

    orgFolder=os.path.join(folderPath, 'org')
    os.mkdir(orgFolder)

    filePath=os.path.join(orgFolder, file.filename)
    file.save(filePath)

    # Cropping Faces
    facesFolder=os.path.join(folderPath, 'faces')
    os.mkdir(facesFolder)
    facesCount=crop_facesFromVideo(filePath, facesFolder)
    print('Faces Count: ', facesCount)

    # # Face Recognition
    # students=Student.query.all()
    # rollNos=[student.rollNo for student in students]
    # facesFiles=os.listdir(facesFolder)
    
    # df=pd.DataFrame(columns=rollNos, index=facesFiles)
    # df=df.fillna(-1.0)

    # for rollNo in rollNos:
    #     for faceFile in facesFiles:
    #         anchorImages=os.listdir(os.path.join(app.static_folder, 'uploads', str(rollNo)))
    #         if len(anchorImages) == 0:
    #             continue
            
    #         confidence=0.0
    #         for anchorImage in anchorImages:
    #             currConfidence=predict(os.path.join(app.static_folder, 'uploads', str(rollNo), anchorImage), os.path.join(facesFolder, faceFile))
    #             # confidence+=currConfidence
    #             confidence=max(confidence, currConfidence)
            
    #         # confidence/=len(anchorImages)            
    #         df.loc[faceFile, rollNo]=confidence

    attendanceResult=[]
    # for rollNo in rollNos:
    #     if df[rollNo].max() >= 0.5:
    #         attendanceResult.append({
    #             'rollNo': rollNo,
    #             'status': 'Present',
    #             'confidence': df[rollNo].max()
    #         })
    #     else:
    #         attendanceResult.append({
    #             'rollNo': rollNo,
    #             'status': 'Absent',
    #             'confidence': df[rollNo].max()
    #         })

    # # Delete the requestID folder 
    # shutil.rmtree(folderPath)

    # Return a dummy json
    return attendanceResult

@app.route('/classify', methods=['POST'])
def classify():
    file = request.files['image']
    requestId=secrets.token_hex(16)

    folderPath=os.path.join(app.static_folder, 'temps', requestId)
    os.mkdir(folderPath)

    filePath=os.path.join(folderPath, file.filename)
    file.save(filePath)

    # Cropping Faces
    facesFolder=os.path.join(folderPath, 'faces')
    os.mkdir(facesFolder)

    facesCount=crop_faces(filePath, facesFolder)
    print('Faces Count: ', facesCount)

    # Face Recognition and Classification
    classifier=getClassifier()
    model=load_shuffleFaceNet()
    embeddings=[]
    for face in os.listdir(facesFolder):
        facePath=os.path.join(facesFolder, face)
        embedding=extract_embeddings(facePath, model)
        embeddings.append(embedding)

    probs=classifier.predict_proba(embeddings)
    mx_prob_label=classifier.predict(embeddings)
    labels=classifier.classes_
    results={}
    for i in range(len(mx_prob_label)):
        label_idx=0
        for j in range(len(labels)):
            if labels[j]==mx_prob_label[i]:
                label_idx=j
                break
        
        results[mx_prob_label[i]]=probs[i][label_idx]

    # Delete the requestID folder
    shutil.rmtree(folderPath)

    # print(labels.tolist())
    return {
        'labels': labels.tolist(),
        'results': results,
        'probs': probs.tolist()
    }

app.app_context().push()
db.create_all()

def make_directories():
    if not os.path.exists(os.path.join(app.static_folder, 'temps')):
        os.mkdir(os.path.join(app.static_folder, 'temps'))
    
    if not os.path.exists(os.path.join(app.static_folder, 'uploads')):
        os.mkdir(os.path.join(app.static_folder, 'uploads'))

if __name__ == "__main__":
    make_directories()
    app.run(
        debug=True,
        host='0.0.0.0'
    )   