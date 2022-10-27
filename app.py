from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pymysql
import secrets
import os
from utils.faceDetection import crop_faces

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
    file=request.files['image']
    file.save(os.path.join(app.static_folder, 'uploads', rollNo, file.filename))
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

    # Face Recognition
    # TODO: use the predict function to detect face and then create a dataframe for the detected faces
    # Where rows will be represent the faces found and column will be the student rollNo
    # The value will be the confidence of the face found
    # Using this dataframe get the attendance.

    # # Delete Temp Folder
    # os.remove(folderPath)

    # Return a dummy json
    return [{
        'rollNo': '180101001',
        'name': 'Dummy Student',
        'branch': 'CSE',
        'semester': 1,
        'anchor_link': '/static/uploads/180101001/'
    }]

app.app_context().push()
db.create_all()

def make_directories():
    if not os.path.exists(os.path.join(app.static_folder, 'temps')):
        os.mkdir(os.path.join(app.static_folder, 'temps'))
    
    if not os.path.exists(os.path.join(app.static_folder, 'uploads')):
        os.mkdir(os.path.join(app.static_folder, 'uploads'))

if __name__ == "__main__":
    make_directories()
    app.run(debug=True)