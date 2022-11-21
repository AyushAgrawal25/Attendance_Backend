# Attendance Backend

## Installation

### Requirements

-1 Python 3.6
-2 Pip

### 1 - Clone the repository

```bash
git clone https://github.com/AyushAgrawal25/Attendance_Backend.git
```

### 2 - Create a virtual environment

```bash
cd Attendance_Backend
sudo pip3 install virtualenv
sudo virtualenv env
```

### 3 - Activate the virtual environment

```bash
source env/bin/activate
```

### 4 - Install the requirements

```bash
pip3 install -r requirements.txt
```

In case installation got killed try using --no-cache-dir

```bash
pip3 install --no-cache-dir -r requirements.txt
```

### 5 - Run the server

```bash
python3 app.py
```

In case issue of LibGL.so.1: cannot open shared object file: No such file or directory

```bash
sudo apt-get install ffmpeg libsm6 libxext6  -y
```
