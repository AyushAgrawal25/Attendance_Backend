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
python3 run.py
```

In case issue of LibGL.so.1: cannot open shared object file: No such file or directory

```bash
sudo apt-get install ffmpeg libsm6 libxext6  -y
```

### 6 - Install gunicorn

```bash
sudo apt-get install gunicorn
```

### 7 - Run the server using gunicorn

```bash
gunicorn -w 3 run:app
```

### 8 - Install supervisor

```bash
sudo apt-get install supervisor
```

### 9 - Create a supervisor config file

```bash
sudo nano /etc/supervisor/conf.d/attendance.conf
```

### 10 - Add the following to the file

```bash
[program:attendance]
command=/home/ayush/Attendance_Backend/env/bin/gunicorn -w 3 run:app
directory=/home/ayush/Attendance_Backend
user=ayush
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stderr_logfile=/var/log/attendance.err.log
stdout_logfile=/var/log/attendance.out.log
```

### 11 - Update supervisor

```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start attendance
```

### 12 - Check the status

```bash
sudo supervisorctl status
```

### 13 - Check the logs

```bash
sudo tail -f /var/log/attendance.err.log
sudo tail -f /var/log/attendance.out.log
```