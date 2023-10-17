from fileinput import filename
from xml.dom.minidom import Document
from flask import Flask,  render_template, request, redirect, url_for, session , Response
from flask_mysqldb import MySQL,MySQLdb 
from os import path 
from notifypy import Notify
import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
from datetime import datetime
from sqlalchemy import case



app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'usuarios'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

@app.route('/')
def home():
    return render_template("contenido.html")  
def generate():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    up = False
    down = False
    global count1,count2,count3
    count1 = 0
    count2=0
    count3=0
    with mp_pose.Pose(static_image_mode=False) as pose:
        while count1<int(name_of_slider):
            ret, frame = cap.read()
            if ret == False:
                break
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks is not None:
                x1 = int(results.pose_landmarks.landmark[11].x * width)
                y1 = int(results.pose_landmarks.landmark[11].y * height)
                x2 = int(results.pose_landmarks.landmark[13].x * width)
                y2 = int(results.pose_landmarks.landmark[13].y * height)
                x3 = int(results.pose_landmarks.landmark[15].x * width)
                y3 = int(results.pose_landmarks.landmark[15].y * height)
                x4 = int(results.pose_landmarks.landmark[12].x * width)
                y4 = int(results.pose_landmarks.landmark[12].y * height)
                x5 = int(results.pose_landmarks.landmark[14].x * width)
                y5 = int(results.pose_landmarks.landmark[14].y * height)
                x6 = int(results.pose_landmarks.landmark[16].x * width)
                y6 = int(results.pose_landmarks.landmark[16].y * height)
                p1 = np.array([x1, y1])
                p2 = np.array([x2, y2])
                p3 = np.array([x3, y3])
                p4 = np.array([x4, y4])
                p5 = np.array([x5, y5])
                p6 = np.array([x6, y6])
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)
                l4 = np.linalg.norm(p5 - p6)
                l5 = np.linalg.norm(p4 - p6)
                l6 = np.linalg.norm(p4 - p5)
                angleLeft = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                angleRigth=degrees(acos((l4**2 + l6**2 - l5**2) / (2 * l4 * l6)))
                bar = np.interp(angleRigth, (150, 70), (450, 100))
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.line(frame, (x2, y2), (x3, y3), (255, 255, 255), 3)   
                cv2.line(frame, (x4, y4), (x5, y5), (255, 255, 255), 3)
                cv2.line(frame, (x5, y5), (x6, y6), (255, 255, 255), 3)
                if angleRigth >= 150 and angleLeft<=70:
                    up = True
                if up == True and angleRigth <= 70 :
                    down = True
                if up == True and down == True and angleRigth<70:
                    count1 += 1
                    up = False
                    down = False
                frame = cv2.flip(frame,1)
                cv2.rectangle(frame, (100, 100), (117, 450), (120,120,0), 3)
                cv2.rectangle(frame, (100, int(bar)), (117, 450), (120,0,120), cv2.FILLED)
                cv2.putText(frame,"ejercicio 1:" +str(count1)+"/"+name_of_slider, (10, 50), 1, 3.5, (128, 0, 250), 2)
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if not flag:
                    continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encodedImage) + b'\r\n')
        
        while count2<int(name_of_slider2):
            ret, frame = cap.read()
            if ret == False:
                break
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks is not None:
                x1 = int(results.pose_landmarks.landmark[23].x * width)
                y1 = int(results.pose_landmarks.landmark[23].y * height)
                x2 = int(results.pose_landmarks.landmark[11].x * width)
                y2 = int(results.pose_landmarks.landmark[11].y * height)
                x3 = int(results.pose_landmarks.landmark[15].x * width)
                y3 = int(results.pose_landmarks.landmark[15].y * height)
                x4 = int(results.pose_landmarks.landmark[24].x * width)
                y4 = int(results.pose_landmarks.landmark[24].y * height)
                x5 = int(results.pose_landmarks.landmark[12].x * width)
                y5 = int(results.pose_landmarks.landmark[12].y * height)
                x6 = int(results.pose_landmarks.landmark[16].x * width)
                y6 = int(results.pose_landmarks.landmark[16].y * height)
                p1 = np.array([x1, y1])
                p2 = np.array([x2, y2])
                p3 = np.array([x3, y3])
                p4 = np.array([x4, y4])
                p5 = np.array([x5, y5])
                p6 = np.array([x6, y6])
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)
                l4 = np.linalg.norm(p5 - p6)
                l5 = np.linalg.norm(p4 - p6)
                l6 = np.linalg.norm(p4 - p5)
                
                angleLeft = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                angleRigth=degrees(acos((l4**2 + l6**2 - l5**2) / (2 * l4 * l6)))
                bar = np.interp(angleLeft, (40, 150), (100, 450))
                if angleRigth >= 150 and angleLeft<=40:
                    up = True
                if up == True and angleLeft >= 150 :
                    down = True
                if up == True and down == True and angleRigth< 40:
                    count2 += 1
                    up = False
                    down = False
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.line(frame, (x2, y2), (x3, y3), (255, 255, 255), 3)   
                cv2.line(frame, (x4, y4), (x5, y5), (255, 255, 255), 3)
                cv2.line(frame, (x5, y5), (x6, y6), (255, 255, 255), 3)
                frame = cv2.flip(frame,1)
                cv2.rectangle(frame, (100, 100), (117, 450), (120,120,0), 3)
                cv2.rectangle(frame, (100, int(bar)), (117, 450), (120,0,120), cv2.FILLED)
                cv2.putText(frame,"ejercicio 2:" +str(count2)+"/"+name_of_slider2, (10, 50), 1, 3.5, (128, 0, 250), 2)
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if not flag:
                    continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encodedImage) + b'\r\n')  
        while count3<int(name_of_slider3):
            ret, frame = cap.read()
            if ret == False:
                break
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks is not None:
                x1 = int(results.pose_landmarks.landmark[23].x * width)
                y1 = int(results.pose_landmarks.landmark[23].y * height)
                x2 = int(results.pose_landmarks.landmark[11].x * width)
                y2 = int(results.pose_landmarks.landmark[11].y * height)
                x3 = int(results.pose_landmarks.landmark[15].x * width)
                y3 = int(results.pose_landmarks.landmark[15].y * height)
                x4 = int(results.pose_landmarks.landmark[24].x * width)
                y4 = int(results.pose_landmarks.landmark[24].y * height)
                x5 = int(results.pose_landmarks.landmark[12].x * width)
                y5 = int(results.pose_landmarks.landmark[12].y * height)
                x6 = int(results.pose_landmarks.landmark[16].x * width)
                y6 = int(results.pose_landmarks.landmark[16].y * height)
                p1 = np.array([x1, y1])
                p2 = np.array([x2, y2])
                p3 = np.array([x3, y3])
                p4 = np.array([x4, y4])
                p5 = np.array([x5, y5])
                p6 = np.array([x6, y6])
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)
                l4 = np.linalg.norm(p5 - p6)
                l5 = np.linalg.norm(p4 - p6)
                l6 = np.linalg.norm(p4 - p5)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.line(frame, (x2, y2), (x3, y3), (255, 255, 255), 3)   
                cv2.line(frame, (x4, y4), (x5, y5), (255, 255, 255), 3)
                cv2.line(frame, (x5, y5), (x6, y6), (255, 255, 255), 3)
                angleLeft = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                angleRigth=degrees(acos((l4**2 + l6**2 - l5**2) / (2 * l4 * l6)))
                bar = np.interp(angleLeft, (30, 150), (100, 450))
                if angleRigth > 150 and angleLeft>150:
                    up = True
                if up == True and angleRigth <= 30:
                    down = True
                if up == True and down == True:
                    count3 += 1
                    up = False
                    down = False
                frame = cv2.flip(frame,1)
                cv2.putText(frame,"ejercicio 3:" +str(count3)+"/"+name_of_slider3, (10, 50), 1, 3.5, (128, 0, 250), 2)
                cv2.rectangle(frame, (100, 100), (117, 450), (120,120,0), 3)
                cv2.rectangle(frame, (100, int(bar)), (117, 450), (120,0,120), cv2.FILLED)
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if not flag:
                    continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encodedImage) + b'\r\n')

    with app.app_context():
        notificacion=Notify()
        notificacion.title = "Rutina concluida"
        notificacion.message="las repeticiones se concluyeron con exito'presione volver"
        notificacion.send()
def generate1():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    up = False
    down = False
    global count4,count5, count6
    count4 = 0
    count5=0
    count6=0
    with mp_pose.Pose(static_image_mode=False) as pose:
        while count4<int(slider1):
            ret, frame = cap.read()
            if ret == False:
                break
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks is not None:
                x1 = int(results.pose_landmarks.landmark[24].x * width)
                y1 = int(results.pose_landmarks.landmark[24].y * height)
                x2 = int(results.pose_landmarks.landmark[26].x * width)
                y2 = int(results.pose_landmarks.landmark[26].y * height)
                x3 = int(results.pose_landmarks.landmark[28].x * width)
                y3 = int(results.pose_landmarks.landmark[28].y * height)
                p1 = np.array([x1, y1])
                p2 = np.array([x2, y2])
                p3 = np.array([x3, y3])
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)
                #cv2.circle(frame, (x1, y1), 6, (0, 255, 255), 4)
                #cv2.circle(frame, (x2, y2), 6, (128, 0, 250), 4)
                #cv2.circle(frame, (x3, y3), 6, (255, 191, 0), 4)
                # Calcular el ángulo
                angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                bar = np.interp(angle, (100, 160), (100, 450))
                if angle >= 160:
                    up = True
                if up == True and down == False and angle <= 110:
                    down = True
                if up == True and down == True and angle >= 160:
                    count4 += 1
                    up = False
                    down = False
                
                frame = cv2.flip(frame,1)
                cv2.putText(frame,"ejercicio 1:" +str(count4)+"/"+slider1, (10, 50), 1, 3.5, (128, 0, 250), 2)
                cv2.rectangle(frame, (100, 100), (117, 450), (120,120,0), 3)
                cv2.rectangle(frame, (100, int(bar)), (117, 450), (120,0,120), cv2.FILLED)
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if not flag:
                    continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encodedImage) + b'\r\n')
         
        while count5<int(slider2):
            ret, frame = cap.read()
            if ret == False:
                break
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks is not None:
                x1 = int(results.pose_landmarks.landmark[23].x * width)
                y1 = int(results.pose_landmarks.landmark[23].y * height)
                x2 = int(results.pose_landmarks.landmark[25].x * width)
                y2 = int(results.pose_landmarks.landmark[25].y * height)
                x3 = int(results.pose_landmarks.landmark[27].x * width)
                y3 = int(results.pose_landmarks.landmark[27].y * height)
                x4 = int(results.pose_landmarks.landmark[24].x * width)
                y4 = int(results.pose_landmarks.landmark[24].y * height)
                x5 = int(results.pose_landmarks.landmark[26].x * width)
                y5 = int(results.pose_landmarks.landmark[26].y * height)
                x6 = int(results.pose_landmarks.landmark[28].x * width)
                y6 = int(results.pose_landmarks.landmark[28].y * height)

                p1 = np.array([x1, y1])
                p2 = np.array([x2, y2])
                p3 = np.array([x3, y3])
                p4 = np.array([x4, y4])
                p5 = np.array([x5, y5])
                p6 = np.array([x6, y6])
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)
                l4 = np.linalg.norm(p5 - p6)
                l5 = np.linalg.norm(p4 - p6)
                l6 = np.linalg.norm(p4 - p5)
                #cv2.circle(frame, (x1, y1), 6, (0, 255, 255), 4)
                #cv2.circle(frame, (x2, y2), 6, (128, 0, 250), 4)
                #cv2.circle(frame, (x3, y3), 6, (255, 191, 0), 4)
                angleLeft = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                angleRigth=degrees(acos((l4**2 + l6**2 - l5**2) / (2 * l4 * l6)))
                bar = np.interp(angleLeft, (100, 180), (100, 450))
                if  angleLeft>=160 and angleRigth<100:
                        up = True
                if up == True and angleLeft<= 100 :
                        down = True
                if up == True and down == True:
                        count5 += 1
                        up = False
                        down = False
                
                frame = cv2.flip(frame,1)
                cv2.putText(frame,"ejercicio 2:" +str(count5)+"/"+slider2, (10, 50), 1, 3.5, (128, 0, 250), 2)
                cv2.rectangle(frame, (100, 100), (117, 450), (120,120,0), 3)
                cv2.rectangle(frame, (100, int(bar)), (117, 450), (120,0,120), cv2.FILLED)
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if not flag:
                    continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encodedImage) + b'\r\n')  
        while count6<int(slider3):
            ret, frame = cap.read()
            if ret == False:
                break
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks is not None:
                x1 = int(results.pose_landmarks.landmark[29].x * width)
                y1 = int(results.pose_landmarks.landmark[29].y * height)
                x2 = int(results.pose_landmarks.landmark[24].x * width)
                y2 = int(results.pose_landmarks.landmark[24].y * height)
                x3 = int(results.pose_landmarks.landmark[30].x * width)
                y3 = int(results.pose_landmarks.landmark[30].y * height)
                p1 = np.array([x1, y1]) 
                p2 = np.array([x2, y2])
                p3 = np.array([x3, y3])
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)
                # Calcular el ángulo
                angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                bar = np.interp(angle, (5, 40), (450, 100))
                if angle <= 5:
                    up = True
                if up == True and down == False and angle >= 40:
                    down = True
                if up == True and down == True :
                    count6 += 1
                    up = False
                    down = False
                
                frame = cv2.flip(frame,1)
                cv2.putText(frame,"ejercicio 3:" +str(count6)+"/"+slider3, (10, 50), 1, 3.5, (128, 0, 250), 2)
                cv2.rectangle(frame, (100, 100), (117, 450), (120,120,0), 3)
                cv2.rectangle(frame, (100, int(bar)), (117, 450), (120,0,120), cv2.FILLED)
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if not flag:
                    continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encodedImage) + b'\r\n')  
    with app.app_context():
        notificacion=Notify()
        notificacion.title = "Rutina concluida"
        notificacion.message="las repeticiones se concluyeron con exito'presione volver"
        notificacion.send()

def generate2():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    up = False
    down = False
    global count7,count8, count9, count10
    count7 = 0
    count8=0
    count9=0
    count10=0
    with mp_pose.Pose(static_image_mode=False) as pose:
        
        while count7<int(slider4):
            ret, frame = cap.read()
            if ret == False:
                break
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks is not None:
                x1 = int(results.pose_landmarks.landmark[0].x * width)
                y1 = int(results.pose_landmarks.landmark[0].y * height)
                x2 = int(results.pose_landmarks.landmark[11].x * width)
                y2 = int(results.pose_landmarks.landmark[11].y * height)
                x3 = int(results.pose_landmarks.landmark[12].x * width)
                y3 = int(results.pose_landmarks.landmark[12].y * height)
                p1 = np.array([x1, y1])
                p2 = np.array([x2, y2])
                p3 = np.array([x3, y3])
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)
                angleLeft = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                if  angleLeft<=50:
                    up = True
                if up == True and angleLeft>=60:
                    down = True
                if up == True and down == True:
                    count7 += 1
                    up = False
                    down = False
                
                frame = cv2.flip(frame,1)
                cv2.putText(frame, str(int(angleLeft)), (510, 470), 1, 1.5, (0, 0, 255), 2)
                cv2.putText(frame,"ejercicio 1:" +str(count7)+"/"+slider4, (10, 50), 1, 3.5, (128, 0, 250), 2)
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if not flag:
                    continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encodedImage) + b'\r\n')  
            
        while count8<int(slider4):
            ret, frame = cap.read()
            if ret == False:
                break
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks is not None:
                x1 = int(results.pose_landmarks.landmark[0].x * width)
                y1 = int(results.pose_landmarks.landmark[0].y * height)
                x2 = int(results.pose_landmarks.landmark[11].x * width)
                y2 = int(results.pose_landmarks.landmark[11].y * height)
                x3 = int(results.pose_landmarks.landmark[12].x * width)
                y3 = int(results.pose_landmarks.landmark[12].y * height)
                p1 = np.array([x1, y1])
                p2 = np.array([x2, y2])
                p3 = np.array([x3, y3])
                
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)
                angleLeft = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                
                if  angleLeft<=40:
                    up = True
                if up == True and angleLeft>=60:
                    down = True
                if up == True and down == True:
                    count8 += 1
                    up = False
                    down = False
                
                frame = cv2.flip(frame,1)
                cv2.putText(frame, str(int(angleLeft)), (510, 470), 1, 1.5, (0, 0, 255), 2)
                cv2.putText(frame,"ejercicio 2:" +str(count8)+"/"+slider4, (10, 50), 1, 3.5, (128, 0, 250), 2)
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if not flag:
                    continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encodedImage) + b'\r\n')  
        while count9<int(slider5):
            ret, frame = cap.read()
            if ret == False:
                break
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks is not None:
                x1 = int(results.pose_landmarks.landmark[11].x * width)
                y1 = int(results.pose_landmarks.landmark[11].y * height)
                x2 = int(results.pose_landmarks.landmark[0].x * width)
                y2 = int(results.pose_landmarks.landmark[0].y * height)
                x3 = int(results.pose_landmarks.landmark[12].x * width)
                y3 = int(results.pose_landmarks.landmark[12].y * height)

                p1 = np.array([x1, y1])
                p2 = np.array([x2, y2])
                p3 = np.array([x3, y3])
                
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)
                angleLeft = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                if  angleLeft<=75:
                    up = True
                if up == True and angleLeft>=90:
                    down = True
                if up == True and down == True:
                    count9 += 1
                    up = False
                    down = False
             
                
                frame = cv2.flip(frame,1)
                cv2.putText(frame, str(int(angleLeft)), (510, 470), 1, 1.5, (0, 0, 255), 2)
                cv2.putText(frame,"ejercicio 3:" +str(count9)+"/"+slider5, (10, 50), 1, 3.5, (128, 0, 250), 2)
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if not flag:
                    continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encodedImage) + b'\r\n')
        while count10<int(slider6):
            ret, frame = cap.read()
            if ret == False:
                break
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks is not None:
                x1 = int(results.pose_landmarks.landmark[11].x * width)
                y1 = int(results.pose_landmarks.landmark[11].y * height)
                x2 = int(results.pose_landmarks.landmark[0].x * width)
                y2 = int(results.pose_landmarks.landmark[0].y * height)
                x3 = int(results.pose_landmarks.landmark[12].x * width)
                y3 = int(results.pose_landmarks.landmark[12].y * height)
                p1 = np.array([x1, y1])
                p2 = np.array([x2, y2])
                p3 = np.array([x3, y3])
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)
                angleLeft = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))

                if  angleLeft<=83:
                    up = True
                if up == True and angleLeft>=90:
                    down = True
                if up == True and down == True:
                    count10 += 1
                    up = False
                    down = False
                
                frame = cv2.flip(frame,1)
                cv2.putText(frame, str(int(angleLeft)), (510, 470), 1, 1.5, (0, 0, 255), 2)
                cv2.putText(frame,"ejercicio 4:" +str(count10)+"/"+slider6, (10, 50), 1, 3.5, (128, 0, 250), 2)
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if not flag:
                    continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encodedImage) + b'\r\n')
            

        with app.app_context():
            notificacion=Notify()
            notificacion.title = "Rutina concluida"
            notificacion.message="las repeticiones se concluyeron con exito'presione volver"
            notificacion.send()
           
@app.route("/video_feed")
def video_feed():
     return Response(generate(),
          mimetype = "multipart/x-mixed-replace; boundary=frame")
@app.route("/video_feed1")
def video_feed1():
     return Response(generate1(),
          mimetype = "multipart/x-mixed-replace; boundary=frame")
@app.route("/video_feed2")
def video_feed2():
     return Response(generate2(),
          mimetype = "multipart/x-mixed-replace; boundary=frame")
def layout():
    session.clear()
    return render_template("contenido.html")
@app.route('/login', methods= ["GET", "POST"])
def login():

    notificacion = Notify()
    if request.method == 'POST':
        global username
        username = request.form['username']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM usuarios_pruebas WHERE username=%s",(username,))
        user = cur.fetchone()
      

        if len(user)>=0:
            if password == user["password"]:
                session['name'] = user['username']
                session['email'] = user['fullname']
                session['tipo'] = user['tipo_de_usu']

                if session['tipo'] == 1:
                    cur = mysql.connection.cursor()
                    cur.execute("SELECT * FROM usuarios_pruebas")
                    data=cur.fetchall()
                    return render_template("administrador/home.html",datas=data)
                elif session['tipo'] == 2:
                    cur = mysql.connection.cursor()
                    cur.execute("SELECT * FROM repeticiones WHERE usuario=%s", (username,))
                    data=cur.fetchall()
                    cur.execute("SELECT * FROM repeticiones3 WHERE usuario=%s", (username,))
                    data1=cur.fetchall()
                    cur.execute("SELECT * FROM repeticiones2 WHERE usuario=%s", (username,))
                    data2=cur.fetchall()
                    return render_template("estandar/principal.html", contacts=data,contacts1=data1,contacts2=data2,username=username)


            else:
                notificacion.title = "Error de Acceso"
                notificacion.message="nombre de usuario o contraseña no valida"
                notificacion.send()
                return render_template("login.html")
        else:
            notificacion.title = "Error de Acceso"
            notificacion.message="No existe el usuario"
            notificacion.send()
            return render_template("login.html")
    else:
        
        return render_template("login.html")

@app.route("/test", methods=["POST"])
def test():
    global name_of_slider,name_of_slider2,name_of_slider3
    name_of_slider = request.form["name_of_slider"]
    name_of_slider2 = request.form["name_of_slider2"]
    name_of_slider3 = request.form["name_of_slider3"]
    return render_template("estandar/hometwo.html",name_of_slider=name_of_slider,username=username,name_of_slider2=name_of_slider2,name_of_slider3=name_of_slider3)
@app.route("/test1", methods=["POST"])
def test1():    
    global slider1,slider2,slider3
    slider1 = request.form["slider"]
    slider2= request.form["slider2"]
    slider3 = request.form["slider3"]
    return render_template("estandar/hometwo1.html",slider1=slider1,username=username,slider2=slider2,slider3=slider3)
@app.route("/test2", methods=["POST"])
def test2():    
    global slider4,slider5,slider6
    slider4 = request.form["slider4"]
    slider5= request.form["slider5"]
    slider6 = request.form["slider6"]
    return render_template("estandar/hometwo2.html",slider4=slider4,username=username,slider6=slider6,slider5=slider5)
@app.route('/registro', methods = ["GET", "POST"])
def registro():
    notificacion = Notify()
    if request.method == 'GET':
        return render_template("registro.html")
    
    else:
        name = request.form['name']
        fullname=request.form['fullname']
        password = request.form['password']
        tip = 2
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO usuarios_pruebas (id, username,password, fullname,tipo_de_usu) VALUES (%s,%s,%s,%s,%s)", ('',name, password, fullname,tip))
        mysql.connection.commit()
        notificacion.title = "Registro Exitoso"
        notificacion.message="El usuario ya existe por favor inicia sesión"
        notificacion.send()
        return redirect(url_for('login'))
@app.route('/log_out')
def log_out():
    return redirect(url_for('home'))
@app.route('/seleccionar')
def seleccionar():
    username
    return render_template("estandar/selector.html",username=username)
@app.route('/seleccionar1')
def seleccionar1():
    username
    return render_template("estandar/selector1.html",username=username)
@app.route('/seleccionar2')
def seleccionar2():
    username
    return render_template("estandar/selector2.html",username=username)
@app.route('/volver')
def volver():
    fecha=datetime.now()
    rep1=count1
    rep2=count2
    rep3=count3
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO repeticiones (usuario,fecha,ej1,ej2,ej3)  VALUES  (%s,%s,%s,%s,%s)",(username ,fecha, rep1, rep2,rep3))
    mysql.connection.commit()
    cur.execute("SELECT * FROM repeticiones WHERE usuario=%s", (username,))
    data=cur.fetchall()
    cur.execute("SELECT * FROM repeticiones3 WHERE usuario=%s", (username,))
    data1=cur.fetchall()
    cur.execute("SELECT * FROM repeticiones2 WHERE usuario=%s", (username,))
    data2=cur.fetchall()
    return render_template("estandar/principal.html", contacts=data,contacts1=data1,contacts2=data2,username=username)
@app.route('/volver1')
def volver1():
    fecha=datetime.now()
    rep4=count4
    rep5=count5
    rep6=count6
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO repeticiones3 (usuario,fecha,ej1,ej2,ej3)  VALUES  (%s,%s,%s,%s,%s)",(username ,fecha, rep4, rep5,rep6))
    mysql.connection.commit()
    cur.execute("SELECT * FROM repeticiones WHERE usuario=%s", (username,))
    data=cur.fetchall()
    cur.execute("SELECT * FROM repeticiones3 WHERE usuario=%s", (username,))
    data1=cur.fetchall()
    cur.execute("SELECT * FROM repeticiones2 WHERE usuario=%s", (username,))
    data2=cur.fetchall()
    return render_template("estandar/principal.html", contacts=data,contacts1=data1,contacts2=data2,username=username)
@app.route('/volver2')
def volver2():
    fecha=datetime.now()
    rep7=count7
    rep8=count8
    rep9=count9
    rep10=count10
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO repeticiones2 (usuario,fecha,ej1,ej2,ej3,ej4)  VALUES  (%s,%s,%s,%s,%s,%s)",(username ,fecha, rep7, rep8,rep9,rep10))
    mysql.connection.commit()
    cur.execute("SELECT * FROM repeticiones WHERE usuario=%s", (username,))
    data=cur.fetchall()
    cur.execute("SELECT * FROM repeticiones3 WHERE usuario=%s", (username,))
    data1=cur.fetchall()
    cur.execute("SELECT * FROM repeticiones2 WHERE usuario=%s", (username,))
    data2=cur.fetchall()
    return render_template("estandar/principal.html", contacts=data,contacts1=data1,contacts2=data2,username=username)


if __name__ == '__main__':
    app.secret_key = "pinchellave"
    app.run(debug=False)

