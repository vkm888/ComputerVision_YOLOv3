# IP Webcam - використав для трасляції з мобільного
# https://play.google.com/store/apps/details?id=com.pas.webcam&hl=ua
# після запуску в локальній мережі доступ в браузері за посиланням, приклад https://192.168.0.164:8080/

# Лічильник: У верхньому лівому куті з'явиться чорна плашка з кількістю визначених об'єктів.
# Детекція миші (mouse): * Якщо в кадрі з'явиться мишка, рамка навколо неї буде червоною, а на екрані з'явиться напис ALARM!.
# Комп'ютер видасть короткий сигнал (Beep).
# Логування: У папці з проектом автоматично створиться файл detections_log.txt. Кожного разу, коли програма бачитиме ціль, вона записуватиме туди точну дату та час.
# YOLOv3-Tiny (Найкращий спосіб) - "легка" версія моделі. Вона працює у 5-10 разів швидше.
# Завантажте файли: yolov3-tiny.weights та yolov3-tiny.cfg
# https://pjreddie.com/media/files/yolov3-tiny.weights
# https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import winsound
import datetime
import os

# --- НАЛАШТУВАННЯ ---
# WEIGHTS = 'yolov3.weights'
# CONFIG = 'yolov3.cfg'
WEIGHTS = 'yolov3-tiny.weights'
CONFIG = 'yolov3-tiny.cfg'
NAMES = 'coco.names'
TARGET_CLASS = "mouse" #"person"  # Спробуйте "person" або "cell phone" mouse для тесту
LOG_FILE = "detections_log.txt"

# Завантаження моделі
net = cv2.dnn.readNet(WEIGHTS, CONFIG)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

with open(NAMES, 'r') as f:
    classes = f.read().splitlines()

# Ініціалізація вікна (якщо воно вже є - закриваємо)
try:
    window.destroy()
except:
    pass

window = tk.Tk()
window.title('YOLO Security System')

canvas = tk.Canvas(window, width=640, height=480)
canvas.pack()

# Відеопотік
adress = "http://192.168.0.164:8080/video" 
cap = cv2.VideoCapture(adress)

# Змінні стану
last_boxes, last_confidences, last_class_ids, last_indexes = [], [], [], []
frame_count = 0

def update_frame():
    global frame_count, last_boxes, last_confidences, last_class_ids, last_indexes
    
    ret, frame = cap.read()
    if not ret:
        window.after(10, update_frame)
        return

    frame_count += 1
    height, width, _ = frame.shape

    # Детекція раз на 3 кадри
    if frame_count % 3 == 0:
        blob = cv2.dnn.blobFromImage(frame, 1/255, (320, 320), (0,0,0), True, False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())
        
        last_boxes, last_confidences, last_class_ids = [], [], []
        for out in outs:
            for det in out:
                scores = det[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                if conf > 0.3:
                    w, h = int(det[2]*width), int(det[3]*height)
                    x, y = int((det[0]*width)-w/2), int((det[1]*height)-h/2)
                    last_boxes.append([x, y, w, h])
                    last_confidences.append(float(conf))
                    last_class_ids.append(class_id)
        last_indexes = cv2.dnn.NMSBoxes(last_boxes, last_confidences, 0.3, 0.4)

        # Перевірка на ціль та звук
        current_labels = [classes[last_class_ids[i]] for i in (last_indexes.flatten() if len(last_indexes)>0 else [])]
        if TARGET_CLASS in current_labels:
            winsound.Beep(1000, 150)
            with open(LOG_FILE, "a") as f:
                f.write(f"[{datetime.datetime.now()}] Виявлено: {TARGET_CLASS}\n")

    # Малювання
    obj_count = 0
    if len(last_indexes) > 0:
        obj_count = len(last_indexes.flatten())
        for i in last_indexes.flatten():
            x, y, w, h = last_boxes[i]
            label = classes[last_class_ids[i]]
            clr = (0, 255, 0) if label != TARGET_CLASS else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), clr, 2)
            cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)

    # Інтерфейс поверх відео
    cv2.rectangle(frame, (0, 0), (180, 40), (0,0,0), -1)
    cv2.putText(frame, f"Count: {obj_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Вивід в Tkinter
    img_rgb = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
    canvas.create_image(0, 0, anchor='nw', image=img_tk)
    canvas.image = img_tk
    
    window.after(1, update_frame)

# Важливо: викликаємо функцію ТА запускаємо цикл
update_frame()
window.mainloop()
cap.release()
