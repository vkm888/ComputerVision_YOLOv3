import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from cap_from_youtube import cap_from_youtube # Спеціальна бібліотека для YouTube
import datetime

# --- НАЛАШТУВАННЯ ---
YOUTUBE_URL = "https://youtu.be/lNXiNakWyCk"
WEIGHTS = 'yolov3-tiny.weights'
CONFIG = 'yolov3-tiny.cfg'
NAMES = 'coco.names'
LOG_FILE = "traffic_stats.txt"

# Класи, які ми хочемо бачити
TARGET_CLASSES = ["car", "truck", "bus", "motorbike", "person"]

# Завантаження YOLO
net = cv2.dnn.readNet(WEIGHTS, CONFIG)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

with open(NAMES, 'r') as f:
    classes = f.read().splitlines()

# Ініціалізація вікна Tkinter
try:
    window.destroy()
except:
    pass

window = tk.Tk()
window.title('YouTube Traffic Analyzer (YOLO Tiny)')

canvas = tk.Canvas(window, width=800, height=450)
canvas.pack(pady=10)

# Підключення до YouTube (беремо 480p для швидкості)
print("Підключаємось до YouTube... зачекайте")
cap = cap_from_youtube(YOUTUBE_URL, '480p') # 720p

last_boxes, last_confidences, last_class_ids, last_indexes = [], [], [], []
frame_count = 0

def update_frame():
    global frame_count, last_boxes, last_confidences, last_class_ids, last_indexes
    
    ret, frame = cap.read()
    if not ret:
        print("Потік перервано або завершено")
        window.after(1000, update_frame)
        return

    frame_count += 1
    h_orig, w_orig, _ = frame.shape

    # Детекція раз на 3 кадри для економії ресурсів
    if frame_count % 3 == 0:
        # 320x320 - баланс між швидкістю та можливістю бачити авто здалеку
        blob = cv2.dnn.blobFromImage(frame, 1/255, (320, 320), (0,0,0), True, False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())
        
        last_boxes, last_confidences, last_class_ids = [], [], []
        for out in outs:
            for det in out:
                scores = det[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                if conf > 0.25: # Трохи нижчий поріг для дрібних авто на дорозі
                    label = classes[class_id]
                    if label in TARGET_CLASSES:
                        w, h = int(det[2]*w_orig), int(det[3]*h_orig)
                        x, y = int((det[0]*w_orig)-w/2), int((det[1]*h_orig)-h/2)
                        last_boxes.append([x, y, w, h])
                        last_confidences.append(float(conf))
                        last_class_ids.append(class_id)
        last_indexes = cv2.dnn.NMSBoxes(last_boxes, last_confidences, 0.3, 0.4)

    # Статистика для поточного кадру
    current_counts = {cls: 0 for cls in TARGET_CLASSES}
    
    if len(last_indexes) > 0:
        for i in last_indexes.flatten():
            x, y, w, h = last_boxes[i]
            label = classes[last_class_ids[i]]
            current_counts[label] += 1
            
            # Кольори для типів авто
            color = (0, 255, 0) # Car - Green 
            if label == "person": color = (80, 43, 229) # person - Амарантовий	Amaranth
            if label == "truck": color = (0, 165, 255) # Truck - Orange
            if label == "bus": color = (255, 0, 255) # Bus - Purple

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Малюємо панель статистики поверх відео
    cv2.rectangle(frame, (0, 0), (180, 110), (0,0,0), -1)
    y_pos = 25
    for obj, count in current_counts.items():
        cv2.putText(frame, f"{obj.capitalize()}: {count}", (15, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y_pos += 25

    # Конвертація для відображення в Tkinter
    # Змінюємо розмір для зручного перегляду
    frame_resized = cv2.resize(frame, (800, 450))
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
    
    canvas.create_image(0, 0, anchor='nw', image=img_tk)
    canvas.image = img_tk
    
    window.after(1, update_frame)

# Запуск
update_frame()
window.mainloop()
cap.release()
