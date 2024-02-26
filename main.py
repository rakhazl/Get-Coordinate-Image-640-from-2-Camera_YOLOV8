from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
import time
import os
import threading

model = YOLO("path_result_training_yolov8")
camera0 = cv2.VideoCapture("path_camera0")
camera1 = cv2.VideoCapture("path_camera1")

# Folder to store captures from camera0
capture_folder= "path_folder_camera_camera0"

def draw_label(frame, model, result, camera_number):
    for r in result:
        annotator = Annotator(frame, line_width=1)
        boxes = r.boxes
        result = model(camera0, camera1)
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
                        
            key = model.names[int(c)]
            probs = box.conf[0].item()
            annotator.box_label(b, 
                                model.names[int(c)]+str(round(box.conf[0].item(), 2)),
                                color=colors(c, True))
            
            # Generate file name based on box coordinates
            file_name = f"capture_{int(b[0])}_{int(b[1])}_{int(b[2])}_{int(b[3])}.jpg"
            print(file_name)
            print(box.xyxy)
            # Save the frame with the generated file name
            cv2.imwrite(os.path.join(capture_folder, file_name), frame)

# Function to detect objects and capture if any object is detected
def detect_and_capture(frame, camera_number):
    result = model(frame, verbose=False, conf=0.5, imgsz=640)
    result = model.track(frame, persist=True)
    draw_label(frame, model, result, camera_number)
        
    # If an object is detected, perform a capture
    for r in result:
        if len(r.boxes) > 0:

            capture_path = os.path.join(capture_folder, f"capture_{str(time.time())}.jpg")
            cv2.imwrite(capture_path, frame)
            print("Ada objek terdeteksi")
            break

# Function to generate frames from camera0
def generate_frames0():
    while True:
        success0, frame0 = camera0.read()
        if not success0:
            break

        detect_and_capture(frame0, 0)  

        ret, buffer = cv2.imencode('.jpg', frame0)
        frame0 = buffer.tobytes()      

# Function to generate frames from camera1
def generate_frames1():
    while True:
        success1, frame1 = camera1.read()
        if not success1:
            break

        detect_and_capture(frame1, 1)  

    ret, buffer = cv2.imencode('.jpg', frame1)
    frame1 = buffer.tobytes()

if __name__ == '__main__':
        print("File app.py Running.")
        t1 = threading.Thread(target=generate_frames0)
        t2 = threading.Thread(target=generate_frames1)
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
