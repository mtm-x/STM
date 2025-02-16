import cv2
from datetime import datetime
import os

def Ambulance(frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detections/ambulance_{timestamp}.jpg"
    if not os.path.exists("detections"):
        os.makedirs("detections")
    cv2.imwrite(filename, frame)
    return filename