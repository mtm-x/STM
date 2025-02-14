from ultralytics import YOLO
import cv2
from scripts.ambulance import Ambulance

model = YOLO("models/yolo11n.pt")


#classes
"""
{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
"""
class detection():

    def __init__(self):
        self.video = cv2.VideoCapture("test_videos/ambulance.mp4")
        print(model.names)
        self.ambulance = 0

    def detect(self):
        
        while True:
            ret, image = self.video.read()
            if not ret:
                break

            results = model(image, classes =[7]) #only detects the cars 
            # annotated_image = image.copy()  # Create a copy of the original frame for annotation

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)  # Get class ID
                    class_name = model.names[class_id]  # Get class name
                    confidence = box.conf.item()  # Get confidence score
                    if class_name in ["truck"]:
                        self.ambulance += 1
                        print(self.ambulance)
                        print(f"Detected: {class_name} (Confidence: {confidence:.2f})")
                        annotated_image = results[0].plot()
                        # # Plot the bounding box and label on the annotated image
                        # c1, c2 = (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
                        # cv2.rectangle(annotated_image, c1, c2, (0, 255, 0), 2)  # Green bounding box
                        # cv2.putText(annotated_image, f'{class_name} {confidence:.2f}', (c1[0], c1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                cv2.imshow("Image", annotated_image)
            if self.ambulance > 25:
                Ambulance()
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = detection()
    detector.detect()