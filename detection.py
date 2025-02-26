import cv2
import threading
import time
import queue    
from ultralytics import YOLO
from scripts.ambulance import Ambulance

model = YOLO("models/emergency.pt")
 
frame_queue = queue.Queue(maxsize=10)
display_frame_queue = queue.Queue(maxsize=10)
TARGET_WIDTH = 640
TARGET_HEIGHT = 480


class Detection():
    def __init__(self):
        
        # self.video = cv2.VideoCapture("http://192.168.165.141:4747/video")
        self.video = cv2.VideoCapture("test_videos/firetruck.mp4")

        if not self.video.isOpened():
            print("Error: Could not open video stream. Please check your camera URL.")
            return

        self.stop_event = False
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        self.ambulance = 0

    def capture_frames(self, video):

        while not self.stop_event:
            ret, frame = video.read()
            if not ret:
                print("Failed to capture frame from video stream.")
                break
            if frame_queue.full():
                frame_queue.get()
            frame_queue.put(frame)
            print("Captured frame")
            time.sleep(1 / 30)  # Sleep to maintain 30 FPS

    def detect_ambulance(self, frame):

        results = model(frame, conf=0.70)
        annotated_image = frame.copy()  # Create a copy of the original frame for annotation

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)  # Get class ID
                class_name = model.names[class_id]  # Get class name
                confidence = box.conf.item()  # Get confidence score
                if class_name :
                    self.ambulance += 1
                    print(self.ambulance)
                    print(f"Detected: {class_name} (Confidence: {confidence:.2f})")
                    # Draw bounding box and label on the annotated image
                    c1, c2 = (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
                    cv2.rectangle(annotated_image, c1, c2, (0, 255, 0), 2)  # Green bounding box
                    cv2.putText(annotated_image, f'{class_name} {confidence:.2f}', (c1[0], c1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Image", annotated_image)
        if self.ambulance > 35:
            Ambulance(frame=annotated_image)
            self.ambulance = 0
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop_event = True
            print("Stopping...")
        return annotated_image

    def process_frames(self):

        while not self.stop_event:
            if not frame_queue.empty():
                frame = frame_queue.get()
                processed_frame = self.detect_ambulance(frame)
                if display_frame_queue.full():
                    display_frame_queue.get()
                display_frame_queue.put(processed_frame)
                print("Processed frame")

    def main(self):

        # Start threads
        threads = [
            threading.Thread(target=self.capture_frames, args=(self.video,)),
            threading.Thread(target=self.process_frames),
        ]
        
        for t in threads:
            t.start()

        try:
            while not self.stop_event:
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("Stopping...")
            
        finally:
            self.stop_event = True
            for t in threads:
                t.join()
            self.video.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = Detection()
    detector.main()