import cv2
import threading
import time
import queue
from ultralytics import YOLO
from scripts.ambulance import Ambulance

model = YOLO("models/best.pt")
FRAME_SKIP = 2
frame_queue = queue.Queue(maxsize=2)
display_frame_queue = queue.Queue(maxsize=2)
TARGET_WIDTH = 640
TARGET_HEIGHT = 480

class detection():

    def __init__(self):
        self.video = cv2.VideoCapture("test_videos/ambulance.mp4")
        print(model.names)
        self.ambulance = 0
        self.stop_event = threading.Event()

    def capture_frames(self,video):
        frame_count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % FRAME_SKIP == 0:
                if frame_queue.full():
                    frame_queue.get()
                frame_queue.put(frame)
            else:
                print("Failed to get frame from camera. Retrying...")
                time.sleep(0.1)
                
    def detect_ambulance(self,frame):
            results = model(frame, conf = 0.75)
                # annotated_image = image.copy()  # Create a copy of the original frame for annotation

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)  # Get class ID
                    class_name = model.names[class_id]  # Get class name
                    confidence = box.conf.item()  # Get confidence score
                    if class_name :
                            self.ambulance += 1
                            print(self.ambulance)
                            print(f"Detected: {class_name} (Confidence: {confidence:.2f})")
                            annotated_image = results[0].plot()
                            # # Plot the bounding box and label on the annotated image
                            # c1, c2 = (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3]))
                            # cv2.rectangle(annotated_image, c1, c2, (0, 255, 0), 2)  # Green bounding box
                            # cv2.putText(annotated_image, f'{class_name} {confidence:.2f}', (c1[0], c1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                cv2.imshow("Image", annotated_image)
            # if self.ambulance > 25:
            #     Ambulance()
            #     break
            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
            return frame

    def process_frames(self,):
        while not self.stop_event.is_set():
            if not frame_queue.empty():
                frame = frame_queue.get()
                processed_frame = self.detect_ambulance(frame)
                if display_frame_queue.full():
                    display_frame_queue.get()
                display_frame_queue.put(processed_frame)
            

    def main(self):

        video = cv2.VideoCapture("test_videos/ambulance.mp4")

        if not video.isOpened():
            print("Error: Could not open video stream. Please check your camera URL.")
            return

        # video.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
        # video.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

        # Start threads
        threads = [
            threading.Thread(target=self.capture_frames, args=(video,)),
            threading.Thread(target=self.process_frames),
        ]
        
        for t in threads:
            t.start()

        try:
            while not self.stop_event.is_set():
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("Stopping...")
            
        finally:
            self.stop_event.set()
            for t in threads:
                t.join()
            self.video.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = detection()
    detector.main()