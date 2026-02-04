import cv2
import time
from datetime import datetime
import threading
import winsound
import os
import requests
from dotenv import load_dotenv

# --- LOAD SECRETS ---
load_dotenv() 

# Fetch the secret from the .env file
NTFY_TOPIC = os.getenv("NTFY_TOPIC")

if not NTFY_TOPIC:
    print("---------------------------------------------------")
    print("CRITICAL ERROR: .env file not found or NTFY_TOPIC is missing!")
    print("Please create a .env file with: NTFY_TOPIC=your_secret_topic")
    print("---------------------------------------------------")
    exit()

# --- CONFIGURATION ---
SHOW_VIDEO_FEED = True         # Set to False to run headless (saves more CPU)
MIN_AREA_SIZE = 1000           # Minimum size of motion to trigger alert
RECORD_EXTENSION = 3           # Seconds to continue recording after motion stops
LIGHT_CHANGE_THRESHOLD = 40.0  # Percentage of screen change to trigger light suppression
LEARNING_RATE = 0.05           # Speed at which the background model adapts
TARGET_WIDTH = 500             # Width for optimization processing

class SurveillanceSystem:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        # Give the camera time to warm up
        time.sleep(2.0) 
        
        # Background model using weighted averages
        self.avg_frame = None 
        
        self.recording = False
        self.out = None
        self.last_motion_time = None
        
        # Rate limit alerts so you don't get spammed
        self.last_alert_time = 0 
        self.alert_cooldown = 30 
        
        print(f"System Armed.")
        print(f"Notifications: ntfy.sh/{NTFY_TOPIC}")
        print(f"Optimized Mode: {'ON' if not SHOW_VIDEO_FEED else 'OFF'}")

    def alert_user_local(self):
        """Triggers a non-blocking beep alert."""
        def sound_alarm():
            try:
                winsound.Beep(2500, 1000)
            except Exception:
                pass 
        
        t = threading.Thread(target=sound_alarm)
        t.daemon = True
        t.start()

    def send_ntfy_alert(self, frame):
        """Encodes the frame as a JPG and sends it to phone via NTFY."""
        current_time = time.time()
        
        if current_time - self.last_alert_time < self.alert_cooldown:
            return

        self.last_alert_time = current_time

        def _worker():
            try:
                _, img_encoded = cv2.imencode('.jpg', frame)
                data = img_encoded.tobytes()

                print(f"[ALERT] Sending notification to ntfy.sh/{NTFY_TOPIC}...")
                response = requests.put(
                    f"https://ntfy.sh/{NTFY_TOPIC}",
                    data=data,
                    headers={
                        "Title": "Intruder Detected!",
                        "Priority": "high",
                        "Tags": "warning,camera"
                    }
                )
                if response.status_code == 200:
                    print("[ALERT] Notification sent successfully!")
                else:
                    print(f"[ALERT] Failed to send: {response.text}")
            except Exception as e:
                print(f"[ALERT] Error sending notification: {e}")

        t = threading.Thread(target=_worker)
        t.daemon = True
        t.start()

    def start_recording(self, frame):
        """Initializes the VideoWriter and starts saving frames."""
        if not self.recording:
            self.recording = True
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            if not os.path.exists("recordings"):
                os.makedirs("recordings")
            
            filename = os.path.join("recordings", f"Intruder_{timestamp}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width = frame.shape[:2]
            self.out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
            
            print(f"[REC] Started recording: {filename}")
            
            self.alert_user_local()
            self.send_ntfy_alert(frame)

    def stop_recording(self):
        """Releases the VideoWriter and stops recording."""
        if self.recording:
            self.recording = False
            if self.out:
                self.out.release()
                self.out = None
            print("[STOP] Recording saved.")

    def run(self):
        try:
            while True:
                check, frame = self.video.read()
                if not check:
                    print("[ERROR] Could not read from webcam.")
                    break

                # --- OPTIMIZATION: DOWNSCALING ---
                # Calculations on a smaller frame to save CPU
                height, width = frame.shape[:2]
                scale_ratio = width / float(TARGET_WIDTH)
                target_height = int(height / scale_ratio)
                
                small_frame = cv2.resize(frame, (TARGET_WIDTH, target_height))

                # Process the SMALL frame
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if self.avg_frame is None:
                    print("[INFO] Starting background model...")
                    self.avg_frame = gray.astype("float")
                    continue

                # 1. Update background model
                cv2.accumulateWeighted(gray, self.avg_frame, LEARNING_RATE)
                frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg_frame))

                # 2. Thresholding and Light Suppression
                thresh_full = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                changed_pixels = cv2.countNonZero(thresh_full)
                total_pixels = small_frame.shape[0] * small_frame.shape[1]
                change_percentage = (changed_pixels / total_pixels) * 100

                if change_percentage > LIGHT_CHANGE_THRESHOLD:
                    print(f"[INFO] Light change ({change_percentage:.1f}%). Resetting.")
                    self.avg_frame = gray.astype("float")
                    if self.recording and (time.time() - self.last_motion_time > RECORD_EXTENSION):
                        self.stop_recording()
                    continue

                # 3. Motion Detection
                thresh = cv2.dilate(thresh_full, None, iterations=2)
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                motion_detected = False
                for contour in contours:
                    # Adjust minimum area for the smaller resolution
                    if cv2.contourArea(contour) < (MIN_AREA_SIZE / scale_ratio):
                        continue
                    
                    motion_detected = True
                    
                    if SHOW_VIDEO_FEED:
                        (x, y, w, h) = cv2.boundingRect(contour)
                        # Scale coordinates back up for drawing on high-res frame
                        big_x = int(x * scale_ratio)
                        big_y = int(y * scale_ratio)
                        big_w = int(w * scale_ratio)
                        big_h = int(h * scale_ratio)
                        cv2.rectangle(frame, (big_x, big_y), (big_x+big_w, big_y+big_h), (0, 255, 0), 3)

                # 4. State Management
                if motion_detected:
                    self.last_motion_time = time.time()
                    self.start_recording(frame)
                
                if self.recording:
                    self.out.write(frame)
                    if time.time() - self.last_motion_time > RECORD_EXTENSION:
                        self.stop_recording()

                # 5. UI
                if SHOW_VIDEO_FEED:
                    cv2.putText(frame, f"Opt: {scale_ratio:.1f}x Change: {change_percentage:.1f}%", (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.imshow("Surveillance Feed", frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
        
        finally:
            if self.recording:
                self.stop_recording()
            self.video.release()
            cv2.destroyAllWindows()
            print("[INFO] System shutdown clean.")

if __name__ == "__main__":
    app = SurveillanceSystem()
    app.run()