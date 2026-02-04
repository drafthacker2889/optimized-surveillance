import cv2
import time
from datetime import datetime
import threading
import winsound
import os

# --- CONFIGURATION ---
SHOW_VIDEO_FEED = True 
MIN_AREA_SIZE = 1000 
RECORD_EXTENSION = 3 

class SurveillanceSystem:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        time.sleep(2.0) 
        
        # UPGRADE: We no longer store just "first_frame"
        # We store a floating point "average" of the background
        self.avg_frame = None 
        
        self.recording = False
        self.out = None
        self.last_motion_time = None
        
        print(f"System Armed. Optimized Mode: {'ON' if not SHOW_VIDEO_FEED else 'OFF'}")

    def alert_user(self):
        def sound_alarm():
            winsound.Beep(2500, 1000) 
        
        if hasattr(winsound, "Beep"):
            t = threading.Thread(target=sound_alarm)
            t.daemon = True
            t.start()
        else:
            print("ALERT: Motion Detected!")

    def start_recording(self, frame):
        if not self.recording:
            self.recording = True
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            if not os.path.exists("recordings"):
                os.makedirs("recordings")
            
            filename = os.path.join("recordings", f"Intruder_{timestamp}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
            
            print(f"[REC] Started recording: {filename}")
            self.alert_user()

    def stop_recording(self):
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
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                # UPGRADE: Initialize avg_frame with float precision
                if self.avg_frame is None:
                    print("[INFO] Starting background model...")
                    self.avg_frame = gray.astype("float")
                    continue

                # UPGRADE: Update the background model
                # The '0.1' is the learning rate. 
                # Higher = adapts fast (good for light changes), Lower = adapts slow.
                # 0.05 is a sweet spot for security.
                cv2.accumulateWeighted(gray, self.avg_frame, 0.05)
                
                # UPGRADE: Compare current frame to the weighted average
                frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg_frame))

                # Everything below is the same standard logic...
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)

                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                status = 0
                for contour in contours:
                    if cv2.contourArea(contour) < MIN_AREA_SIZE:
                        continue
                    status = 1
                    if SHOW_VIDEO_FEED:
                        (x, y, w, h) = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

                if status == 1:
                    self.last_motion_time = time.time()
                    self.start_recording(frame)
                
                if self.recording:
                    self.out.write(frame)
                    if time.time() - self.last_motion_time > RECORD_EXTENSION:
                        self.stop_recording()

                if SHOW_VIDEO_FEED:
                    cv2.imshow("Surveillance Feed", frame)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
        
        finally:
            if self.recording:
                self.stop_recording()
            self.video.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SurveillanceSystem()
    app.run()