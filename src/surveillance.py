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
LIGHT_CHANGE_THRESHOLD = 40.0  # Percentage of screen change to trigger light suppression
LEARNING_RATE = 0.05           # Speed at which the background model adapts

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
        
        print(f"System Armed. Optimized Mode: {'ON' if not SHOW_VIDEO_FEED else 'OFF'}")

    def alert_user(self):
        """Triggers a non-blocking beep alert."""
        def sound_alarm():
            winsound.Beep(2500, 1000) 
        
        if hasattr(winsound, "Beep"):
            t = threading.Thread(target=sound_alarm)
            t.daemon = True
            t.start()
        else:
            print("ALERT: Motion Detected!")

    def start_recording(self, frame):
        """Initializes the VideoWriter and starts saving frames."""
        if not self.recording:
            self.recording = True
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            if not os.path.exists("recordings"):
                os.makedirs("recordings")
            
            filename = os.path.join("recordings", f"Intruder_{timestamp}.mp4")
            
            # Using mp4v codec for .mp4 files
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Assuming standard 640x480; frame.shape can be used for dynamic sizing
            height, width = frame.shape[:2]
            self.out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
            
            print(f"[REC] Started recording: {filename}")
            self.alert_user()

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

                # Pre-processing: Convert to grayscale and blur
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                # Initialize background model if it doesn't exist
                if self.avg_frame is None:
                    print("[INFO] Starting background model...")
                    self.avg_frame = gray.astype("float")
                    continue

                # 1. Update the background model (Weighted Average)
                cv2.accumulateWeighted(gray, self.avg_frame, LEARNING_RATE)
                
                # 2. Compare current frame to the running background average
                frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg_frame))

                # 3. Sudden Light Suppression check
                thresh_full = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                changed_pixels = cv2.countNonZero(thresh_full)
                total_pixels = frame.shape[0] * frame.shape[1]
                change_percentage = (changed_pixels / total_pixels) * 100

                if change_percentage > LIGHT_CHANGE_THRESHOLD:
                    print(f"[INFO] Light change detected ({change_percentage:.1f}%). Resetting background.")
                    self.avg_frame = gray.astype("float")
                    # Stop recording if it was a false alarm from light
                    if self.recording and (time.time() - self.last_motion_time > RECORD_EXTENSION):
                        self.stop_recording()
                    continue

                # 4. Standard Motion Logic (Dilation and Contours)
                thresh = cv2.dilate(thresh_full, None, iterations=2)
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                motion_detected = False
                for contour in contours:
                    if cv2.contourArea(contour) < MIN_AREA_SIZE:
                        continue
                    
                    motion_detected = True
                    if SHOW_VIDEO_FEED:
                        (x, y, w, h) = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

                # 5. Recording state management
                if motion_detected:
                    self.last_motion_time = time.time()
                    self.start_recording(frame)
                
                if self.recording:
                    self.out.write(frame)
                    # Check if motion has stopped for longer than the extension time
                    if time.time() - self.last_motion_time > RECORD_EXTENSION:
                        self.stop_recording()

                # 6. UI Rendering
                if SHOW_VIDEO_FEED:
                    cv2.putText(frame, f"Change: {change_percentage:.1f}%", (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.imshow("Surveillance Feed", frame)
                    
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
        
        finally:
            # Cleanup resources
            if self.recording:
                self.stop_recording()
            self.video.release()
            cv2.destroyAllWindows()
            print("[INFO] System shutdown clean.")

if __name__ == "__main__":
    app = SurveillanceSystem()
    app.run()