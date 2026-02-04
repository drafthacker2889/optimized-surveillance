import cv2

print("Scanning for cameras... (This might take a few seconds)")

# We scan the first 5 indexes
for index in range(5):
    # CAP_DSHOW helps windows find USB cameras faster without hanging
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) 
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera Index {index} is WORKING (Resolution: {frame.shape[1]}x{frame.shape[0]})")
        else:
            print(f"⚠️ Camera Index {index} detected but cannot read frame.")
        cap.release()
    else:
        print(f"❌ Camera Index {index} not found.")

print("Scan complete.")