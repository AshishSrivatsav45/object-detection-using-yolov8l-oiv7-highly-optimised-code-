import cv2
import time
from ultralytics import YOLO
import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--skip", type=int, default=100, help="Number of frames to skip between detections")
args = parser.parse_args()
frame_skip = args.skip

# --- Configuration ---
ip_stream_url = "http://192.168.137.164:8080/video"  # ESP32-CAM stream
model = YOLO("yolov8l-oiv7.pt")  # Load YOLO model
announcement_interval = 5  # seconds before repeating same object

cap = cv2.VideoCapture(ip_stream_url)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

last_announced = {}
prev_announcements = set()
frame_count = 0

# --- Timing for FPS Monitoring ---
fps_start_time = time.time()
fps_frame_counter = 0

def send_to_server(announcement):
    """Send the detection result to the local Flask server."""
    url = "http://127.0.0.1:5000/update"
    data = {"text": announcement}
    try:
        response = requests.post(url, json=data)
        print("Sent to server:", response.json())
    except Exception as e:
        print("Failed to send to server:", e)

def frame_is_blank(frame, threshold=20):
    """
    Check if a frame is blank/covered by converting it to grayscale and 
    comparing the mean pixel intensity to a threshold.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()
    return mean_val < threshold

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    fps_frame_counter += 1

    # Show FPS every second
    if time.time() - fps_start_time >= 1.0:
        print("FPS:", fps_frame_counter)
        fps_frame_counter = 0
        fps_start_time = time.time()

    # Reset debouncing if the frame is nearly blank (e.g., camera is covered)
    if frame_is_blank(frame):
        prev_announcements.clear()
        last_announced.clear()

    # Skip frames for performance
    if frame_count % frame_skip != 0:
        continue

    results = model(frame)
    annotated_frame = results[0].plot()
    frame_width = frame.shape[1]
    
    # --- Aggregation ---
    aggregated = {}
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        classes = results[0].boxes.cls.cpu().numpy()   # Class indices

        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2.0

            # Determine object position in frame
            if center_x < frame_width / 3:
                position = "left"
            elif center_x < 2 * frame_width / 3:
                position = "center"
            else:
                position = "right"

            class_id = int(cls)
            label = results[0].names.get(class_id, f"class_{class_id}")
            # Skip unwanted labels
            ignored_labels = {"clothing", "apparel", "garment"}  # Add more if needed
            if label.lower() in ignored_labels:
                continue

            announcement = f"{label} on the {position}"
            if announcement in aggregated:
                aggregated[announcement] += 1
            else:
                aggregated[announcement] = 1

    # Form current announcements with counts if > 1
    current_announcements = set()
    for ann, count in aggregated.items():
        if count > 1:
            aggregated_text = f"{count} {ann}s"
        else:
            aggregated_text = ann
        current_announcements.add(aggregated_text)

    current_time = time.time()

    # --- Debouncing Logic ---
    if current_announcements != prev_announcements:
        if current_announcements:
            combined_announcement = "Detected: " + ", ".join(sorted(current_announcements))
            print("New scene:", combined_announcement)
            send_to_server(combined_announcement)
            for ann in current_announcements:
                last_announced[ann] = current_time
        prev_announcements = current_announcements.copy()
    else:
        new_announcements = []
        for announcement in current_announcements:
            last_time = last_announced.get(announcement, 0)
            if current_time - last_time > announcement_interval:
                new_announcements.append(announcement)
                last_announced[announcement] = current_time

        if new_announcements:
            combined_announcement = "Detected: " + ", ".join(sorted(new_announcements))
            print("Re-announcing:", combined_announcement)
            send_to_server(combined_announcement)

    cv2.imshow("YOLO Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
