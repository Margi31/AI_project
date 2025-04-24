from ultralytics import YOLO
import cv2
import time
import torch

# Load YOLO model
model = YOLO("yolov8s.pt")
if torch.cuda.is_available():
    model.to("cuda")
    print("🔋 Using GPU")
else:
    print("⚙️ Using CPU")

# Constants
FOCAL_LENGTH = 615  # Approximate focal length
CLOSE_THRESHOLD_CM = 150  # Threshold for red alert
KNOWN_HEIGHTS = {"person": 165}  # Known object height in cm

# Settings
FRAME_SKIP = 1  # Process every frame (increase to skip for speed)
RESIZE_WIDTH = 480  # Resize width to improve speed (optional)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Couldn't open {video_path}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        # Rotate the frame 90 degrees clockwise
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Resize frame to improve performance
        frame = cv2.resize(frame, (RESIZE_WIDTH, int(frame.shape[0] * RESIZE_WIDTH / frame.shape[1])))

        start_time = time.time()
        results = model(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id].lower()
                box_height = y2 - y1

                if label in KNOWN_HEIGHTS:
                    real_height = KNOWN_HEIGHTS[label]
                    distance_cm = (real_height * FOCAL_LENGTH) / box_height
                    distance_text = f"{distance_cm:.1f} cm"

                    color = (0, 255, 0)
                    if distance_cm < CLOSE_THRESHOLD_CM:
                        color = (0, 0, 255)
                        cv2.putText(frame, "🚨 RED ALERT!", (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{label} {conf:.2f} | {distance_text}"
                    cv2.putText(frame, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # FPS counter
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show output
        cv2.namedWindow("Smart Vision Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Smart Vision Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run on V1.mp4 only
process_video("D:/AI_Project/V1.mp4")
