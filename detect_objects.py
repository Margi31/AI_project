from ultralytics import YOLO
import cv2

# Load YOLOv8 model (yolov8s is small but accurate)
model = YOLO("yolov8s.pt")

# Load video file instead of webcam
video_path = "random1.mp4"
cap = cv2.VideoCapture(video_path)

# Approximate focal length in pixels
FOCAL_LENGTH = 615

# Known object heights in cm (add more as needed)
KNOWN_HEIGHTS = {
    "Person": 165,
    "Bottle": 25,
    "Pen": 12,
    "Laptop": 25,
    "Cell phone": 15,
    "Computer": 50,
    "Chair": 100,
    "Book": 20,
    "Page": 20
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Process each detection
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            box_height = y2 - y1

            # Estimate distance using known object height and focal length
            if label in KNOWN_HEIGHTS:
                real_height = KNOWN_HEIGHTS[label]
                distance_cm = (real_height * FOCAL_LENGTH) / box_height
                distance_text = f"{distance_cm:.1f} cm"
            else:
                distance_text = "?"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display label, confidence, and distance
            display_text = f"{label} {conf:.2f} | {distance_text}"
            cv2.putText(frame, display_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show output
    cv2.imshow("AI Object Detection with Distance", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
