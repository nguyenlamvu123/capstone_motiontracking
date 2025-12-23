import cv2
from ultralytics import YOLO
from tracker import MultiObjectTracker

model = YOLO("yolov8n.pt")
tracker = MultiObjectTracker(max_missed=15)

cap = cv2.VideoCapture("input.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((x1, y1, x2-x1, y2-y1))

    tracks = tracker.update(detections)

    for t in tracks:
        x, y, w, h = t.get_bbox()
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"ID {t.id}", (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Simple MOT", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
