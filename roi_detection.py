import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional

def get_roi(frame: np.ndarray, ball_bbox: Optional[Tuple[int, int, int, int]], player_bbox: Optional[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
if ball_bbox is None and player_bbox is None:
return (0, 0, frame.shape[1], frame.shape[0])

if ball_bbox is not None:
    x, y, w, h = ball_bbox
else:
    x, y, w, h = player_bbox

roi_size = max(w, h) * 8  # Increased ROI size
roi_x = max(0, x + w//2 - roi_size//2)
roi_y = max(0, y + h//2 - roi_size//2)
roi_w = min(roi_size, frame.shape[1] - roi_x)
roi_h = min(roi_size, frame.shape[0] - roi_y)
return (int(roi_x), int(roi_y), int(roi_w), int(roi_h))
def postprocess(frame: np.ndarray, boxes: np.ndarray, confs: np.ndarray, class_probs: np.ndarray, target_classes: List[str]) -> Dict[str, Optional[List[Tuple[int, int, int, int, float]]]]:
frameHeight, frameWidth = frame.shape[:2]
classIds, confidences, box_coords = [], [], []
target_class_ids = [classes.index(cls) for cls in target_classes]

for i in range(len(boxes)):
    box = boxes[i]
    confidence = confs[i]
    classId = int(class_probs[i])
    
    if (classId in target_class_ids) and ((confidence > confThreshold) or (classes[classId] == "sports ball" and confidence > 0.1)):
        center_x, center_y, width, height = box
        center_x = int(center_x * frameWidth)
        center_y = int(center_y * frameHeight)
        width = int(width * frameWidth)
        height = int(height * frameHeight)
        left = int(center_x - width / 2)
        top = int(center_y - height / 2)
        classIds.append(classId)
        confidences.append(float(confidence))
        box_coords.append([left, top, width, height])

indices = cv2.dnn.NMSBoxes(box_coords, confidences, confThreshold, nmsThreshold)
results: Dict[str, Optional[List[Tuple[int, int, int, int, float]]]] = {"player": None, "soccer ball": None}
for i in indices:
    i = int(i)
    box = box_coords[i]
    class_name = classes[classIds[i]]
    if class_name == "sports ball":
        results["soccer ball"] = [(*box, confidences[i])]
    elif class_name == "person":
        if results["player"] is None or confidences[i] > results["player"][0][4]:
            results["player"] = [(*box, confidences[i])]
return results
YOLO parameters
objectnessThreshold = 0.3
confThreshold = 0.4
nmsThreshold = 0.3
inpWidth = 640
inpHeight = 640

Load YOLOv8 model
try:
model = YOLO("yolov8x.pt") # Using YOLOv8x for better accuracy
except Exception as e:
print(f"Error loading YOLO model: {e}")
exit()

classesFile = "coco.names"
try:
with open(classesFile, 'rt', encoding='utf-8') as f:
classes = f.read().rstrip('\n').split('\n')
except FileNotFoundError:
print(f"Error: {classesFile} not found. Make sure it's in the correct directory.")
exit()

Process inputs
videoPath = r"C:\Users\Fathi\OneDrive\Documents\1 object tracking\opencv-detection-tracking-demo\opencv-detection-tracking-demo\soccer-neymar.mp4"
cap = cv2.VideoCapture(videoPath)

if not cap.isOpened():
print(f"Error: Could not open video file {videoPath}")
exit()

ball_bbox = None
player_bbox = None

Get screen resolution
screen_res = 1920, 1080 # Adjust this to your laptop's screen resolution
scale_width = screen_res[0] / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
scale_height = screen_res[1] / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
scale = min(scale_width, scale_height)

window_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
window_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)

cv2.namedWindow('Soccer Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Soccer Tracking', window_width, window_height)

while True:
hasFrame, frame = cap.read()
if not hasFrame:
print("End of video stream.")
break

roi = get_roi(frame, ball_bbox, player_bbox)
roi_frame = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

results = model(roi_frame)[0]
boxes = results.boxes.xywhn.cpu().numpy()
confs = results.boxes.conf.cpu().numpy()
class_probs = results.boxes.cls.cpu().numpy()

detected_objects = postprocess(roi_frame, boxes, confs, class_probs, ["person", "sports ball"])

if detected_objects["soccer ball"]:
    x, y, w, h, conf = detected_objects["soccer ball"][0]
    ball_bbox = (int(x + roi[0]), int(y + roi[1]), int(w), int(h))
    cv2.rectangle(frame, ball_bbox, (255, 178, 50), 3)
    cv2.putText(frame, f"Soccer ball {conf:.2f}", (ball_bbox[0], ball_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 178, 50), 2)
else:
    ball_bbox = None

if detected_objects["player"]:
    x, y, w, h, conf = detected_objects["player"][0]
    player_bbox = (int(x + roi[0]), int(y + roi[1]), int(w), int(h))
    cv2.rectangle(frame, player_bbox, (50, 178, 255), 3)
    cv2.putText(frame, f"Player {conf:.2f}", (player_bbox[0], player_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 178, 255), 2)
else:
    player_bbox = None

cv2.rectangle(frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)

# Resize frame to fit screen
frame = cv2.resize(frame, (window_width, window_height))

cv2.imshow("Soccer Tracking", frame)
key = cv2.waitKey(1)
if key == 27:  # ESC key
    print("ESC pressed. Exiting...")
    break
cap.release()
cv2.destroyAllWindows()
