import torch
from pathlib import Path
import cv2
import numpy as np
from sklearn.metrics import average_precision_score
from ultralytics import YOLO
from tracker import FastObjectTracker
from procesowanie import process_frame_fast
from filtr_Kalmana_i_ReID import Track

# Load the YOLOv5 model from the ultralytics repository
model = YOLO("yolov8n.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the directory paths for the PASCAL VOC dataset
dataset_dir = Path('C://Users//Bartek//Downloads//VOCtrainval_11-May-2012//VOCdevkit//VOC2012')
image_dir = dataset_dir / 'JPEGImages'
annotation_dir = dataset_dir / 'Annotations'

#Funkcja ładowania obrazu
def load_image(img_path):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #konwersja na inny format koloru BGR --> RGB
    return img

#Funkcja ładowania etykiet (adnotacji)
def load_labels(annotation_path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    labels = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        labels.append([xmin, ymin, xmax, ymax])
    return labels

#Załaduj kilka obrazów i etykiet
image_paths = list(image_dir.glob('*.jpg'))[:5] #Używamy pierwszych 5 obrazów
images = [load_image(img_path) for img_path in image_paths]
annotations = [load_labels(annotation_dir / (img_path.stem + '.xml')) for img_path in image_paths]

# Function to detect objects
def detect_objects(model, img):
    results = model(img)
    return results

# Perform detection on loaded images
detections = []
for img in images:
    results = model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()   # [x1, y1, x2, y2]
    confs = results[0].boxes.conf.cpu().numpy()   # confidence
    clss  = results[0].boxes.cls.cpu().numpy()    # class IDs
    dets = np.hstack([boxes, confs.reshape(-1,1), clss.reshape(-1,1)])  
    # format: [x1, y1, x2, y2, conf, cls]
    detections.append(dets)

# Print sample detection and annotation
print("Sample Detection:", detections[0])
print("Sample Annotation:", annotations[0])

# Function to compute IoU
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

from sklearn.metrics import average_precision_score

# Function to compute mAP
def compute_map(detections, annotations, iou_threshold=0.5):
    aps = []
    for det, ann in zip(detections, annotations):
        if len(ann) == 0:
            continue  # Skip images with no annotations

        tp = 0
        fp = 0
        used = [False] * len(ann)

        for d in det:
            matched = False
            for idx, a in enumerate(ann):
                if used[idx]:
                    continue  # Skip already matched ground truth
                iou = compute_iou(d[:4], a)
                if iou >= iou_threshold:
                    tp += 1
                    used[idx] = True
                    matched = True
                    break
            if not matched:
                fp += 1  # False positive if no match

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / len(ann) if len(ann) > 0 else 0
        aps.append(precision * recall)

    return np.mean(aps) if len(aps) > 0 else 0

# Calculate mAP
mAP = compute_map(detections, annotations)
print(f"Mean Average Precision (mAP): {mAP:.4f}")