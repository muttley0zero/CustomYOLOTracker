import os
import cv2
import logging
import argparse
from tqdm import tqdm
from tracker import FastObjectTracker
from filtr_Kalmana_i_ReID import Track
from ladowanie_yolo import load_yolov8_fast
from types import SimpleNamespace
import csv
import time
import numpy as np
from rysowanie_detekcji import visualize_tracking
from reid_loader import load_reid_model, get_embedding
import torch

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ścieżki do danych MOT16
MOT16_ROOT = r"C:\Users\Bartek\Downloads\MOT16"
OUTPUT_ROOT = os.path.join(".", "TrackEval-master", "data", "trackers", "mot_challenge", "MOT16-train")

reid_model = load_reid_model("checkpoint_osnet.pth", device="cuda")

class MOT16Writer:
    def __init__(self, output_root=OUTPUT_ROOT):
        self.output_root = output_root
        self.seq_files = {}
        self.buffer = {}
        self.buffer_size = 50

    def initialize_sequence(self, seq_name):
        tracker_name = "MyTracker"
        output_dir = os.path.join(self.output_root, tracker_name, "data")
        os.makedirs(output_dir, exist_ok=True)
        seq_file = os.path.join(output_dir, f"{seq_name}.txt")
        open(seq_file, 'w').close()
        self.buffer[seq_name] = []
        return seq_file

    def add_track(self, track, frame_number, seq_name, seq_file):
        if track.position is None:
            return
            
        x1, y1, x2, y2 = map(int, track.position)
        width = x2 - x1
        height = y2 - y1
        conf = getattr(track, 'score', 1.0)

        self.buffer[seq_name].append([
            frame_number + 1, track.track_id, round(x1, 2), round(y1, 2),
            round(width, 2), round(height, 2), round(conf, 2), -1, -1, -1
        ])

        if len(self.buffer[seq_name]) >= self.buffer_size:
            self.flush_buffer(seq_name, seq_file)

    def flush_buffer(self, seq_name, seq_file):
        if not self.buffer[seq_name]:
            return
            
        with open(seq_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(self.buffer[seq_name])
        self.buffer[seq_name] = []


def parse_yolo_results(results, min_confidence):
    """Zoptymalizowana wersja parsowania wyników YOLO z uwzględnieniem min_confidence"""
    detections = []
    
    for r in results:
        if r.boxes is None:
            continue
            
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)
        
        # Uwzględnienie zarówno klasy jak i progu ufności
        person_mask = (class_ids == 0) & (confidences >= min_confidence)
        boxes_xyxy = boxes_xyxy[person_mask]
        confidences = confidences[person_mask]
        class_ids = class_ids[person_mask]
        
        for i in range(len(boxes_xyxy)):
            detections.append({
                'bbox': boxes_xyxy[i].tolist(),
                'confidence': float(confidences[i]),
                'class_id': int(class_ids[i]),
                'embedding': None
            })
    
    return detections


def get_embeddings_batch(reid_model, device, frame, bboxes):
    """Batch processing dla ReID"""
    if len(bboxes) == 0:
        return []
    
    crops = []
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            crop = np.zeros((128, 64, 3), dtype=np.uint8)
        else:
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (64, 128))
        crops.append(crop)
    
    crops = np.stack(crops)
    crops = torch.from_numpy(crops).permute(0, 3, 1, 2).float().to(device)
    crops = crops / 255.0
    
    with torch.no_grad():
        embeddings = reid_model(crops).cpu().numpy()
    
    return [embeddings[i] for i in range(embeddings.shape[0])]


def process_sequence(seq_name, yolo_model, writer, tracker_params):
    global reid_model
    
    seq_path = os.path.join(MOT16_ROOT, "train", seq_name, "img1")
    if not os.path.exists(seq_path):
        logger.warning(f"Brak katalogu sekwencji: {seq_name}")
        return

    frame_files = sorted([f for f in os.listdir(seq_path) if f.endswith(".jpg")])
    if not frame_files:
        logger.warning(f"Brak klatek w sekwencji: {seq_name}")
        return

    logger.info(f"Przetwarzanie sekwencji {seq_name} ({len(frame_files)} klatek)")
    
    # Inicjalizacja trackera z uwzględnieniem min_confidence
    tracker = FastObjectTracker(
        max_disappeared=50,
        iou_threshold_small=tracker_params.iou_threshold_small,
        iou_threshold_medium=tracker_params.iou_threshold_medium,
        iou_threshold_large=tracker_params.iou_threshold_large,
        reid_weight=tracker_params.reid_weight,
        min_confidence=tracker_params.min_confidence  # Bezpośrednie użycie parametru
    )
    
    seq_file = writer.initialize_sequence(seq_name)
    fps_list = []
    last_time = time.time()
    
    # Tworzymy okno do wyświetlania wyników
    window_name = f"Tracking - {seq_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Przetwarzanie klatek
    for frame_number, file_name in enumerate(tqdm(frame_files, desc=f"Processing {seq_name}")):
        frame_path = os.path.join(seq_path, file_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # Detekcja YOLO z użyciem przekazanych parametrów
        results = yolo_model(
            frame, 
            conf=tracker_params.yolo_conf,  # Bezpośrednie użycie yolo_conf
            iou=0.5, 
            classes=[0], 
            verbose=False
        )
        
        # Parsowanie wyników z uwzględnieniem min_confidence
        detections = parse_yolo_results(results, tracker_params.min_confidence)

        if not detections:
            tracks = tracker.update(
                detections=np.empty((0, 4)),
                embeddings=[],
                class_ids=[],
                frame_number=frame_number,
                confs=[]
            )
        else:
            bboxes = np.array([det['bbox'] for det in detections])
            confidences = [det['confidence'] for det in detections]
            class_ids = [det['class_id'] for det in detections]

            embeddings = get_embeddings_batch(reid_model, "cuda", frame, bboxes)

            tracks = tracker.update(
                detections=bboxes,
                embeddings=embeddings,
                class_ids=class_ids,
                frame_number=frame_number,
                confs=confidences
            )

        # Zapis wyników
        used_ids = set()
        for track_id, track in tracks.items():
            if track_id not in used_ids:
                writer.add_track(track, frame_number, seq_name, seq_file)
                used_ids.add(track_id)

        # Wizualizacja
        if frame_number % 1 == 0:
            vis_frame = visualize_tracking(frame, tracks)
            cv2.imshow(window_name, vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        current_time = time.time()
        fps = 1 / (current_time - last_time)
        last_time = current_time
        fps_list.append(fps)

    writer.flush_buffer(seq_name, seq_file)
    cv2.destroyWindow(window_name)
    
    avg_fps = np.mean(fps_list) if fps_list else 0
    logger.info(f"Średnie FPS dla sekwencji {seq_name}: {avg_fps:.2f}")


def main():
    global reid_model
    
    parser = argparse.ArgumentParser(description='MOT16 Tracker with configurable parameters')
    parser.add_argument('--iou_threshold_small', type=float, default=0.3)
    parser.add_argument('--iou_threshold_medium', type=float, default=0.4)
    parser.add_argument('--iou_threshold_large', type=float, default=0.5)
    parser.add_argument('--reid_weight', type=float, default=0.5)
    parser.add_argument('--min_confidence', type=float, default=0.3)
    parser.add_argument('--yolo_model', type=str, default="yolov8n.pt")
    parser.add_argument('--yolo_conf', type=float, default=0.4)
    
    args = parser.parse_args()
    
    logger.info(f"Używane parametry trackera:")
    logger.info(f"  min_confidence: {args.min_confidence}")
    logger.info(f"  yolo_conf: {args.yolo_conf}")

    # Ładowanie modeli
    logger.info("Ładowanie modelu YOLO...")
    yolo_model, device = load_yolov8_fast(args.yolo_model, args.yolo_conf)
    
    logger.info("Ładowanie modelu ReID...")
    reid_model = load_reid_model("checkpoint_osnet.pth", device="cuda")
    
    writer = MOT16Writer()
    train_dir = os.path.join(MOT16_ROOT, "train")
    seq_list = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    for seq_name in seq_list:
        process_sequence(seq_name, yolo_model, writer, args)

    logger.info("Przetwarzanie zakończone!")


if __name__ == "__main__":
    main()