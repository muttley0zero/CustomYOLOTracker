import os
import cv2
import logging
from tqdm import tqdm
from tracker import FastObjectTracker
from filtr_Kalmana_i_ReID import Track
from ladowanie_yolo import load_yolov8_fast
from types import SimpleNamespace

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KITTI_ROOT = r"C:\Users\Bartek\Downloads\data_tracking_image_2\training\image_02"
OUTPUT_DIR = r"training/my_tracker/data"

# Plik globalny dla wszystkich sekwencji
GLOBAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "results.txt")

class KittiResultsWriter:
    def __init__(self, global_file=GLOBAL_OUTPUT_FILE, output_dir=OUTPUT_DIR):
        self.global_file = global_file
        self.output_dir = output_dir
        self.saved = {}  # (seq_name, frame_number) -> set(track_ids)
        os.makedirs(self.output_dir, exist_ok=True)
        open(self.global_file, 'w').close()  # Clear global file

    def add_track(self, track, frame_number, seq_name):
        """Zapisuje track do pliku globalnego i osobnego dla sekwencji (bez duplikatów)."""
        if not hasattr(track, 'class_id') or track.class_id not in [0, 2, 3, 5, 7]:
            return
        if not hasattr(track, 'position') or track.position is None:
            return

        # 🔹 Zabezpieczenie przed duplikatami ID w tej samej klatce
        key = (seq_name, frame_number)
        if key not in self.saved:
            self.saved[key] = set()
        if track.track_id in self.saved[key]:
            logger.warning(f"Duplicate track ID {track.track_id} in frame {frame_number}, sequence {seq_name}")
            return
        self.saved[key].add(track.track_id)

        score = getattr(track, 'score', 0.5)
        kitti_classes = {0: "Pedestrian", 2: "Car", 3: "Car", 5: "Car", 7: "Car"}
        obj_type = kitti_classes.get(track.class_id, "DontCare")

        x1, y1, x2, y2 = track.position
        line = (
            f"{frame_number} {track.track_id} {obj_type} -1 -1 -10 "
            f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
            f"-1000 -1000 -1000 -10 -1 -1 -1\n"
        )

        # Zapis do pliku globalnego
        with open(self.global_file, "a") as f:
            f.write(line)
            
        # Zapis do osobnego pliku sekwencji
        seq_file = os.path.join(self.output_dir, f"{seq_name}.txt")
        with open(seq_file, "a") as f:
            f.write(line)

def parse_yolo_results(results):
    """Konwertuje wyniki YOLO na listę słowników zgodnych z trackerem."""
    detections = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for bbox, score, cls in zip(boxes, scores, classes):
            if score < 0.3:
                continue
            detections.append({
                'bbox': bbox.tolist(),
                'confidence': float(score),
                'class_id': int(cls),
                'embedding': None
            })
    return detections


def process_sequence(seq_name, tracker, yolo_model, writer):
    seq_path = os.path.join(KITTI_ROOT, seq_name)
    if not os.path.exists(seq_path):
        logger.warning(f"Brak katalogu sekwencji: {seq_name}")
        return

    frame_files = sorted([f for f in os.listdir(seq_path) if f.endswith(".png")])
    if not frame_files:
        logger.warning(f"Brak klatek w sekwencji: {seq_name}")
        return

    logger.info(f"Przetwarzanie sekwencji {seq_name} ({len(frame_files)} klatek)")

    for frame_number, file_name in enumerate(tqdm(frame_files, desc=f"Seq {seq_name}")):
        frame_path = os.path.join(seq_path, file_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            logger.warning(f"Nie udało się wczytać klatki: {frame_path}")
            continue

        # Detekcja YOLO
        results = yolo_model.predict(frame)
        dets = parse_yolo_results(results)

        detections, embeddings, class_ids, bboxes = [], [], [], []
        for det in dets:
            x1, y1, x2, y2 = det['bbox']
            detections.append([x1, y1, x2, y2])
            embeddings.append(det.get('embedding', None))
            class_ids.append(det['class_id'])
            bboxes.append([x1, y1, x2, y2])

        # Aktualizacja trackera
        tracks = tracker.update(
            detections,
            embeddings=embeddings,
            class_ids=class_ids,
            bboxes=bboxes,
            frame_number=frame_number
        )

        # 🔹 Ensure no duplicate track IDs in the same frame
        seen_ids = set()
        for track_id, track in tracks.items():
            if track.position is None:
                continue
                
            # Skip if we've already seen this track ID in this frame
            if track_id in seen_ids:
                continue
            seen_ids.add(track_id)
            
            x1, y1, x2, y2 = track.position
            score = getattr(track, "score", 1.0)
            cls = getattr(track, "class_id", -1)
            
            # Create a simple track object for writing
            simple_track = SimpleNamespace()
            simple_track.track_id = track_id
            simple_track.position = [x1, y1, x2, y2]
            simple_track.class_id = cls
            simple_track.score = score
            
            writer.add_track(simple_track, frame_number, seq_name)

def main():
    logger.info("Ładowanie modelu YOLO...")
    yolo_model, device = load_yolov8_fast()
    logger.info("Model YOLO załadowany!")

    tracker = FastObjectTracker()
    writer = KittiResultsWriter()

    for seq_num in range(21):
        seq_name = f"{seq_num:04d}"
        process_sequence(seq_name, tracker, yolo_model, writer)

    logger.info("Wszystkie sekwencje zostały przetworzone!")


if __name__ == "__main__":
    main()
