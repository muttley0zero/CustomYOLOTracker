import logging
import cv2
from datetime import datetime
import numpy as np
import torch
import torchvision.transforms as T
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def save_to_file(obj_type, track_id, timestamp):
    """Zapisywanie zdarzenia do pliku txt"""
    filename = "wyniki_detekcji.txt"
    try:
        with open(filename, "a", encoding='utf-8') as f:
            f.write(f"{timestamp} - {obj_type} - ID: {track_id}\n")
    except Exception as e:
        logger.error(f"Błąd zapisu do pliku: {e}")

# Przygotowanie transformacji dla ReID
reid_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 128)),  # standard dla ReID
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def get_embedding(reid_model, device, frame, bbox):
    """Wycinanie obiektu i generowanie embeddingu przez ReID"""
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    tensor = reid_transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        features = reid_model(tensor)
    return features.squeeze().cpu().numpy()

def process_frame_fast(model, reid_model, tracker, frame, frame_count, device, conf_thresh=0.6, use_reid=False):
    """Przetwarzanie klatki i aktualizacja trackera z poprawnymi bboxami, z opcjonalnym ReID"""
    results = {
        'people_count': 0,
        'vehicle_count': 0,
        'people_detections': [],
        'vehicle_detections': [],
        'tracks': {}
    }

    try:
        h0, w0 = frame.shape[:2]
        input_frame = cv2.resize(frame, (640, 640))
        input_frame = input_frame[:, :, ::-1].transpose(2, 0, 1)
        input_frame = np.ascontiguousarray(input_frame)
        input_tensor = torch.from_numpy(input_frame).to(device)
        input_tensor = input_tensor.half() if device.type == 'cuda' else input_tensor.float()
        input_tensor /= 255.0
        if input_tensor.ndimension() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(input_tensor, verbose=False)[0]

        bboxes, class_ids, confs, embeddings = [], [], [], []

        if pred is not None and pred.boxes is not None:
            boxes = pred.boxes.xyxy.cpu().numpy()
            confidences = pred.boxes.conf.cpu().numpy()
            classes = pred.boxes.cls.cpu().numpy()

            scale_x, scale_y = w0 / 640.0, h0 / 640.0

            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                if conf < conf_thresh:
                    continue

                x1, y1, x2, y2 = map(int, box)
                # przeskalowanie do oryginalnej klatki
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                detection_data = (cx, cy, int(cls), float(conf), (x1, y1, x2, y2))

                if cls == 0:
                    results['people_detections'].append(detection_data)
                elif cls in [2, 3, 5, 7]:
                    results['vehicle_detections'].append(detection_data)

                # do trackera
                bboxes.append([x1, y1, x2, y2])
                class_ids.append(int(cls))
                confs.append(float(conf))

                # Generuj embedding dla ReID jeśli używamy ReID i to osoba
                if use_reid and int(cls) == 0:
                    embedding = get_embedding(reid_model, device, frame, [x1, y1, x2, y2])
                    if embedding is not None:
                        embeddings.append(embedding)
                    else:
                        embeddings.append(np.zeros(512))  # placeholder
                else:
                    embeddings.append(np.zeros(512))  # placeholder dla nie-osób lub gdy nie używamy ReID

            #--- OGRANICZENIE DO 30 NAJLEPSZYCH DETEKCJI ---
            if bboxes:
                idxs = np.argsort(confs)[::-1][:30]  # sortuj malejąco po confidence
                bboxes   = [bboxes[i] for i in idxs]
                class_ids = [class_ids[i] for i in idxs]
                confs    = [confs[i] for i in idxs]
                embeddings = [embeddings[i] for i in idxs] if embeddings else []
        

        # Aktualizacja trackera – teraz z embeddingami
        tracker.update(
            detections=np.array(bboxes, dtype=np.float32) if bboxes else np.empty((0,4), dtype=np.float32),
            embeddings=embeddings if embeddings else [np.zeros(512)] * len(bboxes),  # zapewnij, że embeddings są tej samej długości
            class_ids=class_ids,
            confs=confs,
            frame_number=frame_count
        )

        # Wyniki licznika
        results['people_count'] = len(results['people_detections'])
        results['vehicle_count'] = len(results['vehicle_detections'])
        results['tracks'] = tracker.tracks

    except Exception as e:
        logger.error(f"Błąd w process_frame_fast: {e}", exc_info=True)

    return results