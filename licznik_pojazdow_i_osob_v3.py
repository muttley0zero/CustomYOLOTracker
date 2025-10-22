import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["OPENCV_FFMPEG_THREADS"] = "1"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "threads;1|rtsp_transport;tcp"
os.environ["PYTHONWARNINGS"] = "ignore"

import cv2
import time
import logging
import torch
import argparse
import numpy as np
from wideo import VideoStream
from tracker import FastObjectTracker
from ladowanie_yolo import load_yolov8_fast
from rysowanie_detekcji import visualize_tracking
from datetime import datetime
from yolo_tracker import get_tracker_predictions
from reid_loader import load_reid_model
import csv

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_kitti_result(frame, track_id, class_id, bbox, score, seq_name="0000"):
    """Zapisuje wynik trackera w formacie KITTI MOTS."""
    kitti_classes = {0: "Pedestrian", 1: "Car", 2: "Truck", 3: "Bus", 5: "Bicycle", 7: "Motorcycle"}
    obj_type = kitti_classes.get(class_id, "DontCare")

    x1, y1, x2, y2 = bbox
    line = f"{frame} {track_id} {obj_type} -1 -1 -10 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} -1 -1 -1 -1000 -1000 -1000 -10 {score:.2f}\n"

    out_dir = os.path.join("TrackEval", "data", "tracking", "results", "my_tracker")
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, f"{seq_name}.txt")
    with open(out_file, "a") as f:
        f.write(line)

def save_active_track_to_file(track, tracker_type="custom"):
    """Zapisuje aktywny track do pliku tekstowego z oznaczeniem typu trackera"""
    filename = f"wyniki_detekcji_{tracker_type}.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Mapowanie klas na nazwy
    class_names = {
        0: "Person",
        1: "Bicycle",
        2: "Car",
        3: "Motorcycle",
        5: "Bus",
        7: "Truck"
    }
    
    class_id = getattr(track, "class_id", -1)
    obj_type = class_names.get(class_id, f"Unknown_{class_id}")
    
    try:
        with open(filename, "a", encoding='utf-8') as f:
            f.write(f"{timestamp} - {obj_type} - ID: {track.track_id}\n")
        logger.info(f"Zapisano track ID {track.track_id} do pliku ({tracker_type}).")
    except Exception as e:
        logger.error(f"Błąd zapisu do pliku: {e}")

def parse_yolo_results(results, min_confidence):
    """Parsowanie wyników YOLO z uwzględnieniem min_confidence dla wszystkich klas"""
    detections = []
    
    for r in results:
        if r.boxes is None:
            continue
            
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)
        
        # Akceptujemy wszystkie klasy (osoby i pojazdy) z odpowiednim progiem ufności
        mask = confidences >= min_confidence
        boxes_xyxy = boxes_xyxy[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        for i in range(len(boxes_xyxy)):
            detections.append({
                'bbox': boxes_xyxy[i].tolist(),
                'confidence': float(confidences[i]),
                'class_id': int(class_ids[i]),
                'embedding': None
            })
    
    return detections

def get_embeddings_batch(reid_model, device, frame, bboxes, class_ids):
    """Batch processing dla ReID tylko dla osób (klasa 0)"""
    if len(bboxes) == 0:
        return []
    
    embedding_dim = 512  # Adjust based on your ReID model output
    crops = []
    valid_indices = []
    
    for i, bbox in enumerate(bboxes):
        # Tylko dla osób (klasa 0) obliczamy embeddingi
        if class_ids[i] != 0:
            continue
            
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
        valid_indices.append(i)
    
    # Initialize with zeros for all detections
    result_embeddings = [np.zeros(embedding_dim) for _ in range(len(bboxes))]
    
    if not crops:
        return result_embeddings
    
    try:
        crops = np.stack(crops)
        crops = torch.from_numpy(crops).permute(0, 3, 1, 2).float().to(device)
        crops = crops / 255.0

        with torch.no_grad():
            embeddings = reid_model(crops).cpu().numpy()

        # Update only the valid indices
        for i, idx in enumerate(valid_indices):
            result_embeddings[idx] = embeddings[i]
    except Exception as e:
        logger.error(f"Error processing embeddings: {e}")
    
    return result_embeddings

def process_frame_custom_tracker(model, reid_model, tracker, frame, frame_count, device, tracker_params):
    """Przetwarza klatkę używając custom trackera z parametrami jak w MOT16"""
    # Detekcja YOLO z użyciem przekazanych parametrów
    results = model(
        frame, 
        conf=tracker_params.yolo_conf,
        iou=0.5, 
        classes=None,  # Wykrywamy wszystkie klasy
        verbose=False
    )
    
    # Parsowanie wyników z uwzględnieniem min_confidence
    detections = parse_yolo_results(results, tracker_params.min_confidence)

    if not detections:
        tracks = tracker.update(
            detections=np.empty((0, 4)),
            embeddings=[],
            class_ids=[],
            frame_number=frame_count,
            confs=[]
        )
    else:
        bboxes = np.array([det['bbox'] for det in detections])
        confidences = [det['confidence'] for det in detections]
        class_ids = [det['class_id'] for det in detections]

        embeddings = get_embeddings_batch(reid_model, device, frame, bboxes, class_ids)

        tracks = tracker.update(
            detections=bboxes,
            embeddings=embeddings,
            class_ids=class_ids,
            frame_number=frame_count,
            confs=confidences
        )
    
    return tracks

def add_tracker_label(frame, tracker_type, frame_width):
    """Dodaje napis z typem trackera w prawym górnym rogu"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    margin = 10
    
    if tracker_type == "custom":
        text = "Custom Tracker"
        color = (0, 255, 0)  # Zielony
    else:  # yolo
        text = "YOLO Tracker"
        color = (0, 0, 255)  # Czerwony
    
    # Oblicz rozmiar tekstu
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Pozycja tekstu - prawy górny róg
    text_x = frame_width - text_width - margin
    text_y = text_height + margin
    
    # Dodaj tekst
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
    
    return frame

def main():
    parser = argparse.ArgumentParser(description='Porównanie trackerów ReID')
    parser.add_argument('--tracker', type=str, default='custom', 
                    choices=['custom', 'botsort', 'bytetrack', 'both'],
                    help='Typ trackera do użycia: custom, botsort, bytetrack lub both')
    parser.add_argument('--video_path', type=str, default=None, help='Ścieżka do lokalnego pliku wideo. Jeśli brak, używany jest RTSP.')
    parser.add_argument('--iou_threshold_small', type=float, default=0.3)
    parser.add_argument('--iou_threshold_medium', type=float, default=0.4)
    parser.add_argument('--iou_threshold_large', type=float, default=0.5)
    parser.add_argument('--reid_weight', type=float, default=0.5)
    parser.add_argument('--min_confidence', type=float, default=0.3)
    parser.add_argument('--yolo_conf', type=float, default=0.4)
    
    args = parser.parse_args()

    use_custom_tracker = args.tracker in ['custom', 'both']
    use_yolo_tracker = args.tracker in ['botsort', 'bytetrack', 'both']
    
    # Szybkie ładowanie modelu
    logger.info("Ładowanie modelu YOLO...")
    model, device = load_yolov8_fast("yolov8n.pt", args.yolo_conf)
    
    # Dodaj ładowanie modelu ReID
    logger.info("Ładowanie modelu ReID...")
    reid_model = load_reid_model(
        model_path='C:/Users/Bartek/Downloads/licznik_osob_i_pojazdow- 15.09.2025/osnet_ain_x0_75_imagenet.pth',
        device=device)

    # Inicjalizacja trackera z parametrami jak w MOT16
    if use_custom_tracker:
        tracker = FastObjectTracker(
            max_disappeared=50,
            iou_threshold_small=args.iou_threshold_small,
            iou_threshold_medium=args.iou_threshold_medium,
            iou_threshold_large=args.iou_threshold_large,
            reid_weight=args.reid_weight,
            min_confidence=args.min_confidence
        )
    else:
        tracker = None
    
    # Inicjalizacja źródła wideo
    if args.video_path:
        logger.info(f"Użycie lokalnego wideo: {args.video_path}")
        video_source = args.video_path
    else:
        rtsp_url = "rtsp://admin:12345@150.254.16.85/live.sdp"
        logger.info(f"Użycie strumienia RTSP: {rtsp_url}")
        video_source = rtsp_url

    video_stream = VideoStream(video_source)
    video_stream.start()
    
    time.sleep(1)
    
    # Pomiary wydajności
    frame_count = 0
    fps_list = []
    last_time = time.time()
    frames_history = {}
    
    # Słownik do przechowywania zapisanych ID śledzeń dla YOLO trackera
    yolo_saved_tracks = set()
    
    # Utwórz okno
    cv2.namedWindow("Tracker Comparison", cv2.WINDOW_NORMAL)
    
    screen_width = 1280
    screen_height = 720
    
    ret, frame = video_stream.read()
    if ret and frame is not None:
        frame_height, frame_width = frame.shape[:2]
        scale_width = screen_width / frame_width
        scale_height = screen_height / frame_height
        scale = min(scale_width, scale_height)
        window_width = int(frame_width * scale)
        window_height = int(frame_height * scale)
    
    cv2.resizeWindow("Tracker Comparison", window_width, window_height)
    
    logger.info(f"Rozpoczynam porównanie trackerów: {args.tracker}")
    
    try:
        while True:
            ret, frame = video_stream.read()
            
            if not ret or frame is None:
                logger.warning("Brak klatki, ponawiam próbę...")
                time.sleep(0.1)
                continue

            frames_history[frame_count] = frame.copy()
            if len(frames_history) > 50:
                frames_history.pop(frame_count - 50, None)
            
            # Przetwarzanie w zależności od wybranego trackera
            if use_custom_tracker and use_yolo_tracker:
                # Porównanie obu trackerów
                frame1 = frame.copy()
                frame2 = frame.copy()
                
                # Custom tracker
                custom_tracks = process_frame_custom_tracker(
                    model, reid_model, tracker, frame1, frame_count, device, args
                )
                
                for track_id, track in custom_tracks.items():
                    if track.first_seen == frame_count and track.disappeared == 0 and not getattr(track, 'logged', False):
                        save_active_track_to_file(track, "custom")
                        track.logged = True
                
                # YOLO built-in tracker
                yolo_results = get_tracker_predictions(model, frame2)
                
                # Zapisz nowe śledzenia YOLO
                if yolo_results[0].boxes is not None and yolo_results[0].boxes.id is not None:
                    ids = yolo_results[0].boxes.id.cpu().numpy()
                    classes = yolo_results[0].boxes.cls.cpu().numpy()
                    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                    
                    for i, track_id in enumerate(ids):
                        if track_id not in yolo_saved_tracks:
                            # Tworzymy obiekt track z odpowiednimi atrybutami
                            from filtr_Kalmana_i_ReID import Track as CustomTrack
                            track = CustomTrack(
                                detection=boxes[i][:2],
                                frame_number=frame_count,
                                track_id=int(track_id),
                                class_id=int(classes[i]),
                                bbox=boxes[i]
                            )
                            save_active_track_to_file(track, "yolo")
                            yolo_saved_tracks.add(track_id)
                
                # Dodaj napisy do każdej klatki
                frame1 = add_tracker_label(frame1, "custom", frame_width)
                frame2 = add_tracker_label(frame2, "yolo", frame_width)
                
                # Wizualizacja custom tracker
                vis_frame1 = visualize_tracking(frame1, custom_tracks)
                
                # Wizualizacja YOLO tracker
                vis_frame2 = frame2.copy()
                if yolo_results[0].boxes is not None and yolo_results[0].boxes.id is not None:
                    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                    ids = yolo_results[0].boxes.id.cpu().numpy()
                    confs = yolo_results[0].boxes.conf.cpu().numpy()
                    classes = yolo_results[0].boxes.cls.cpu().numpy()
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(vis_frame2, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"ID: {int(ids[i])} {model.names[int(classes[i])]} {confs[i]:.2f}"
                        cv2.putText(vis_frame2, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Połącz wyniki dla wyświetlenia
                combined_frame = cv2.hconcat([vis_frame1, vis_frame2])
                display_frame = combined_frame
                
            elif use_custom_tracker:
                # Tylko custom tracker
                tracks = process_frame_custom_tracker(
                    model, reid_model, tracker, frame, frame_count, device, args
                )
                
                for track_id, track in tracks.items():
                    if track.first_seen == frame_count and track.disappeared == 0 and not getattr(track, 'logged', False):
                        save_active_track_to_file(track, "custom")
                        track.logged = True
                
                display_frame = visualize_tracking(frame.copy(), tracks, frame_count)
                display_frame = add_tracker_label(display_frame, "custom", frame_width)
                
            elif use_yolo_tracker:
                # Tylko YOLO tracker
                results = get_tracker_predictions(model, frame)

                # Zapisz nowe śledzenia YOLO
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    ids = results[0].boxes.id.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()

                    for i, track_id in enumerate(ids):
                        if track_id not in yolo_saved_tracks:
                            detection = boxes[i][:2]  # środek bboxa
                            bbox = boxes[i]
                            class_id = int(classes[i])
                            confidence = float(confs[i])

                            # Utwórz instancję klasy Track
                            from filtr_Kalmana_i_ReID import Track as CustomTrack
                            track = CustomTrack(
                                detection=detection,
                                frame_number=frame_count,
                                track_id=int(track_id),
                                class_id=class_id,
                                bbox=bbox,
                                confidence=confidence
                            )

                            save_active_track_to_file(track, args.tracker)
                            yolo_saved_tracks.add(track_id)

                # Narysuj detekcje YOLO
                display_frame = frame.copy()
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"ID: {int(ids[i])} {model.names[int(classes[i])]} {confs[i]:.2f}"
                        cv2.putText(display_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                display_frame = add_tracker_label(display_frame, args.tracker, frame_width)

            
            # Oblicz FPS
            current_time_sec = time.time()
            fps = 1.0 / (current_time_sec - last_time) if frame_count > 0 else 0
            last_time = current_time_sec
            fps_list.append(fps)
            
            # Wyświetl FPS
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Wyświetl klatkę
            cv2.imshow("Tracker Comparison", display_frame)
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    except KeyboardInterrupt:
        logger.info("Przerwano przez użytkownika")
    except Exception as e:
        logger.error(f"Błąd głównej pętli: {e}")
    finally:
        video_stream.stop()
        cv2.destroyAllWindows()
        
        if fps_list:
            avg_fps = sum(fps_list) / len(fps_list)
            logger.info(f"Średnie FPS: {avg_fps:.1f}")
        
        logger.info("Porównanie zakończone!")

if __name__ == "__main__":
    logger.info(f"PyTorch wersja: {torch.__version__}")
    logger.info(f"CUDA dostępne: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"Liczba GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    main()