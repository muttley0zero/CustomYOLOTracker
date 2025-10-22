import logging
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

def visualize_tracking(frame, tracker, frame_number=0):
    """
    Rysuje bounding boxy, ID tracka oraz confidence na klatce.
    Wyświetla tylko aktywne tracki (disappeared == 0).
    
    Args:
        frame: np.ndarray, aktualna klatka obrazu
        tracker: dict[int, Track], słownik track_id -> Track
        frame_number: int, numer klatki
    
    Returns:
        frame: klatka z naniesionymi bounding boxami i etykietami
    """
    for tid, track in tracker.items():
        # Pomijamy tracki, które chwilowo zniknęły
        if track.position is None or track.disappeared > 0:
            continue

        # Pobierz bbox (zawsze używamy przewidywanej pozycji z filtra Kalmana)
        x1, y1, x2, y2 = map(int, track.predict())

        # Kolor bbox: zielony dla osób, czerwony dla pojazdów
        if getattr(track, "class_id", None) == 0:
            color = (0, 255, 0)  # osoba
        else:
            color = (0, 0, 255)  # pojazd / inne

        # Rysowanie bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Etykieta: ID + confidence
        conf_text = f"{track.confidence:.2f}" if track.confidence is not None else "N/A"
        label = f"ID: {tid} | conf: {conf_text}"
        cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Numer klatki w rogu
    cv2.putText(frame, f"Frame: {frame_number}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame


def draw_detections(frame, results, is_yolo=False):
    """Rysuj detekcje na klatce z obsługą własnego trackera lub YOLO"""
    try:
        if is_yolo:
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()

                people_count = 0
                vehicle_count = 0

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(classes[i])

                    if class_id == 0:
                        color = (0, 255, 0)
                        people_count += 1
                        label = f"Person ID: {int(ids[i])} {confs[i]:.2f}"
                    else:
                        color = (0, 0, 255)
                        vehicle_count += 1
                        vehicle_types = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
                        vehicle_name = vehicle_types.get(class_id, "Vehicle")
                        label = f"{vehicle_name} ID: {int(ids[i])} {confs[i]:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Liczniki
                cv2.putText(frame, f"People: {people_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            # Własny tracker
            for detection in results.get('people_detections', []):
                if len(detection) >= 5:
                    cx, cy, class_id, conf, bbox = detection
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"Person: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        cv2.circle(frame, (cx, cy), 3, color, -1)

            for detection in results.get('vehicle_detections', []):
                if len(detection) >= 5:
                    cx, cy, class_id, conf, bbox = detection
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        color = (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        vehicle_types = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
                        vehicle_name = vehicle_types.get(class_id, "Vehicle")
                        label = f"{vehicle_name}: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        cv2.circle(frame, (cx, cy), 3, color, -1)

            # Liczniki
            cv2.putText(frame, f"People: {results.get('people_count', 0)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Vehicles: {results.get('vehicle_count', 0)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    except Exception as e:
        logger.error(f"Błąd w draw_detections: {e}")

    return frame
