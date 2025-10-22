import numpy as np
import logging
import os
from filtr_Kalmana_i_ReID import Track

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Globalna zmienna do śledzenia zapisanych tracków
saved_tracks = set()

def save_to_kitti_format(filename, frame, track_id, class_id, bbox, score):
    """
    Zapisuje detekcję w formacie KITTI MOTS.
    
    Args:
        filename (str): Pełna ścieżka do pliku wynikowego
        frame (int): Numer klatki
        track_id (int): ID śledzonego obiektu
        class_id (int): ID klasy (0: 'Pedestrian', 2: 'Car')
        bbox (list): [x1, y1, x2, y2] (lewy górny i prawy dolny róg)
        score (float): Pewność detekcji
    """
    kitti_classes = {0: "Pedestrian", 2: "Car", 3: "Car", 5: "Car", 7: "Car"}
    obj_type = kitti_classes.get(class_id, "DontCare")
    
    x1, y1, x2, y2 = bbox
    line = f"{frame} {track_id} {obj_type} -1 -1 -10 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} -1000 -1000 -1000 -10 -1 -1 -1\n"
    
    # Zapis do pliku
    with open(filename, "a") as f:
        f.write(line)

def save_active_track_to_file(track, frame_number, output_file="training/my_tracker/data/<seq_name>.txt"):
    """
    Zapisuje aktywny track w formacie KITTI do jednego wspólnego pliku.
    
    Args:
        track: Obiekt tracku (instancja klasy Track)
        frame_number (int): Numer klatki
        output_file (str): Pełna ścieżka do pliku wynikowego
    """
    # Sprawdź, czy track ma przypisany class_id
    if not hasattr(track, 'class_id') or track.class_id is None:
        return

    # Obsługujemy tylko osoby i pojazdy
    if track.class_id not in [0, 2, 3, 5, 7]:
        return

    # Upewnij się, że track ma pozycję bbox
    if not hasattr(track, 'position') or track.position is None:
        return

    score = getattr(track, 'score', 0.5)

    # Tworzenie katalogu jeśli nie istnieje
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Zapis tracka do pliku
    save_to_kitti_format(
        filename=output_file,
        frame=frame_number,
        track_id=track.track_id,
        class_id=track.class_id,
        bbox=track.position,
        score=score
    )

    if track.track_id not in saved_tracks:
        logger.info(f"Zapisano nowy track ID {track.track_id}")
        saved_tracks.add(track.track_id)

# Przykład użycia:
if __name__ == "__main__":
    example_track = Track(
        detection=[100, 200],
        frame_number=0,
        track_id=1,
        class_id=0,       # 0=Pedestrian
        position=[50, 50, 150, 200],  # [x1, y1, x2, y2]
        score=0.9
    )
    save_active_track_to_file(example_track, frame_number=0)
