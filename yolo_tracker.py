# yolo_tracker.py
from ultralytics import YOLO
import torch

def apply_tracker(model):
    """Konfiguruje model do używania wbudowanego trackera BoT-SORT"""
    # Ta funkcja może być pusta, ponieważ nie potrzebujemy modyfikować modelu
    # dla standardowego trackera YOLO, ale zachowujemy dla spójności.
    return model

def get_tracker_predictions(model, frame, tracker_type="botsort", persist=True):
    """Uzyskuje predykcje używając wbudowanego trackera (BoT-SORT lub ByteTrack)"""
    tracker_cfg = f"{tracker_type}.yaml"  # "botsort.yaml" albo "bytetrack.yaml"
    results = model.track(
        frame,
        conf=0.5,
        persist=persist,
        verbose=False,
        tracker=tracker_cfg
    )
    return results