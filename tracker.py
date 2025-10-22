import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from filtr_Kalmana_i_ReID import Track
import logging
import torch
import os
import cv2
import argparse
from tqdm import tqdm
from ladowanie_yolo import load_yolov8_fast
from types import SimpleNamespace
import csv
import time
from reid_loader import load_reid_model, get_embedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Konfiguracja globalna
MOT16_ROOT = r"C:\Users\Bartek\Downloads\MOT16"
OUTPUT_ROOT = os.path.join(".", "TrackEval-master", "data", "trackers", "mot_challenge", "MOT16-train")
reid_model = None  # Zainicjujemy później

class FastObjectTracker:
    def __init__(self, max_disappeared=100, iou_threshold_small=0,
                 iou_threshold_medium=0.12, iou_threshold_large=0.11,
                 reid_weight=0.41, min_confidence=0.6,
                 device='cuda'):
        self.tracks = {}
        self.all_time_track_count = 0
        self.max_disappeared = max_disappeared
        self.iou_threshold_small = iou_threshold_small
        self.iou_threshold_medium = iou_threshold_medium
        self.iou_threshold_large = iou_threshold_large
        self.reid_weight = reid_weight
        self.min_confidence = min_confidence
        self.device = device

    def reset(self):
        self.tracks = {}
        self.all_time_track_count = 0

    @staticmethod
    def _iou(boxA, boxB):
        # Zabezpieczenie na złe bboxy
        if boxA[2] <= boxA[0] or boxA[3] <= boxA[1]:
            return 0.0
        if boxB[2] <= boxB[0] or boxB[3] <= boxB[1]:
            return 0.0

        xA = np.maximum(boxA[0], boxB[0])
        yA = np.maximum(boxA[1], boxB[1])
        xB = np.minimum(boxA[2], boxB[2])
        yB = np.minimum(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

        return interArea / (boxAArea + boxBArea - interArea + 1e-6)

    @staticmethod
    def _safe_normalize(arr):
        norm = np.linalg.norm(arr, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return arr / norm

    @staticmethod
    def _cosine_distance_matrix(a, b):
        """Zwraca macierz odległości kosinusowej między a i b, bez NaN"""
        if len(a) == 0 or len(b) == 0:
            return np.ones((len(a), len(b)), dtype=np.float32)

        a_norm = FastObjectTracker._safe_normalize(a)
        b_norm = FastObjectTracker._safe_normalize(b)

        sim = np.dot(a_norm, b_norm.T)
        sim = np.clip(sim, -1.0, 1.0)  # sanity-check
        return 1.0 - sim

    def _adaptive_iou_threshold(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        size = (w + h) / 2

        if size < 30: 
            return self.iou_threshold_small
        elif size < 80: 
            return self.iou_threshold_medium
        else: 
            return self.iou_threshold_large

    def update(self, detections, embeddings=None, class_ids=None, frame_number=0, confs=None):
        embeddings = embeddings if embeddings is not None else [None] * len(detections)
        class_ids = class_ids if class_ids is not None else [None] * len(detections)
        confs = confs if confs is not None else [None] * len(detections)

        predicted_boxes, track_ids, track_embeddings = [], [], []
        for tid, track in self.tracks.items():
            predicted_boxes.append(track.predict())
            track_ids.append(tid)
            track_embeddings.append(track.embedding if track.embedding is not None 
                                   else np.zeros((128,), dtype=np.float32))

        predicted_boxes = np.array(predicted_boxes) if predicted_boxes else np.empty((0, 4))
        track_embeddings = np.array(track_embeddings) if track_embeddings else np.empty((0, 128))
        detections = np.array(detections) if detections is not None and len(detections) > 0 else np.empty((0, 4))

        matched_tracks, matched_detections = set(), set()

        if len(predicted_boxes) > 0 and len(detections) > 0:
            track_centroids = np.column_stack(((predicted_boxes[:, 0] + predicted_boxes[:, 2]) / 2,
                                               (predicted_boxes[:, 1] + predicted_boxes[:, 3]) / 2))
            det_centroids = np.column_stack(((detections[:, 0] + detections[:, 2]) / 2,
                                             (detections[:, 1] + detections[:, 3]) / 2))

            tree = cKDTree(det_centroids)
            candidate_pairs = []
            for i, tc in enumerate(track_centroids):
                idxs = tree.query_ball_point(tc, r=100)
                candidate_pairs.extend([(i, j) for j in idxs])

            if candidate_pairs:
                cost_matrix = np.full((len(predicted_boxes), len(detections)), 1e6, dtype=np.float32)
                emb_array = np.array([emb if emb is not None else np.zeros(128) for emb in embeddings], dtype=np.float32)
                reid_dist_matrix = self._cosine_distance_matrix(track_embeddings, emb_array)

                for i, j in candidate_pairs:
                    if (class_ids[j] is not None and track_ids[i] in self.tracks and 
                        self.tracks[track_ids[i]].class_id is not None and 
                        class_ids[j] != self.tracks[track_ids[i]].class_id):
                        continue

                    iou_val = self._iou(predicted_boxes[i], detections[j])
                    adaptive_threshold = self._adaptive_iou_threshold(detections[j])

                    if iou_val >= adaptive_threshold:
                        reid_cost = reid_dist_matrix[i, j]
                        iou_cost = 1 - iou_val

                        cx_track = (predicted_boxes[i, 0] + predicted_boxes[i, 2]) / 2
                        cy_track = (predicted_boxes[i, 1] + predicted_boxes[i, 3]) / 2
                        cx_det = (detections[j, 0] + detections[j, 2]) / 2
                        cy_det = (detections[j, 1] + detections[j, 3]) / 2
                        centroid_cost = np.sqrt((cx_track - cx_det) ** 2 + (cy_track - cy_det) ** 2) / 100.0
                        if not np.isfinite(centroid_cost):
                            centroid_cost = 1.0

                        cost = 0.5 * reid_cost + 0.3 * iou_cost + 0.2 * centroid_cost
                        if not np.isfinite(cost):
                            cost = 1e6

                        cost_matrix[i, j] = cost

                # sanity-check: wymuś brak NaN/inf
                cost_matrix = np.nan_to_num(cost_matrix, nan=1e6, posinf=1e6, neginf=1e6)

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < 1e5:
                        tid = track_ids[r]
                        self.tracks[tid].update(detections[c], embeddings[c], class_ids[c], 
                                                detections[c], confs[c])
                        self.tracks[tid].disappeared = 0
                        matched_tracks.add(tid)
                        matched_detections.add(c)

        to_delete = []
        for tid, track in self.tracks.items():
            if tid not in matched_tracks:
                track.disappeared += 1
                if track.disappeared > self.max_disappeared:
                    to_delete.append(tid)
        for tid in to_delete:
            self.tracks.pop(tid, None)

        for i, det in enumerate(detections):
            if i not in matched_detections and (confs[i] is None or confs[i] >= self.min_confidence):
                new_track = Track(det, frame_number, self.all_time_track_count,
                                  embedding=embeddings[i], class_id=class_ids[i],
                                  bbox=det, confidence=confs[i])
                self.tracks[self.all_time_track_count] = new_track
                self.all_time_track_count += 1

        return self.tracks

    def get_active_tracks(self):
        return {tid: t for tid, t in self.tracks.items() if t.disappeared == 0}
